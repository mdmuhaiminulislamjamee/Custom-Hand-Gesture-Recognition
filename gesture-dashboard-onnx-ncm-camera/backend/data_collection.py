"""Local, image-first dataset collection. Labels come from the operator.

Per-image JSON is authoritative; exports can be rebuilt without losing raw
images when the current detector misses a hand. No prediction is a save gate.
"""
from __future__ import annotations

import base64
import copy
import hashlib
import json
import os
import re
import secrets
import threading
import time
from collections import OrderedDict
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path, PurePosixPath

from PIL import Image

from .config import ACTIVE_CLASS_NAMES, CLASS_NAMES


DEFAULT_COLLECTION_ROOT = Path(os.getenv("GESTURE_COLLECTION_DIR", "D:/Data-Collection"))
FINGER_ORIENTATIONS = {"toward_camera", "away_from_camera"}
VIEWS = {
    "front": "Face the hand toward the camera. Keep the wrist visible.",
    "yaw_left": "Turn the hand slightly left while keeping the gesture's direction.",
    "yaw_right": "Turn the hand slightly right while keeping the gesture's direction.",
    "pitch_toward": "Tilt the fingers slightly toward the camera.",
    "pitch_away": "Tilt the fingers slightly away from the camera.",
    "roll_left": "Relax the wrist with a small left tilt; keep the intended direction clear.",
    "roll_right": "Relax the wrist with a small right tilt; keep the intended direction clear.",
    "casual": "Place the hand naturally. Vary its position and distance between photos.",
}
LABEL_TITLES: dict[str, str] = dict(zip(CLASS_NAMES, (
    "Left", "Right", "Up · index finger", "Down · index finger", "Open palm",
    "Like · thumb up", "Dorsal · fingers down", "OK",
)))
LABEL_TITLES.update({
    "fist": "Fist - mute",
    "thumb_down": "Thumb down - volume down",
    "peace": "Peace - turn 90 degrees",
    "rock": "Rock - backup",
    "stop_inverted": "Stop inverted - return to main position",
})


# Keep existing step IDs stable: saved images and their JSON sidecars use them.
# Context and detector cases are reviewed raw images, but cannot safely train a
# classifier whose only input is one hand's 76 landmark features.
NEGATIVE_GROUPS = (
    ("Dorsal hard negatives", "single", "hand_classifier", "The old Dorsal output is retired. Capture back-of-hand poses outside the valid Stop inverted range; vary distance, background and lighting for both hands.", tuple(
        (f"dorsal_{view}", "Show the back of the hand outside XY -145 to -35 degrees, including near-horizontal and downward poses. Stop inverted is valid only within that upright band, with fingers together or spread; do not label poses inside the band negative.")
        for view in ("front", "yaw_left", "yaw_right", "casual")
    )),
    ("No hand", "none", "detector_only", "Vary the person, background, lighting and camera position; keep every hand out of view.", (
        ("background", "Keep both hands out of view. Vary your head position and the background."),
        ("face_ear", "Keep hands out of view while your face and ears are visible."),
        ("torso_no_hands", "Show your torso and shoulders with both hands outside the frame."),
        ("desk_clutter_no_hands", "Show a desk with ordinary objects but no visible hands."),
    )),
    ("Unrelated hand shapes", "single", "hand_classifier", "Use both hands and vary yaw, pitch, roll, distance and lighting. Keep the pose distinct from every command.", (
        ("middle_finger", "Raise only the middle finger. This is not a directional command."),
        ("ring_finger", "Raise only the ring finger. This is not a directional command."),
        ("little_finger", "Raise only the little finger. This is not a directional command."),
        ("relaxed_hand", "Show a loose, partly curled hand that is clearly neither a closed fist nor four straight Dorsal fingers."),
        ("thumb_little_sign", "Extend thumb and little finger while folding the others. This retired sign is not a command."),
        ("partial_fist", "Partly curl the fingers without closing a full Fist or opening the palm."),
        ("downward_fist", "Close the fist and tilt its wrist-to-knuckle direction below horizontal, from right toward down or left toward down. Include fingers toward the camera and a 30-degree forward pitch; all these lower angles are no_gesture."),
        ("bent_index", "Bend the index finger halfway while keeping the others relaxed; do not point in a command direction."),
        ("three_fingers", "Extend three fingers with the others folded; avoid the two-finger Peace and four-finger Dorsal shapes."),
        ("four_fingers_spread", "Show four loosely curled or downward spread fingers with the thumb tucked. Avoid valid upright Stop inverted and full Open Palm."),
        ("thumb_sideways", "Show a sideways thumb with the other fingers loosely curled; avoid Like and Thumb Down."),
        ("almost_ok", "Bring thumb and index near each other without closing an OK circle; keep the other fingers relaxed."),
        ("crossed_fingers", "Cross index and middle fingers; keep the remaining fingers folded. This is not Peace."),
    )),
    ("Near-command shapes", "single", "hand_classifier", "Make the stated difference from the command obvious in the saved frame. Retake any image that could fairly be labeled as a real command.", (
        ("index_middle_together", "Extend index and middle fingers close together. Straight touching fingers can be valid Peace; retain this ambiguous boundary for review, excluded from classifier negative training."),
        ("thumb_up_half", "Raise the thumb only partway or diagonally while the other fingers remain loose; do not form Like."),
        ("thumb_down_half", "Lower the thumb only partway or diagonally while the other fingers remain loose; do not form Thumb Down."),
        ("palm_thumb_tucked", "Show the palm with four fingers extended but the thumb clearly tucked; this is not a five-finger Open Palm."),
        ("palm_two_fingers_curled", "Show a partly open palm with two clearly curled fingers; do not form Peace or Open Palm."),
        ("ok_circle_fingers_folded", "Make a thumb-index circle but curl at least two of the other fingers; this is not the three-finger-extended OK command."),
        ("rock_middle_partly_open", "Extend index and little finger while leaving the middle finger clearly partly open; avoid the folded-middle Rock shape."),
        ("dorsal_fingers_spread", "Show spread back-of-hand fingers outside XY -145 to -35 degrees, including near-horizontal and downward poses. Spread Stop inverted inside that upright band is valid and must not be labeled negative."),
        ("dorsal_one_finger_curled", "Show the back of the hand with one of the four fingers clearly curled; do not make Dorsal."),
        ("fist_one_finger_loose", "Curl most fingers but leave the ring or little finger clearly partly open; do not make a closed Fist."),
    )),
    ("Resting Dorsal", "single", "context_only", "Capture ordinary rest, not an intentional command. Include shoulder, elbow and wrist when possible; vary seated and standing views.", (
        ("dorsal_body_not_visible", "Show the back of the hand with the body cropped and fingers outside XY -145 to -35 degrees. Stop inverted inside that upright band remains valid even without the body visible."),
        ("standing_arm_hanging", "Stand naturally with one arm hanging and the back of its relaxed hand toward the camera."),
        ("standing_hand_on_thigh", "Stand with a relaxed hand resting against the hip or thigh, fingers dropping downward."),
        ("seated_hand_on_lap", "Sit with a relaxed hand lying on the lap, back of hand visible."),
        ("seated_hand_on_desk", "Sit with forearm resting on the desk and hand relaxed near the desk edge."),
        ("seated_fingers_over_edge", "Let the relaxed fingers hang over the desk edge as in the reported false Dorsal example."),
        ("seated_folded_arm_dropped", "Sit with elbow bent beside the body and the relaxed hand hanging downward."),
        ("chin_rest_dorsal", "Rest the chin or cheek lightly on a relaxed hand, with the back of the hand visible."),
        ("hand_on_keyboard", "Rest the hand naturally over a keyboard without making a command."),
        ("hand_on_mouse", "Rest the hand naturally on a mouse without making a command."),
    )),
    ("Resting Fist", "single", "context_only", "Capture ordinary rest or grasping, not an intentional command. A closed hand alone can look identical to Fist.", (
        ("fist_fingers_away", "Close the hand with its fingers facing away from the camera. This opposite-facing view is no_gesture; retain the full image and orientation for review."),
        ("standing_loose_fist", "Stand with a loosely closed hand hanging at the side."),
        ("seated_fist_on_lap", "Sit with a naturally closed hand resting on the lap."),
        ("seated_fist_on_desk", "Sit with a naturally closed hand resting on the desk."),
        ("fist_near_chest_rest", "Keep a relaxed closed hand near the chest without presenting a command."),
        ("gripping_phone", "Hold a phone naturally; the fingers may resemble a fist."),
        ("gripping_cup", "Hold a cup or bottle naturally; do not present a Fist command."),
        ("gripping_bag", "Hold a bag handle or strap naturally with curled fingers."),
        ("holding_small_object", "Hold a small everyday object with curled fingers; keep the object visible."),
    )),
    ("Incidental pointing", "single", "context_only", "Capture everyday index-finger use, not an intended Left, Right, Up or Down command. Keep the surrounding activity visible.", (
        ("index_resting_on_desk", "Rest or tap the index finger on the desk while the arm remains supported."),
        ("index_touching_face", "Touch or scratch the face with the index finger without signaling a command."),
        ("index_holding_pen", "Hold or use a pen with the index finger extended along it."),
        ("index_partly_extended", "Leave the index finger only partly extended during a relaxed hand movement."),
        ("pointing_during_reach", "Reach for an object with an incidental extended index finger; keep the object visible."),
        ("index_near_camera", "Move an incidental extended index finger toward the camera during ordinary activity."),
    )),
    ("Occlusion and motion", "single", "detector_only", "Save raw difficult frames for detector and tracking evaluation. A partly observed hand may not provide trustworthy classifier features.", (
        ("partial_hand_frame", "Let part of a relaxed hand leave the image edge; do not form a command."),
        ("fingers_hidden_object", "Hide some relaxed fingers behind an object while keeping part of the hand visible."),
        ("hand_under_sleeve", "Let a sleeve partly cover the wrist or hand while the hand remains relaxed."),
        ("hand_over_face", "Place a relaxed hand partly in front of the face without making a command."),
        ("hand_turning", "Capture a relaxed hand during a wrist turn between poses, away from a stable command."),
        ("moving_hand_blur", "Move a relaxed hand during capture to include realistic motion blur."),
    )),
    ("Two hands", "both", "context_only", "Show both hands. These raw images test hand selection and context; the single-hand classifier cannot learn the two-hand relationship.", (
        ("both_hands_relaxed", "Keep both hands relaxed and visible, seated or standing."),
        ("both_hands_same_distance", "Show two relaxed hands at approximately the same camera distance."),
        ("both_hands_different_distances", "Show two relaxed hands with one clearly nearer the camera."),
        ("hands_crossed", "Cross or overlap two relaxed hands without forming a command."),
        ("one_hand_behind_other", "Partly hide one relaxed hand behind the other."),
    )),
)


def collection_plan():
    plan = []
    for label in ACTIVE_CLASS_NAMES:
        views = list(VIEWS) if label in {*CLASS_NAMES[:4], "peace", "rock"} else ["front", "yaw_left", "yaw_right", "casual"]
        instruction = {
            "left": "Extend ONLY the index finger toward your left (right side of the unmirrored preview). Use either hand; vary palm, back and edge views, thumb position and a small upward/downward tilt while keeping the index primarily horizontal. Fold the middle, ring and little fingers.",
            "right": "Extend ONLY the index finger toward your right (left side of the unmirrored preview). Use either hand; vary palm, back and edge views, thumb position and a small upward/downward tilt while keeping the index primarily horizontal. Fold the middle, ring and little fingers.",
            "up": "Point ONLY the index finger upward. Fold the middle, ring, and little fingers.",
            "down": "Point ONLY the index finger downward. Fold the middle, ring, and little fingers.",
            "open_palm": "Show your palm with all five fingers visible. Start upright and tilt at most 70 degrees toward either side: image XY -160 to -20 degrees, including endpoints. Beyond either limit is no_gesture. Vary the wrist angle within this range across captures.",
            "like": "Give a thumbs-up with the other four fingers folded.",
            "ok": "Join thumb and index into a circle; extend the other three fingers.",
            "fist": "Close all four fingers into the palm and keep the thumb across them. Sweep from horizontal left through up to horizontal right with both hands. Include a 30-degree pitch toward the camera. Fists below horizontal are no_gesture.",
            "thumb_down": "Point the thumb downward and keep the other four fingers folded.",
            "peace": "Extend index and middle fingers as a V. Fold ring and little fingers; vary wrist direction.",
            "rock": "Extend index and little finger. Fold middle and ring fingers. The thumb may be folded or extended.",
            "stop_inverted": "Show the back of the hand with four fingers extended, together or spread. Start upright and tilt at most 55 degrees toward either side: image XY -145 to -35 degrees, including endpoints. Beyond either limit is no_gesture. Use both hands in separate captures. This returns to the main position.",
        }[label]
        for view in views:
            plan.append(dict(id=f"{label}/{view}", label=label, title=LABEL_TITLES[label],
                             view=view, instruction=instruction, variation=VIEWS[view],
                             group="Commands", hand_mode="single", training_use="hand_classifier",
                             allowed_orientations=["toward_camera"] if label == "fist" else sorted(FINGER_ORIENTATIONS),
                             target=12, per_hand_target=6, per_orientation_target=6 if label == "fist" else 3))
    for group, hand_mode, training_use, variation, cases in NEGATIVE_GROUPS:
        for view, instruction in cases:
            single = hand_mode == "single"
            plan.append(dict(id=f"no_gesture/{view}", label="no_gesture", title="No gesture",
                             view=view, instruction=instruction, variation=variation,
                             group=group, hand_mode=hand_mode,
                             training_use=("hand_classifier" if view == "dorsal_body_not_visible" else
                                           "context_only" if view in {"index_middle_together", "downward_fist", "fist_fingers_away"} else training_use),
                             allowed_orientations=(["away_from_camera"] if view == "fist_fingers_away"
                                                   else sorted(FINGER_ORIENTATIONS)) if single else [],
                             target=12, per_hand_target=6 if single else None,
                             per_orientation_target=(6 if view == "fist_fingers_away" else 3) if single else None))
    return plan


def identifier(value, description):
    value = str(value).strip().lower()
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,47}", value):
        raise ValueError(f"{description}: use 1–48 letters, numbers, hyphens, or underscores.")
    if value.upper() in {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)), *(f"LPT{i}" for i in range(1, 10))}:
        raise ValueError(f"{description} is a reserved Windows name.")
    return value


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    # Include a random component so two store instances cannot overwrite one
    # another's temporary file before either one replaces the destination.
    temporary = path.with_name(f".{path.name}.{secrets.token_hex(8)}.tmp")
    try:
        temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


class CollectionStore:
    def __init__(self, root=DEFAULT_COLLECTION_ROOT):
        self.root = Path(root).resolve()
        self._lock = threading.RLock()
        self._frames = OrderedDict()
        self._latest = None
        self._latest_at = 0.0

    def _inside(self, *parts):
        path = self.root.joinpath(*parts).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError("Collection path escapes the configured storage folder.")
        return path

    def initialize(self):
        with self._lock:
            self.root.mkdir(parents=True, exist_ok=True)
            self._inside("participants").mkdir(exist_ok=True)
            descriptor_path = self._inside("dataset.json")
            required = {
                "schema_version": 2,
                "type": "gesture_image_collection",
                "class_names": [*ACTIVE_CLASS_NAMES, "no_gesture"],
                "direction_semantics": "NCM unmirrored: image +x is left, -x is right, -y is up",
                "labels": "Human-confirmed; model predictions are diagnostics only",
                "split_policy": "Keep each participant in one split during retraining",
                "raw_images": "Original NCM or webcam JPEGs, no overlays or augmentation",
            }
            if descriptor_path.exists():
                try:
                    descriptor = json.loads(descriptor_path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError) as error:
                    raise ValueError("Collection dataset descriptor is not valid JSON.") from error
                if not isinstance(descriptor, dict):
                    raise ValueError("Collection dataset descriptor must be a JSON object.")
                migrated = {**descriptor, **required}
                if migrated != descriptor:
                    atomic_json(descriptor_path, migrated)
            else:
                atomic_json(descriptor_path, required)

    def observe(self, frame_id, jpeg, result, source="ncm"):
        # One immutable image/result pair avoids attaching the next camera frame
        # to the previous frame's landmarks. No disk I/O in the inference path.
        with self._lock:
            if source not in {"ncm", "webcam"}:
                raise ValueError("Unknown collection camera source.")
            self._latest = (
                int(frame_id), bytes(jpeg), copy.deepcopy(result), source
            )
            self._latest_at = time.monotonic()

    def _freeze_entry(self, frame_id, jpeg, result, source):
        if source not in {"ncm", "webcam"}:
            raise ValueError("Unknown collection camera source.")
        token = secrets.token_hex(16)
        self._frames[token] = (
            time.monotonic(), int(frame_id), bytes(jpeg), copy.deepcopy(result), source
        )
        while len(self._frames) > 12:
            self._frames.popitem(last=False)
        return dict(
            token=token,
            frame_id=int(frame_id),
            source=source,
            image_url="data:image/jpeg;base64," + base64.b64encode(jpeg).decode("ascii"),
            prediction=result.get("runtime_prediction", "no_gesture"),
            reason=result.get("action_reason", result.get("message", "")),
            landmarks=result.get("landmarks") or [],
            world_landmarks=result.get("world_landmarks") or [],
            plane_angles=result.get("plane_angles"),
            has_features=len(result.get("feature_vector") or []) == 76,
            issues=(result.get("quality") or {}).get("issues", []),
        )

    def freeze_bytes(self, frame_id, jpeg, result, source="webcam"):
        with self._lock:
            if source not in {"ncm", "webcam"}:
                raise ValueError("Unknown collection camera source.")
            try:
                with Image.open(BytesIO(jpeg)) as image:
                    if image.format != "JPEG":
                        raise ValueError("The uploaded webcam frame must be a JPEG image.")
                    image.verify()
            except Exception as error:
                raise ValueError("The uploaded webcam frame is not a valid image.") from error
            return self._freeze_entry(frame_id, jpeg, result, source)

    def freeze(self):
        with self._lock:
            if self._latest is None or time.monotonic() - self._latest_at > 2.0:
                raise ValueError("No fresh camera frame. Connect the camera and wait for the preview.")
            frame_id, jpeg, result, source = self._latest
            return self._freeze_entry(frame_id, jpeg, result, source)

    def add_participant(self, participant_id):
        participant_id = identifier(participant_id, "Participant ID")
        with self._lock:
            self.initialize()
            path = self._inside("participants", participant_id, "participant.json")
            if not path.exists():
                atomic_json(path, dict(participant_id=participant_id,
                                       created_at_utc=datetime.now(timezone.utc).isoformat()))
            for label in [*CLASS_NAMES, "no_gesture"]:
                self._inside("participants", participant_id, label).mkdir(exist_ok=True)
            return self.summary(participant_id)

    def summary(self, participant_id=None):
        with self._lock:
            participants = []
            if self._inside("participants").exists():
                participants = sorted(p.parent.name for p in self._inside("participants").glob("*/participant.json"))
            counts, counts_by_hand, counts_by_hand_orientation, recent = {}, {}, {}, []
            if participant_id:
                participant_id = identifier(participant_id, "Participant ID")
                base = self._inside("participants", participant_id)
                for path in base.glob("*/*/*/*.json"):
                    if not path.resolve().is_relative_to(self.root):
                        continue
                    record = json.loads(path.read_text(encoding="utf-8"))
                    key = record["step_id"]
                    counts[key] = counts.get(key, 0) + 1
                    hand = record.get("handedness", "unspecified")
                    hand_counts = counts_by_hand.setdefault(key, {})
                    hand_counts[hand] = hand_counts.get(hand, 0) + 1
                    orientation = record.get("finger_orientation", "unspecified")
                    orientation_counts = counts_by_hand_orientation.setdefault(key, {}).setdefault(hand, {})
                    orientation_counts[orientation] = orientation_counts.get(orientation, 0) + 1
                    recent.append(dict(sample_id=record["sample_id"], label=record["label"],
                                       view=record["view"], image_path=record["image_path"],
                                       handedness=hand, finger_orientation=orientation,
                                       created_at_utc=record["created_at_utc"]))
            return dict(root=str(self.root), available=self.root.is_dir(), participants=participants,
                        participant_id=participant_id, counts=counts, counts_by_hand=counts_by_hand,
                        counts_by_hand_orientation=counts_by_hand_orientation,
                        total=sum(counts.values()),
                        recent=sorted(recent, key=lambda r: r["created_at_utc"], reverse=True)[:5],
                        plan=collection_plan())

    def _metadata_from_ref(self, participant_id, sample_ref):
        reference = PurePosixPath(str(sample_ref))
        if reference.is_absolute() or ".." in reference.parts or reference.suffix.lower() != ".json":
            raise ValueError("Invalid selected image reference.")
        metadata_path = self._inside(*reference.parts)
        participant_base = self._inside("participants", participant_id)
        if not metadata_path.is_relative_to(participant_base):
            raise ValueError("Selected image does not belong to this participant.")
        if not metadata_path.is_file():
            raise ValueError("A selected image no longer exists.")
        return metadata_path

    def list_samples(self, participant_id, step_id=None):
        participant_id = identifier(participant_id, "Participant ID")
        if step_id is not None and not any(step["id"] == step_id for step in collection_plan()):
            raise ValueError("Unknown collection prompt.")
        with self._lock:
            base = self._inside("participants", participant_id)
            if not self._inside("participants", participant_id, "participant.json").is_file():
                raise ValueError("Create or select a participant first.")
            samples = []
            for metadata_path in base.glob("*/*/*/*.json"):
                if not metadata_path.resolve().is_relative_to(base):
                    continue
                record = json.loads(metadata_path.read_text(encoding="utf-8"))
                if step_id is not None and record.get("step_id") != step_id:
                    continue
                samples.append(dict(
                    sample_ref=metadata_path.relative_to(self.root).as_posix(),
                    sample_id=record["sample_id"], step_id=record["step_id"],
                    label=record["label"], view=record["view"],
                    handedness=record.get("handedness", "unspecified"),
                    finger_orientation=record.get("finger_orientation", "unspecified"),
                    image_path=record["image_path"], created_at_utc=record["created_at_utc"],
                ))
            return dict(samples=sorted(samples, key=lambda item: item["created_at_utc"], reverse=True))

    def sample_image(self, participant_id, sample_ref):
        participant_id = identifier(participant_id, "Participant ID")
        with self._lock:
            metadata_path = self._metadata_from_ref(participant_id, sample_ref)
            image_path = metadata_path.with_suffix(".jpg")
            if not image_path.is_file():
                raise ValueError("The selected image file is missing.")
            return image_path.read_bytes()

    def delete_selected(self, participant_id, sample_refs):
        participant_id = identifier(participant_id, "Participant ID")
        references = list(dict.fromkeys(str(reference) for reference in sample_refs))
        if not references or len(references) > 100:
            raise ValueError("Select between 1 and 100 images to delete.")
        with self._lock:
            selected = []
            for reference in references:
                metadata_path = self._metadata_from_ref(participant_id, reference)
                record = json.loads(metadata_path.read_text(encoding="utf-8"))
                selected.append((metadata_path, metadata_path.with_suffix(".jpg"), record))
            deleted = []
            for metadata_path, image_path, record in selected:
                if image_path.is_file():
                    image_path.unlink()
                metadata_path.unlink()
                deleted.append(dict(
                    sample_ref=metadata_path.relative_to(self.root).as_posix(),
                    sample_id=record.get("sample_id"), label=record.get("label"),
                    view=record.get("view"), handedness=record.get("handedness", "unspecified"),
                    finger_orientation=record.get("finger_orientation", "unspecified"),
                    image_path=record.get("image_path"),
                ))
            return dict(status="deleted", deleted=deleted, summary=self.summary(participant_id))

    def delete_last(self, participant_id):
        participant_id = identifier(participant_id, "Participant ID")
        with self._lock:
            base = self._inside("participants", participant_id)
            if not self._inside("participants", participant_id, "participant.json").is_file():
                raise ValueError("Create or select a participant first.")
            latest = None
            for metadata_path in base.glob("*/*/*/*.json"):
                resolved_metadata = metadata_path.resolve()
                if not resolved_metadata.is_relative_to(base) or resolved_metadata.suffix.lower() != ".json":
                    continue
                try:
                    record = json.loads(resolved_metadata.read_text(encoding="utf-8"))
                    created = datetime.fromisoformat(record["created_at_utc"])
                except (OSError, UnicodeError, json.JSONDecodeError, KeyError, ValueError):
                    continue
                candidate = (created, resolved_metadata, record)
                if latest is None or candidate[0] > latest[0]:
                    latest = candidate
            if latest is None:
                raise ValueError("This participant has no saved images to delete.")
            _, metadata_path, record = latest
            image_path = metadata_path.with_suffix(".jpg").resolve()
            if not image_path.is_relative_to(base):
                raise ValueError("The last image path is outside this participant's folder.")
            if image_path.is_file():
                image_path.unlink()
            metadata_path.unlink()
            return dict(status="deleted", deleted=dict(
                            sample_id=record.get("sample_id"), label=record.get("label"),
                            view=record.get("view"), handedness=record.get("handedness", "unspecified"),
                            finger_orientation=record.get("finger_orientation", "unspecified"),
                            image_path=record.get("image_path")),
                        summary=self.summary(participant_id))

    def save(self, *, token, participant_id, session_id, step_id, handedness="unspecified",
             finger_orientation="unspecified",
             lighting="normal", distance="medium", note=""):
        participant_id = identifier(participant_id, "Participant ID")
        session_id = identifier(session_id, "Session ID")
        step = next((s for s in collection_plan() if s["id"] == step_id), None)
        if step is None:
            raise ValueError("Unknown collection prompt.")
        if step["label"] == "fist" and finger_orientation == "away_from_camera":
            raise ValueError("Opposite-facing fists are no_gesture. Use the reverse-facing Fist negative prompt.")
        if step["view"] == "fist_fingers_away" and finger_orientation == "toward_camera":
            raise ValueError("Reverse-facing Fist negatives must have their fingers away from the camera.")
        if handedness not in {"left", "right", "both", "none", "unspecified"} or finger_orientation not in {*FINGER_ORIENTATIONS, "none", "unspecified"} or lighting not in {"normal", "dim", "backlit"} or distance not in {"near", "medium", "far"}:
            raise ValueError("Unknown recording condition.")
        with self._lock:
            entry = self._frames.get(token)
            if entry is None or time.monotonic() - entry[0] > 180:
                raise ValueError("The reviewed image expired. Capture a fresh image.")
            _, frame_id, jpeg, prediction, source = entry
            if not self._inside("participants", participant_id, "participant.json").is_file():
                raise ValueError("Create or select a participant first.")
            try:
                with Image.open(BytesIO(jpeg)) as image:
                    if image.format != "JPEG":
                        raise ValueError("The frozen camera frame is not a JPEG image.")
                    width, height = image.size
                    image.verify()
            except Exception as error:
                raise ValueError("The frozen camera frame is not a valid JPEG image.") from error
            digest = hashlib.sha256(jpeg).hexdigest()
            sample_id = digest[:24]
            directory = self._inside("participants", participant_id, step["label"], step["view"], session_id)
            metadata_path = directory / f"{sample_id}.json"
            if metadata_path.exists():
                raise ValueError("This image is already saved in this session. Move the hand and capture again.")
            image_path = directory / f"{sample_id}.jpg"
            directory.mkdir(parents=True, exist_ok=True)
            captured_at = datetime.now(timezone.utc).isoformat()
            record = dict(schema_version=2, sample_id=sample_id, participant_id=participant_id,
                          session_id=session_id, step_id=step_id, label=step["label"], view=step["view"],
                          group=step["group"], hand_mode=step["hand_mode"], training_use=step["training_use"],
                          created_at_utc=captured_at, source=source, frame_id=frame_id,
                          image_path=image_path.relative_to(self.root).as_posix(), image_sha256=digest,
                          width=width, height=height, handedness=handedness,
                          finger_orientation=finger_orientation, lighting=lighting,
                          distance=distance, note=str(note).strip()[:500], reviewed=True,
                          feature_vector=prediction.get("feature_vector"), landmarks=prediction.get("landmarks"),
                          feature_space="smoothed_landmarks_in_center_92_percent_square",
                          prediction={k: prediction.get(k) for k in (
                              "raw_prediction", "runtime_prediction", "known_gesture_mass", "action_reason", "quality")})
            # Bytes are the frozen raw camera JPEG, never the browser's display.
            temporary = image_path.with_name(f".{image_path.name}.{secrets.token_hex(8)}.tmp")
            try:
                temporary.write_bytes(jpeg)
                temporary.replace(image_path)
                atomic_json(metadata_path, record)
            except Exception:
                # A sidecar is required for a sample to be discoverable. Do
                # not leave an unindexed image behind if writing it fails.
                image_path.unlink(missing_ok=True)
                raise
            finally:
                temporary.unlink(missing_ok=True)
            self._frames.pop(token, None)
            return dict(status="saved", image_path=str(image_path), sample_id=sample_id,
                        has_features=bool(record["feature_vector"]), summary=self.summary(participant_id))
