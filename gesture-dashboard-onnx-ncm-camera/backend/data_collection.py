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

from .config import CLASS_NAMES


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
LABEL_TITLES = dict(zip(CLASS_NAMES, (
    "Left", "Right", "Up · index finger", "Down · index finger", "Open palm",
    "Like · thumb up", "Dorsal · fingers down", "OK",
)))
LABEL_TITLES.update({
    "fist": "Fist - mute",
    "thumb_down": "Thumb down - volume down",
})


def collection_plan():
    plan = []
    for label in CLASS_NAMES:
        views = list(VIEWS) if label in CLASS_NAMES[:4] else ["front", "yaw_left", "yaw_right", "casual"]
        instruction = {
            "left": "Extend ONLY the index finger toward your left (right side of the unmirrored preview).",
            "right": "Extend ONLY the index finger toward your right (left side of the unmirrored preview).",
            "up": "Point ONLY the index finger upward. Fold the middle, ring, and little fingers.",
            "down": "Point ONLY the index finger downward. Fold the middle, ring, and little fingers.",
            "open_palm": "Show an open palm with all five fingers visible, pointing mostly upward.",
            "like": "Give a thumbs-up with the other four fingers folded.",
            "dorsal": "Show the back of the hand with four fingers together pointing down.",
            "ok": "Join thumb and index into a circle; extend the other three fingers.",
            "fist": "Close all four fingers into the palm and keep the thumb naturally across them.",
            "thumb_down": "Point the thumb downward and keep the other four fingers folded.",
        }[label]
        for view in views:
            plan.append(dict(id=f"{label}/{view}", label=label, title=LABEL_TITLES[label],
                             view=view, instruction=instruction, variation=VIEWS[view],
                             target=12, per_hand_target=6, per_orientation_target=3))
    for view, instruction in {
        "background": "Keep both hands out of view. Vary your head position and the background.",
        "face_ear": "Keep hands down. Turn your head so eyes and ears are visible.",
        "middle_finger": "Raise only the middle finger. This is NOT a directional command.",
        "ring_finger": "Raise only the ring finger. This is NOT a directional command.",
        "little_finger": "Raise only the little finger. This is NOT a directional command.",
        "relaxed_hand": "Rest or loosely curl the hand without making a command.",
    }.items():
        plan.append(dict(id=f"no_gesture/{view}", label="no_gesture", title="No gesture",
                         view=view, instruction=instruction,
                         variation="These examples teach the model which scenes should produce no action.",
                         target=12, per_hand_target=None if view in {"background", "face_ear"} else 6,
                         per_orientation_target=None if view in {"background", "face_ear"} else 3))
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
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


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
                "class_names": [*CLASS_NAMES, "no_gesture"],
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
            try:
                with Image.open(BytesIO(jpeg)) as image:
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
        if handedness not in {"left", "right", "none", "unspecified"} or finger_orientation not in {*FINGER_ORIENTATIONS, "none", "unspecified"} or lighting not in {"normal", "dim", "backlit"} or distance not in {"near", "medium", "far"}:
            raise ValueError("Unknown recording condition.")
        with self._lock:
            entry = self._frames.get(token)
            if entry is None or time.monotonic() - entry[0] > 180:
                raise ValueError("The reviewed image expired. Capture a fresh image.")
            _, frame_id, jpeg, prediction, source = entry
            if not self._inside("participants", participant_id, "participant.json").is_file():
                raise ValueError("Create or select a participant first.")
            with Image.open(BytesIO(jpeg)) as image:
                width, height = image.size
                image.verify()
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
            temporary = image_path.with_suffix(".jpg.tmp")
            temporary.write_bytes(jpeg)
            temporary.replace(image_path)
            atomic_json(metadata_path, record)
            self._frames.pop(token, None)
            return dict(status="saved", image_path=str(image_path), sample_id=sample_id,
                        has_features=bool(record["feature_vector"]), summary=self.summary(participant_id))
