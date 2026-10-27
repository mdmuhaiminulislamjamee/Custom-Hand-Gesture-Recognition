"""Legacy offline RGB contact helper; not used by live HandDetector.

Face Mesh supplies the surface outline. Depth agreement uses only landmarks
from ONE Pose Landmarker world frame; independent hand/face z origins are
never subtracted. All caches and contact holds belong to the caller's stream.
"""
from dataclasses import dataclass
import hashlib
from pathlib import Path
import time

import numpy as np


FACE_MODEL_NAME = "face_landmarker.task"
FACE_MODEL_SHA256 = "64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff"
CONTACT_REASON = "face_contact_estimated"
CONTACT_PENDING_REASON = "face_contact_pending"
FACE_OVERLAP_REASON = "face_overlap_depth_unavailable"
FACE_HAND_REASON = "face_detected_as_hand"
FACE_VETO_REASONS = {CONTACT_REASON, CONTACT_PENDING_REASON, FACE_OVERLAP_REASON, FACE_HAND_REASON}
FACE_OVAL = (10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
             397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
             172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109)
FACE_HOLES = (
    (33, 160, 158, 133, 153, 144),
    (362, 385, 387, 263, 373, 380),
    (78, 81, 13, 311, 308, 402, 14, 178),
)


@dataclass(frozen=True)
class ContactPolicy:
    confirmation_seconds: float = .20
    release_seconds: float = .15
    maximum_sample_gap: float = .30
    # Dimensionless fractions of the observed head width, not centimetres.
    maximum_depth_gap: float = .20
    # Require a larger gap to establish separation than to estimate contact;
    # the uncertain band protects against noisy pose z at a touching hand.
    minimum_clear_depth_gap: float = .50
    maximum_drift: float = .08
    maximum_speed: float = .60


class FaceMeshGuard:
    """Bounded IMAGE inference; cached observations never cross sessions."""

    def __init__(self, model_path: Path):
        self.detector = None
        self.error = None
        try:
            if not model_path.is_file():
                raise ValueError("Missing face mesh model; run scripts/download_face_landmarker.py")
            if hashlib.sha256(model_path.read_bytes()).hexdigest() != FACE_MODEL_SHA256:
                raise ValueError("Face mesh model checksum mismatch")
            import mediapipe as mp
            self.mp = mp
            self.detector = mp.tasks.vision.FaceLandmarker.create_from_options(
                mp.tasks.vision.FaceLandmarkerOptions(
                    base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
                    running_mode=mp.tasks.vision.RunningMode.IMAGE, num_faces=2,
                    min_face_detection_confidence=.65, min_face_presence_confidence=.65,
                    output_face_blendshapes=False, output_facial_transformation_matrixes=False))
        except Exception as error:
            self.error = str(error)

    def detect(self, rgb, *, state):
        if self.detector is None:
            return [], dict(ready=False, error=self.error, face_count=0)
        import cv2
        now = time.monotonic()
        cache = state.setdefault("face_mesh", {})
        shape = tuple(rgb.shape[:2])
        if cache.get("shape") != shape:
            cache.clear()
            cache["shape"] = shape
        refreshed = now - cache.get("attempted_at", float("-inf")) >= .10
        duration = 0.
        if refreshed:
            previous = cache.get("faces", [])
            cache.update(attempted_at=now, faces=[], error=None)
            started = time.perf_counter()
            scale = min(1., 512 / max(shape))
            small = cv2.resize(rgb, (round(shape[1] * scale), round(shape[0] * scale)))
            try:
                result = self.detector.detect(self.mp.Image(
                    image_format=self.mp.ImageFormat.SRGB, data=np.ascontiguousarray(small)))
                for landmarks in result.face_landmarks:
                    points = np.asarray([[p.x, p.y] for p in landmarks], dtype=float)
                    if len(points) >= 468 and np.isfinite(points).all():
                        cache["faces"].append(dict(points=points, observed_at=now))
            except Exception as error:
                cache["error"] = str(error)
            # A hand over the mouth can briefly hide the mesh. Retain each
            # missed outline for geometry only; its timestamp does not advance
            # and it cannot count as fresh confirmation or fresh depth.
            for face in previous:
                if now - face["observed_at"] > .50:
                    continue
                if not any(np.linalg.norm(face["points"][1] - new["points"][1]) < .10
                           for new in cache["faces"]):
                    cache["faces"].append(face)
            cache["faces"] = cache["faces"][:4]
            duration = (time.perf_counter() - started) * 1000
        return cache.get("faces", []), dict(
            ready=cache.get("error") is None, error=cache.get("error"),
            face_count=len(cache.get("faces", [])), refreshed=refreshed,
            inference_ms=round(duration, 2))

    def close(self):
        if self.detector is not None:
            self.detector.close()
            self.detector = None


def box_face_observations(recent_boxes, image_shape):
    """Tight unexpanded ellipse if an occluded face loses its detailed mesh.

    This weaker region can pause uncertain overlap, but cannot confirm contact.
    It is intentionally not the old expanded face/ear rectangle.
    """
    dimensions = np.asarray([image_shape[1], image_shape[0]])
    angle = np.linspace(-np.pi / 2, 3 * np.pi / 2, len(FACE_OVAL), endpoint=False)
    observations = []
    for (x, y, width, height), seen in recent_boxes:
        center = np.asarray([x + .5 * width, y + .52 * height])
        mesh = np.tile(center, (478, 1))
        mesh[list(FACE_OVAL)] = center + np.column_stack((.48 * width * np.cos(angle),
                                                        .50 * height * np.sin(angle)))
        observations.append(dict(points=mesh / dimensions, observed_at=seen, source="face_box_fallback"))
    return observations


def update_face_references(faces, hands, image_shape, *, state, now):
    """Trust overlap only for a face seen independently of the current hand.

    Face Landmarker can briefly fit a face to a fist/thumb. A newly appearing
    face on an existing hand must not veto every gesture. A real face seen
    before the hand reaches it remains a reference while it is tracked.
    Confident Pose head landmarks provide a separate reference below.
    """
    import cv2
    dimensions = np.asarray([image_shape[1], image_shape[0]])
    previous = [reference for reference in state.get("clear_face_references", [])
                if 0 <= now - reference["seen"] <= .75]
    updated = []
    for face in faces:
        mesh = np.asarray(face["points"]) * dimensions
        outline = cv2.convexHull(mesh[:468].astype(np.float32))
        width = max(float(np.ptp(mesh[:468], axis=0).max()), 1.)
        center = mesh[1]
        inherited = any(
            .65 <= width / reference["width"] <= 1.55
            and np.linalg.norm(center - reference["center"]) <= .35 * width
            for reference in previous
        )
        clear = True
        for points in hands:
            hand = np.asarray(points) * dimensions
            samples = np.vstack((hand, hand[[0, 5, 9, 13, 17]].mean(axis=0)))
            if any(cv2.pointPolygonTest(outline, tuple(point), False) >= 0 for point in samples):
                clear = False
                break
        face["clear_reference"] = bool(inherited or clear)
        if face["clear_reference"]:
            updated.append(dict(center=center, width=width, seen=face["observed_at"]))
    state["clear_face_references"] = (updated + previous)[:8]


def _current_image_poses(bodies, dimensions, now):
    """Image anatomy is usable even when Pose cannot reconstruct world depth."""
    poses = []
    for body in bodies:
        if not 0 <= now - body.get("observed_at", float("-inf")) <= .20:
            continue
        pose = np.asarray(body.get("landmarks"), dtype=float)
        if pose.shape != (33, 4):
            continue
        valid = (np.isfinite(pose).all(axis=1) & np.all(pose[:, 2:] >= .65, axis=1)
                 & np.all((pose[:, :2] >= -.05) & (pose[:, :2] <= 1.05), axis=1))
        poses.append((pose, pose[:, :2] * dimensions, valid))
    return poses


def _matched_arm(hand, palm_scale, poses):
    """Link the wrist to a plausible shoulder/elbow/hip chain without a head."""
    matches = []
    for pose, pixels, valid in poses:
        for side, shoulder_id, elbow_id, wrist_id, hip_id in (
            ("Left", 11, 13, 15, 23), ("Right", 12, 14, 16, 24)
        ):
            if not np.all(valid[[shoulder_id, elbow_id, wrist_id, hip_id]]):
                continue
            shoulder, elbow, wrist, hip = pixels[[shoulder_id, elbow_id, wrist_id, hip_id]]
            torso = float(np.linalg.norm(hip - shoulder))
            if torso < 50:
                continue
            down = (hip - shoulder) / torso
            upper_length = float(np.linalg.norm(elbow - shoulder)) / torso
            lower_length = float(np.linalg.norm(wrist - elbow)) / torso
            error = float(np.linalg.norm(hand[0] - wrist)) / palm_scale
            if (down[1] < .55 or not .12 <= upper_length <= .95
                    or not .12 <= lower_length <= .95 or error > .65):
                continue
            matches.append((error, side, shoulder, down, torso))
    matches.sort(key=lambda match: match[0])
    if not matches or (len(matches) > 1 and matches[1][0] - matches[0][0] < .20):
        return None
    _, side, shoulder, down, torso = matches[0]
    return dict(side=side, shoulder=shoulder, down=down, torso=torso)


def _supported_head(mesh, face_width, poses):
    """A head from any person protects a real face overlapping another torso."""
    for _, pixels, valid in poses:
        head_ids = np.flatnonzero(valid[:11])
        if not valid[0] or len(head_ids) < 4:
            continue
        # Compare the actual nose and its surrounding head points, rather
        # than body-world span; losing world depth must not erase a real face.
        if (np.linalg.norm(pixels[0] - mesh[1]) <= .35 * face_width
                and np.median(np.linalg.norm(pixels[head_ids] - mesh[1], axis=1))
                <= .85 * face_width):
            return True
    return False


def _torso_face_false_positive(mesh, outline, face_width, arm, poses):
    """Discard a shirt/forearm face only with positive, current arm anatomy.

    A genuine head matching any Pose protects its region. With a cropped
    head, the entire alleged face must lie below the shoulder plane and its
    nose must lie well onto the torso. This does not exempt arm-supported
    hands from real facial overlap or relax the hand classifier.
    """
    if arm is None or _supported_head(mesh, face_width, poses):
        return False
    shoulder, down, torso = arm["shoulder"], arm["down"], arm["torso"]
    center_height = float(np.dot(mesh[1] - shoulder, down)) / torso
    contour_height = (outline.reshape(-1, 2) - shoulder) @ down / torso
    return bool(center_height >= .25 and contour_height.min() >= .05)


def contact_evidence(points, image_shape, faces, bodies, *, now, policy=ContactPolicy()):
    """Surface overlap with a conservative pause until separation is known.

    Mouth/eye occlusion must not create holes through which a touching hand
    becomes a command. Clear shared-world depth separation can allow overlap;
    missing depth cannot certify that separation. This is not skin sensing.
    """
    import cv2
    result = dict(near_face=False, depth_available=False, depth_agrees=False,
                  candidate=False, depth_gap_ratio=None, face_alias=False,
                  arm_supported=False, status="clear")
    dimensions = np.array([image_shape[1], image_shape[0]])
    hand = np.asarray(points, dtype=float) * dimensions
    if hand.shape != (21, 2) or not np.isfinite(hand).all():
        return result, None
    palm = hand[[0, 5, 9, 13, 17]].mean(axis=0)
    palm_scale = float(np.median([np.linalg.norm(hand[a] - hand[b])
                                 for a, b in ((0, 5), (0, 9), (0, 17), (5, 17))]))
    if palm_scale < 8:
        return result, None
    image_poses = _current_image_poses(bodies, dimensions, now)
    arm = _matched_arm(hand, palm_scale, image_poses)
    if arm is not None:
        result.update(arm_supported=True, arm_side=arm["side"])
    # Tips, finger joints, and the palm centre. Pose has only three finger
    # anchors; middle/ring joints use index/pinky as coarse depth proxies.
    sample_ids = [4, 8, 12, 16, 20, 6, 10, 14, 18, 7, 11, 15, 19, 5, 9, 13, 17, 2, 3]
    samples = np.vstack((hand[sample_ids], palm))
    overlaps = []
    for face in faces:
        if not 0 <= now - face["observed_at"] <= .50:
            continue
        mesh = face["points"] * dimensions
        # A profile nose/lip can project beyond the oval's cheek contour. The
        # projected mesh hull includes those actual facial surface landmarks.
        outline = cv2.convexHull(mesh[:468].astype(np.float32))
        face_width = float(np.linalg.norm(mesh[234] - mesh[454]))
        if face_width < 30 or np.ptp(hand, axis=0).max() > 2. * np.ptp(outline, axis=0).max():
            continue
        if _torso_face_false_positive(mesh, outline, face_width, arm, image_poses):
            result["ignored_torso_face_count"] = result.get("ignored_torso_face_count", 0) + 1
            continue
        inside = [i for i, point in enumerate(samples)
                  if cv2.pointPolygonTest(outline, tuple(point), False) >= 0]
        if inside:
            all_inside = np.asarray([cv2.pointPolygonTest(outline, tuple(point), False) >= 0
                                     for point in hand])
            # The reported face hallucination is a small, complete "hand"
            # inside the nose/eye region with its wrist also inside the face.
            compact_alias = bool(all_inside.mean() >= .85 and all_inside[0]
                                 and np.ptp(hand, axis=0).max() <= .80 * np.ptp(outline, axis=0).max())
            # A compact cloud centred inside a real face still receives the
            # face-as-hand check. Normal hands require an independently seen
            # face or a matching body head before unknown depth can veto them.
            supported_face = bool(face.get("clear_reference", True)
                                  or _supported_head(mesh, face_width, image_poses))
            if not supported_face and not compact_alias:
                result["unverified_face_overlap_count"] = result.get("unverified_face_overlap_count", 0) + 1
                result["status"] = "unverified_face_overlap"
                continue
            overlaps.append((face, mesh, face_width, inside, compact_alias))
    if not overlaps:
        return result, None
    # Pick a local face reference even when Pose cannot reconstruct a curled
    # hand. It still permits a hold/release and explicit uncertainty reporting.
    overlaps.sort(key=lambda item: float(np.linalg.norm(item[1][1] - samples[item[3]].mean(axis=0))
                                        / item[2]))
    face, mesh, face_width, inside, alias = overlaps[0]
    tracking = dict(relative=(hand - mesh[1]) / face_width, face_center=mesh[1],
                    face_width=face_width, sample_at=face["observed_at"])
    result.update(near_face=True, candidate=True, face_alias=alias, status="depth_unavailable",
                  region_source=face.get("source", "mesh"))
    matches = []
    for body in bodies:
        if not 0 <= now - body["observed_at"] <= .20:
            continue
        pose = np.asarray(body["landmarks"], dtype=float)
        world = np.asarray(body.get("world_landmarks"), dtype=float)
        if pose.shape != (33, 4) or world.shape != (33, 3):
            continue
        valid = (np.isfinite(pose).all(axis=1) & np.isfinite(world).all(axis=1)
                 & np.all(pose[:, 2:] >= .55, axis=1))
        head_ids = np.flatnonzero(valid[:11])
        if not valid[0] or len(head_ids) < 4:
            continue
        body_pixels = pose[:, :2] * dimensions
        head = world[head_ids]
        head_width = float(np.linalg.norm(head[:, None] - head, axis=2).max())
        if not .08 <= head_width <= .30:
            continue
        for side, ids in (("Left", [15, 19, 17, 21]), ("Right", [16, 20, 18, 22])):
            # A closed fist often hides Pose's thumb/pinky. Require the wrist
            # and at least one finger, then compare only reliable anchors.
            available = np.asarray(ids)[valid[ids]]
            if not valid[ids[0]] or len(available) < 2:
                continue
            errors = np.linalg.norm(body_pixels[ids] - hand[[0, 8, 20, 4]], axis=1) / palm_scale
            strong_anchors = bool(np.all(pose[ids, 2:] >= .70))
            # Pose's coarse finger anchors need not coincide with the detailed
            # Hand Landmarker tips, especially on an open, spread palm. Permit
            # a bounded extra discrepancy only when every anchor is confident;
            # retain the wrist bound and ambiguous-person rejection below.
            finger_error_limit = 1.25 if strong_anchors else 1.0
            if errors[0] > .75 or np.median(errors[1:][valid[ids[1:]]]) > finger_error_limit:
                continue
            for face, mesh, face_width, inside, alias in overlaps:
                face_error = float(np.linalg.norm(body_pixels[0] - mesh[1]) / face_width)
                # On a turned head Pose can put its nose inside the cheek.
                # Use the surrounding head landmarks as corroboration before
                # accepting that larger nose offset, never the nose alone.
                coherent_head = (len(head_ids) >= 5 and
                    np.median(np.linalg.norm(body_pixels[head_ids] - mesh[1], axis=1))
                    <= .85 * face_width)
                face_error_limit = .50 if strong_anchors and coherent_head else .25
                if face_error > face_error_limit:
                    continue
                matches.append((float(errors[valid[ids]].mean()) + face_error, side, ids,
                                body, mesh, face_width, inside, head_ids, head_width, face, alias, valid))
    matches.sort(key=lambda row: row[0])
    if not matches or (len(matches) > 1 and matches[1][0] - matches[0][0] < .20):
        return result, tracking
    _, side, ids, body, mesh, face_width, inside, head_ids, head_width, face, alias, valid = matches[0]
    pose, world = body["landmarks"], body["world_landmarks"]
    wrist, index, pinky, thumb = ids
    proxies = [thumb, index, index, pinky, pinky, index, index, pinky, pinky,
               index, index, pinky, pinky, index, index, pinky, pinky, thumb, thumb]
    # If a contacting finger's depth is hidden, pause rather than assert that
    # an unrelated visible finger proves the whole hand is clear of the face.
    required = [proxies[i] for i in inside if i < len(proxies)]
    required += [wrist, index, pinky] if len(proxies) in inside else []
    result.update(arm_supported=True, face_alias=False)
    tracking = dict(relative=(hand - mesh[1]) / face_width, face_center=mesh[1],
                    face_width=face_width, sample_at=min(face["observed_at"], body["observed_at"]))
    if not all(valid[i] for i in required):
        return result, tracking
    depths = np.append(world[proxies, 2], world[[wrist, index, pinky], 2].mean())
    gaps = []
    for i in inside:
        nearest = np.argsort(np.linalg.norm(pose[head_ids, :2] * dimensions - samples[i], axis=1))[:2]
        surface_depth = float(np.median(world[head_ids[nearest], 2]))
        gaps.append(abs(float(depths[i]) - surface_depth) / head_width)
    gap = min(gaps)
    agrees = gap <= policy.maximum_depth_gap
    strong_anchors = bool(np.all(pose[ids, 2:] >= .70))
    separated = bool(gap >= policy.minimum_clear_depth_gap and strong_anchors)
    result.update(depth_available=True, depth_agrees=agrees, candidate=not separated,
                  depth_gap_ratio=round(gap, 3), arm_side=side,
                  region_source=face.get("source", "mesh"),
                  status="candidate" if agrees else "depth_separated" if separated else "depth_uncertain")
    # Reliable separation is also evidence against a face hallucination.
    if separated:
        result["face_alias"] = False
    # Internal tracking evidence never enters the public JSON payload.
    tracking = dict(relative=(hand - mesh[1]) / face_width,
                    face_center=mesh[1], face_width=face_width,
                    sample_at=min(face["observed_at"], body["observed_at"]))
    return result, tracking


def face_rejection_reason(evidence):
    if evidence.get("face_alias"):
        return FACE_HAND_REASON
    if evidence.get("candidate"):
        return CONTACT_PENDING_REASON if evidence.get("depth_available") else FACE_OVERLAP_REASON
    return None


def apply_contact_policy(rows, image_shape, faces, bodies, *, state, now, policy=ContactPolicy()):
    """Update at most once per final hand per frame, after crop deduplication."""
    cache = state.setdefault("face_contact", {})
    shape = tuple(image_shape[:2])
    if cache.get("shape") != shape or now < cache.get("time", now):
        cache.clear()
        cache["shape"] = shape
    old = [track for track in cache.get("tracks", [])
           if now - track["seen"] <= policy.maximum_sample_gap]
    dimensions = np.array([shape[1], shape[0]])
    used = set()
    updated = []
    for row in rows:
        hand = np.asarray(row["points"]) * dimensions
        center = hand[[0, 5, 9, 13, 17]].mean(axis=0)
        scale = max(row["metrics"]["palm_scale_px"], 1.)
        matches = []
        for i, track in enumerate(old):
            if i in used or (row.get("handedness") and track["side"]
                             and row["handedness"] != track["side"]):
                continue
            distance = float(np.linalg.norm(center - track["center"]) / max(scale, track["scale"]))
            if distance < .8 and .6 < scale / track["scale"] < 1.7:
                matches.append((distance, i))
        matches.sort()
        unambiguous = bool(matches) and (len(matches) == 1 or matches[1][0] - matches[0][0] > .20)
        if unambiguous:
            _, index = matches[0]
            used.add(index)
            track = old[index]
        else:
            track = dict(confirmed=False, started=None, count=0, last_positive=float("-inf"))
        evidence, spatial = contact_evidence(row["points"], image_shape, faces, bodies, now=now, policy=policy)
        candidate = evidence["candidate"] and row.get("rejection_reason") in {None, *FACE_VETO_REASONS}
        if candidate:
            same_face = ("face_center" in track and np.linalg.norm(spatial["face_center"] - track["face_center"])
                         <= .4 * spatial["face_width"])
            if not same_face:
                track.update(confirmed=False, started=None, count=0)
            fresh = spatial["sample_at"] > track.get("sample_at", float("-inf"))
            relative = spatial["relative"]
            elapsed = max(now - track.get("sample_time", now), .001)
            drift = (float(np.median(np.linalg.norm(relative - track["anchor"], axis=1)))
                     if track.get("started") is not None else 0.)
            speed = (float(np.median(np.linalg.norm(relative - track["relative"], axis=1))) / elapsed
                     if same_face and "relative" in track else 0.)
            stable = drift <= policy.maximum_drift and speed <= policy.maximum_speed
            if track["started"] is None or not stable:
                track.update(started=now, count=0, anchor=relative.copy())
            verified = evidence["depth_agrees"] and evidence.get("region_source") == "mesh"
            if fresh and verified:
                track["count"] += 1
                track["sample_at"] = spatial["sample_at"]
            elif not verified:
                track["count"] = 0
            if (verified and track["count"] >= 2
                    and now - track["started"] >= policy.confirmation_seconds - 1e-6):
                track["confirmed"] = True
            track.update(relative=relative.copy(), sample_time=now)
            track.update(last_positive=now, face_center=spatial["face_center"])
        else:
            track.update(started=None, count=0)
            if now - track["last_positive"] >= policy.release_seconds - 1e-6:
                track["confirmed"] = False
        held = not candidate and now - track["last_positive"] < policy.release_seconds - 1e-6
        blocked = bool(candidate or held)
        evidence.update(contact_confirmed=track["confirmed"], suppressed=blocked, release_hold=held,
                        status="face_alias" if evidence["face_alias"] else "release_hold" if held else
                        "contact" if track["confirmed"] else "confirming" if candidate and evidence["depth_agrees"]
                        else evidence["status"])
        row["face_context"] = evidence
        if blocked:
            reason = (FACE_HAND_REASON if evidence["face_alias"] else CONTACT_REASON if track["confirmed"]
                      else face_rejection_reason(evidence) or CONTACT_PENDING_REASON)
            if row.get("rejection_reason") in {None, *FACE_VETO_REASONS}:
                row["rejection_reason"] = reason
            row["recovery_score"] = -1.
        track.update(center=center, scale=scale, seen=now, side=row.get("handedness"))
        updated.append(track)
    # A one-frame missed hand must not erase confirmed contact. Expiration
    # remains bounded and association must still match when it reappears.
    retained = [track for i, track in enumerate(old) if i not in used
                and now - track["last_positive"] < policy.release_seconds]
    cache.update(tracks=(updated + retained)[:4], time=now)
