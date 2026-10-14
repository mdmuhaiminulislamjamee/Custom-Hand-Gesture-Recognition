"""Identify relaxed hanging and folded arms from confident body landmarks.

This is a posture rule, not clothing segmentation or measured hand/body depth.
It uses current hand pixels against confident arm/torso landmarks. Only a
Dorsal gesture on a relaxed arm below the waist is suppressed. The pose
cache belongs to a camera session; no body observations cross camera streams.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
import time

import numpy as np


POSE_MODEL_NAME = "pose_landmarker_lite.task"
POSE_MODEL_SHA256 = "59929e1d1ee95287735ddd833b19cf4ac46d29bc7afddbbf6753c459690d574a"
RESTING_HAND_REASON = "relaxed_hand_near_hip_or_thigh"
ARM_POSTURE_CONFIDENCE = .65
RELAXED_ELBOW_MIN_ANGLE_DEG = 135.
HANGING_UPPER_ARM_MIN_DOWN_ALIGNMENT = .60
HANGING_FOREARM_MIN_DOWN_ALIGNMENT = .75
POSE_REFRESH_SECONDS = .15
POSE_HOLD_SECONDS = .40
RELAXED_POSTURE_HOLD_SECONDS = .65


def _visible(pose, indices, threshold=.50):
    rows = pose[list(indices)]
    return bool(np.isfinite(rows).all() and np.all(rows[:, 2:] >= threshold)
                and np.all((rows[:, :2] >= -.05) & (rows[:, :2] <= 1.05)))


def _segment_distance(point, start, end):
    segment = end - start
    fraction = np.clip(np.dot(point - start, segment) / max(np.dot(segment, segment), 1.), 0., 1.)
    return float(np.linalg.norm(point - (start + fraction * segment)))


def _folded_resting_arm(hand, shoulder, elbow, wrist, palm_scale):
    """Find a dropped hand on a folded arm when the waist is out of frame.

    The elbow must extend outside the shoulder, while the forearm returns
    across it and the fingers hang down. A merely bent arm, or a downward hand
    held away from the torso, does not establish this resting posture.
    """
    upper, lower = elbow - shoulder, wrist - elbow
    upper_length, lower_length = float(np.linalg.norm(upper)), float(np.linalg.norm(lower))
    if min(upper_length, lower_length) < 35:
        return False
    bend = float(np.degrees(np.arccos(np.clip(
        -np.dot(upper, lower) / (upper_length * lower_length), -1., 1.))))
    finger_axis = hand[[8, 12, 16, 20]].mean(axis=0) - hand[[5, 9, 13, 17]].mean(axis=0)
    finger_length = float(np.linalg.norm(finger_axis))
    outward = elbow[0] - shoulder[0]
    inward = wrist[0] - shoulder[0]
    return bool(
        55 <= bend <= 125
        and upper[1] / upper_length >= .55
        and lower[1] >= .15 * upper_length
        and abs(outward) >= .12 * upper_length
        and outward * inward < 0
        and abs(inward) >= .12 * upper_length
        and finger_length >= .70 * palm_scale
        and finger_axis[1] / max(finger_length, 1.) >= .72
        and hand[[8, 12, 16, 20], 1].mean() - hand[0, 1] >= 1.1 * palm_scale
    )


def body_hand_context(points, image_shape, poses):
    """Return location and posture evidence for the nearest matched body wrist.

    Head and knees need not be visible: a confident shoulder/hip pair defines
    the torso axis and waist. Missing arm/torso evidence never
    becomes a positive claim that the hand is resting.
    """
    dimensions = np.array([image_shape[1], image_shape[0]], dtype=float)
    hand = np.asarray(points, dtype=float) * dimensions
    if hand.shape != (21, 2) or not np.isfinite(hand).all():
        return None
    palm_scale = float(np.median([np.linalg.norm(hand[a] - hand[b])
                                  for a, b in ((0, 5), (0, 9), (0, 17), (5, 17))]))
    palm = hand[[0, 5, 9, 13, 17]].mean(axis=0)
    matches = []
    arm_only_matches = []
    for person, pose in enumerate(poses):
        pose = np.asarray(pose, dtype=float)
        if pose.shape != (33, 4):
            continue
        pixels = pose[:, :2] * dimensions
        for side, shoulder_id, elbow_id, wrist_id, hip_id, knee_id in (
            ("Left", 11, 13, 15, 23, 25), ("Right", 12, 14, 16, 24, 26)
        ):
            arm_visible = _visible(pose, (shoulder_id, elbow_id, wrist_id),
                                   ARM_POSTURE_CONFIDENCE)
            hip_visible = _visible(pose, (shoulder_id, wrist_id, hip_id),
                                   ARM_POSTURE_CONFIDENCE)
            if not hip_visible and not arm_visible:
                continue
            shoulder, elbow, body_wrist, hip = pixels[[shoulder_id, elbow_id, wrist_id, hip_id]]
            if arm_visible:
                upper_length = float(np.linalg.norm(elbow - shoulder))
                wrist_gap = float(np.linalg.norm(hand[0] - body_wrist))
                if (upper_length >= 40 and palm_scale <= 1.20 * upper_length
                        and wrist_gap <= max(.35 * upper_length, .80 * palm_scale)):
                    arm_only_matches.append((wrist_gap / upper_length, person, side,
                                             shoulder, elbow, body_wrist,
                                             shoulder_id, elbow_id, wrist_id))
            if not hip_visible:
                continue
            torso = float(np.linalg.norm(hip - shoulder))
            if torso < 50 or palm_scale <= 0 or palm_scale > .36 * torso:
                continue
            wrist_gap = float(np.linalg.norm(hand[0] - body_wrist))
            if wrist_gap > min(.28 * torso, max(.15 * torso, 1.25 * palm_scale)):
                continue
            matches.append((wrist_gap / torso, person, side, pose, pixels,
                            shoulder, elbow, body_wrist, hip, shoulder_id, elbow_id, wrist_id,
                            hip_id, knee_id, torso))
    if not matches:
        if not arm_only_matches:
            return None
        (_, person, side, shoulder, elbow, body_wrist, shoulder_id,
         elbow_id, wrist_id) = min(arm_only_matches, key=lambda row: row[0])
        folded = _folded_resting_arm(hand, shoulder, elbow, body_wrist, palm_scale)
        return dict(body_index=person, arm_side=side, below_waist=False,
                    relaxed=folded, folded_resting_arm=folded, hanging_arm=False,
                    arm_confident=True, arm_landmark_indices=[shoulder_id, elbow_id, wrist_id],
                    waist_landmark_index=None, wrist_below_waist=False,
                    elbow_angle_deg=None, upper_arm_down_alignment=None,
                    forearm_down_alignment=None, wrist_height=None,
                    palm_height=None, thigh_distance=None, arm_straightness=None)
    # Associate with the nearest reliable body wrist before checking posture.
    # A background person's resting arm must not override a nearer raised arm.
    (_, person, side, pose, pixels, shoulder, elbow, body_wrist, hip, shoulder_id,
     elbow_id, wrist_id, hip_id, knee_id, torso) = min(matches, key=lambda row: row[0])
    down = (hip - shoulder) / torso
    if down[1] < .65:
        return None
    wrist = hand[0]
    wrist_height = float(np.dot(wrist - shoulder, down) / torso)
    thigh_end = hip + down * torso
    if _visible(pose, (knee_id,)) and np.dot(pixels[knee_id] - hip, down) > .30 * torso:
        thigh_end = pixels[knee_id]
    palm_height = float(np.dot(palm - hip, down) / torso)
    thigh_distance = _segment_distance(palm, hip, thigh_end) / torso
    straightness = None
    elbow_angle = None
    upper_alignment = None
    lower_alignment = None
    hanging_arm = False
    arm_confident = _visible(pose, (shoulder_id, elbow_id, wrist_id, hip_id),
                             ARM_POSTURE_CONFIDENCE)
    if arm_confident:
        upper, lower = elbow - shoulder, body_wrist - elbow
        upper_length, lower_length = np.linalg.norm(upper), np.linalg.norm(lower)
        if min(upper_length, lower_length) >= .14 * torso:
            straightness = float(np.clip(np.dot(upper, lower) /
                                         (upper_length * lower_length), -1., 1.))
            elbow_angle = float(np.degrees(np.arccos(-straightness)))
            upper_alignment = float(np.dot(upper / upper_length, down))
            lower_alignment = float(np.dot(lower / lower_length, down))
            hanging_arm = bool(elbow_angle >= RELAXED_ELBOW_MIN_ANGLE_DEG
                               and upper_alignment >= HANGING_UPPER_ARM_MIN_DOWN_ALIGNMENT
                               and lower_alignment >= HANGING_FOREARM_MIN_DOWN_ALIGNMENT)
    # Allow waist uncertainty from a sloping shoulder-to-hip axis, as in an
    # arm hanging beside a slightly bent torso. The wrist can be just above
    # the hip line while the hand hangs below it. A raised wrist or a palm
    # above the waist cannot become a resting hand just because the arm
    # happens to be straight. Finger extension
    # and palm orientation are intentionally absent from this body rule.
    wrist_below_waist = bool(wrist_height >= .88)
    below_waist = bool(wrist_below_waist and palm_height >= 0.)
    folded_resting_arm = bool(arm_confident and _folded_resting_arm(
        hand, shoulder, elbow, body_wrist, palm_scale))
    relaxed = bool((hanging_arm and below_waist) or folded_resting_arm)
    return dict(body_index=person, arm_side=side,
                below_waist=below_waist, relaxed=relaxed,
                folded_resting_arm=folded_resting_arm,
                hanging_arm=hanging_arm, arm_confident=arm_confident,
                arm_landmark_indices=[shoulder_id, elbow_id, wrist_id],
                waist_landmark_index=hip_id,
                wrist_below_waist=wrist_below_waist,
                elbow_angle_deg=round(elbow_angle, 2) if elbow_angle is not None else None,
                relaxed_elbow_min_angle_deg=RELAXED_ELBOW_MIN_ANGLE_DEG,
                upper_arm_down_alignment=(round(upper_alignment, 3)
                                          if upper_alignment is not None else None),
                forearm_down_alignment=(round(lower_alignment, 3)
                                        if lower_alignment is not None else None),
                wrist_height=round(wrist_height, 3),
                palm_height=round(palm_height, 3),
                thigh_distance=round(thigh_distance, 3),
                arm_straightness=round(straightness, 3) if straightness is not None else None)


def dorsal_body_rejection(context, label):
    """Suppress Dorsal on a confident hanging or folded resting arm.

    Legacy trouser-region diagnostics have no effect. Below-waist location
    alone is insufficient: the elbow and hanging arm must be established.
    """
    if (context and label == "dorsal" and (
            (context.get("below_waist") and context.get("relaxed"))
            or context.get("folded_resting_arm"))):
        return RESTING_HAND_REASON
    return None


def retain_relaxed_posture(rows, image_shape, *, state, now):
    """Bridge a brief Pose dropout only for the same stationary resting hand.

    A current non-relaxed context, hand movement, changed scale or ambiguous
    association cancels the hold. Reused context never refreshes its timestamp.
    Called once on final, deduplicated observations, not on recovery crops.
    """
    dimensions = np.asarray([image_shape[1], image_shape[0]])
    old = [item for item in state.get("relaxed_postures", [])
           if 0 <= now - item["seen"] <= RELAXED_POSTURE_HOLD_SECONDS]
    proposals = []
    for row in rows:
        wrist = np.asarray(row["points"])[0] * dimensions
        scale = max(float(row["metrics"]["palm_scale_px"]), 1.)
        matches = [index for index, item in enumerate(old)
                   if .75 <= scale / item["scale"] <= 1.33
                   and np.linalg.norm(wrist - item["wrist"]) <= .65 * min(scale, item["scale"])
                   and not (row.get("handedness_confidence", 0) >= .75
                            and row.get("handedness") and item["side"]
                            and row["handedness"] != item["side"])]
        proposals.append(matches[0] if len(matches) == 1 else None)
    updated = []
    for row, match in zip(rows, proposals):
        context = row.get("body_context")
        inherited = None
        if context is None and match is not None and proposals.count(match) == 1:
            inherited = old[match]
            context = dict(inherited["context"], posture_held=True,
                           posture_age_seconds=round(now - inherited["seen"], 3))
            row["body_context"] = context
        if not dorsal_body_rejection(context, "dorsal"):
            continue
        updated.append(inherited if inherited is not None else dict(
            wrist=np.asarray(row["points"])[0] * dimensions,
            scale=max(float(row["metrics"]["palm_scale_px"]), 1.),
            side=row.get("handedness"), context=dict(context), seen=now))
        reason = dorsal_body_rejection(context, row.get("resolved_label"))
        if reason and row.get("rejection_reason") in {None, RESTING_HAND_REASON}:
            row["rejection_reason"] = reason
            row["recovery_score"] = -1.
    # A fully missed hand can return within the bounded hold. A visible hand
    # with contrary posture must clear its old evidence immediately.
    state["relaxed_postures"] = updated if rows else old


class BodyGuard:
    """Small IMAGE-mode pose model with bounded, per-session observations."""

    def __init__(self, model_path: Path):
        self.detector = None
        self.error = None
        try:
            if not model_path.is_file():
                raise ValueError("Missing body pose model; run scripts/download_pose_landmarker.py")
            if hashlib.sha256(model_path.read_bytes()).hexdigest() != POSE_MODEL_SHA256:
                raise ValueError("Body pose model checksum mismatch; run scripts/download_pose_landmarker.py")
            import mediapipe as mp
            self.mp = mp
            self.detector = mp.tasks.vision.PoseLandmarker.create_from_options(
                mp.tasks.vision.PoseLandmarkerOptions(
                    base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
                    running_mode=mp.tasks.vision.RunningMode.IMAGE,
                    num_poses=2, min_pose_detection_confidence=.50,
                    min_pose_presence_confidence=.50,
                    output_segmentation_masks=False,
                ))
        except Exception as error:
            self.error = str(error)

    def detect(self, rgb, *, state):
        """Refresh at most every 150 ms, retaining missed bodies for 400 ms."""
        if self.detector is None:
            return [], dict(ready=False, error=self.error, body_count=0)
        import cv2
        now = time.monotonic()
        cache = state.setdefault("body_guard", {})
        shape = tuple(rgb.shape[:2])
        if cache.get("shape") != shape:
            cache.clear()
            cache["shape"] = shape
        refreshed = now - cache.get("attempted_at", float("-inf")) >= POSE_REFRESH_SECONDS
        duration_ms = 0.0
        if refreshed:
            cache["attempted_at"] = now
            # Contact evidence must use fresh shared-world coordinates, never
            # the longer occlusion cache used by the waist policy.
            cache["contact_poses"] = []
            started = time.perf_counter()
            height, width = shape
            scale = min(1., 512 / max(height, width))
            small = cv2.resize(rgb, (round(width * scale), round(height * scale)))
            try:
                result = self.detector.detect(self.mp.Image(
                    image_format=self.mp.ImageFormat.SRGB, data=np.ascontiguousarray(small)))
                poses = []
                world_poses = getattr(result, "pose_world_landmarks", []) or []
                for index, landmarks in enumerate(result.pose_landmarks):
                    pose = np.array([[p.x, p.y,
                                      1. if p.visibility is None else p.visibility,
                                      1. if p.presence is None else p.presence]
                                     for p in landmarks], dtype=float)
                    if pose.shape == (33, 4):
                        world = None
                        if index < len(world_poses):
                            candidate_world = np.asarray([[p.x, p.y, p.z]
                                                          for p in world_poses[index]], dtype=float)
                            if candidate_world.shape == (33, 3):
                                world = candidate_world
                        # Current image arm landmarks also disambiguate a
                        # real hand from a false face region on the torso.
                        # Missing world output must not discard that evidence.
                        cache["contact_poses"].append(dict(
                            landmarks=pose, world_landmarks=world, observed_at=now))
                    if pose.shape == (33, 4) and (
                        _visible(pose, (11, 15, 23)) or _visible(pose, (12, 16, 24))
                        or _visible(pose, (11, 13, 15)) or _visible(pose, (12, 14, 16))
                    ):
                        poses.append(pose)
                # Retain missed individuals, even if another person remains visible.
                retained = []
                for old, seen in cache.get("recent", []):
                    old_center = old[[23, 24], :2].mean(axis=0)
                    old_scale = max(np.linalg.norm(old[23, :2] - old[11, :2]), .10)
                    replaced = any(np.linalg.norm(new[[23, 24], :2].mean(axis=0) - old_center)
                                   < .40 * old_scale for new in poses)
                    if now - seen <= POSE_HOLD_SECONDS and not replaced:
                        retained.append((old, seen))
                cache["recent"] = ([(pose, now) for pose in poses] + retained)[:4]
                cache["error"] = None
            except Exception as error:
                cache["error"] = str(error)
            duration_ms = (time.perf_counter() - started) * 1000
        recent = [(pose, seen) for pose, seen in cache.get("recent", [])
                  if now - seen <= POSE_HOLD_SECONDS]
        cache["recent"] = recent
        return [pose for pose, _ in recent], dict(
            ready=cache.get("error") is None, error=cache.get("error"), body_count=len(recent),
            refreshed=refreshed, inference_ms=round(duration_ms, 2),
            oldest_pose_age_ms=round(max((now - seen for _, seen in recent), default=0.) * 1000),
        )

    def contact_observations(self, *, state):
        cache = state.get("body_guard", {})
        if self.detector is None or cache.get("error"):
            return []
        now = time.monotonic()
        return [pose for pose in cache.get("contact_poses", [])
                if 0 <= now - pose["observed_at"] <= .20]

    def close(self):
        if self.detector is not None:
            self.detector.close()
            self.detector = None
