"""Locate a hand against a standing person's hip and thigh.

This is a posture rule, not clothing segmentation or measured hand/body depth.
It uses current hand pixels against confident arm/torso landmarks. The pose
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
DORSAL_BELOW_WAIST_REASON = "dorsal_below_waist"
POSE_REFRESH_SECONDS = .15
POSE_HOLD_SECONDS = .40


def _visible(pose, indices, threshold=.50):
    rows = pose[list(indices)]
    return bool(np.isfinite(rows).all() and np.all(rows[:, 2:] >= threshold)
                and np.all((rows[:, :2] >= -.05) & (rows[:, :2] <= 1.05)))


def _segment_distance(point, start, end):
    segment = end - start
    fraction = np.clip(np.dot(point - start, segment) / max(np.dot(segment, segment), 1.), 0., 1.)
    return float(np.linalg.norm(point - (start + fraction * segment)))


def body_hand_context(points, image_shape, poses):
    """Return location and posture evidence for the nearest matched body wrist.

    Head and knees need not be visible: a confident shoulder/hip pair defines
    the torso axis and upper-thigh region. Missing arm/torso evidence never
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
    for person, pose in enumerate(poses):
        pose = np.asarray(pose, dtype=float)
        if pose.shape != (33, 4):
            continue
        pixels = pose[:, :2] * dimensions
        for side, shoulder_id, elbow_id, wrist_id, hip_id, knee_id in (
            ("Left", 11, 13, 15, 23, 25), ("Right", 12, 14, 16, 24, 26)
        ):
            if not _visible(pose, (shoulder_id, wrist_id, hip_id)):
                continue
            shoulder, elbow, body_wrist, hip = pixels[[shoulder_id, elbow_id, wrist_id, hip_id]]
            torso = float(np.linalg.norm(hip - shoulder))
            if torso < 50 or palm_scale <= 0 or palm_scale > .36 * torso:
                continue
            wrist_gap = float(np.linalg.norm(hand[0] - body_wrist))
            if wrist_gap > min(.28 * torso, max(.15 * torso, 1.25 * palm_scale)):
                continue
            matches.append((wrist_gap / torso, person, side, pose, pixels,
                            shoulder, elbow, hip, elbow_id, knee_id, torso))
    if not matches:
        return None
    # Associate with the nearest reliable body wrist before checking posture.
    # A background person's resting arm must not override a nearer raised arm.
    _, person, side, pose, pixels, shoulder, elbow, hip, elbow_id, knee_id, torso = min(matches, key=lambda row: row[0])
    down = (hip - shoulder) / torso
    if down[1] < .65:
        return None
    wrist = hand[0]
    wrist_height = float(np.dot(wrist - shoulder, down) / torso)
    finger_drop = (hand[[8, 12, 16, 20]] - wrist) @ down
    thigh_end = hip + down * torso
    if _visible(pose, (knee_id,)) and np.dot(pixels[knee_id] - hip, down) > .30 * torso:
        thigh_end = pixels[knee_id]
    palm_height = float(np.dot(palm - hip, down) / torso)
    thigh_distance = _segment_distance(palm, hip, thigh_end) / torso
    straightness = None
    hanging_arm = False
    if _visible(pose, (elbow_id,)):
        upper, lower = elbow - shoulder, wrist - elbow
        upper_length, lower_length = np.linalg.norm(upper), np.linalg.norm(lower)
        if min(upper_length, lower_length) >= .14 * torso:
            straightness = float(np.dot(upper, lower) / (upper_length * lower_length))
            hanging_arm = bool(straightness >= .72 and np.dot(upper, down) >= .20 * torso
                               and np.dot(lower / lower_length, down) >= .65
                               and .72 <= wrist_height <= 1.80)
    fingers_down = bool(np.dot(hand[9] - wrist, down) >= .35 * palm_scale
                        and np.count_nonzero(finger_drop > .65 * palm_scale) >= 3)
    relaxed = bool(hanging_arm and fingers_down and -.22 <= palm_height <= 1.15
                   and thigh_distance <= .42)
    return dict(body_index=person, arm_side=side,
                below_waist=bool(palm_height >= -.10), relaxed=relaxed,
                wrist_height=round(wrist_height, 3),
                palm_height=round(palm_height, 3),
                thigh_distance=round(thigh_distance, 3),
                arm_straightness=round(straightness, 3) if straightness is not None else None)


def dorsal_body_rejection(context, label):
    """Only dorsal is subject to the standing waist and relaxed-hand policy."""
    if label != "dorsal" or not context:
        return None
    if context["relaxed"]:
        return RESTING_HAND_REASON
    if context["below_waist"]:
        return DORSAL_BELOW_WAIST_REASON
    return None


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
            started = time.perf_counter()
            height, width = shape
            scale = min(1., 512 / max(height, width))
            small = cv2.resize(rgb, (round(width * scale), round(height * scale)))
            try:
                result = self.detector.detect(self.mp.Image(
                    image_format=self.mp.ImageFormat.SRGB, data=np.ascontiguousarray(small)))
                poses = []
                for landmarks in result.pose_landmarks:
                    pose = np.array([[p.x, p.y,
                                      1. if p.visibility is None else p.visibility,
                                      1. if p.presence is None else p.presence]
                                     for p in landmarks], dtype=float)
                    if pose.shape == (33, 4) and (
                        _visible(pose, (11, 15, 23)) or _visible(pose, (12, 16, 24))
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

    def close(self):
        if self.detector is not None:
            self.detector.close()
            self.detector = None
