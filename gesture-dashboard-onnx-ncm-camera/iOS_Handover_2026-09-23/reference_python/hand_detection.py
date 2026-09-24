"""Native-detail hand tracking with a bounded orientation/crop recovery pass."""
from __future__ import annotations

import threading
import time
from pathlib import Path

import numpy as np
from .face_guard import FaceGuard, overlaps_face_as_small_hand

DETECTOR_SETTINGS = {
    "running_mode": "VIDEO with IMAGE recovery",
    "minimum_detection_confidence": 0.55,
    "minimum_presence_confidence": 0.55,
    "minimum_tracking_confidence": 0.60,
    "recovery_detection_confidence": 0.20,
    "recovery_presence_confidence": 0.30,
    "face_guard": "BlazeFace small-hand overlap exclusion",
    "recovery_confirmation_frames": 2,
    "minimum_palm_pixels": 8,
    "minimum_hand_span_pixels": 24,
    "minimum_hand_short_side_pixels": 10,
    "maximum_analysis_dimension": 960,
    "maximum_detection_passes_per_frame": 2,
    "recovery_start_budget_ms": 55,
    "distance_method": "user-calibrated apparent palm size; no absolute MediaPipe depth",
}


def restore_points(points, view, image_shape):
    """Undo np.rot90 and a crop without changing the camera's command axes."""
    x, y, width, height, turns = view[:5]
    restored = np.asarray(points, dtype=np.float32).copy()
    for _ in range(turns % 4):
        restored = np.column_stack((1.0 - restored[:, 1], restored[:, 0]))
    restored[:, 0] = (restored[:, 0] * width + x) / image_shape[1]
    restored[:, 1] = (restored[:, 1] * height + y) / image_shape[0]
    return restored


def restore_world_points(points, turns):
    """Undo detector-view rotations for MediaPipe metric world landmarks."""
    restored = np.asarray(points, dtype=np.float32).reshape(21, 3).copy()
    for _ in range(int(turns) % 4):
        x = restored[:, 0].copy()
        restored[:, 0] = -restored[:, 1]
        restored[:, 1] = x
    return restored


def hand_pixel_metrics(points, image_shape):
    pixels = np.asarray(points) * np.asarray([image_shape[1], image_shape[0]])
    segments = [float(np.linalg.norm(pixels[a] - pixels[b]))
                for a, b in ((0, 5), (0, 9), (0, 17), (5, 17))]
    extent = np.ptp(pixels, axis=0)
    return {
        "palm_segments_px": segments,
        "palm_scale_px": float(np.median(segments)),
        "hand_span_px": float(extent.max()),
        "hand_short_side_px": float(extent.min()),
        "analysis_width": int(image_shape[1]),
        "analysis_height": int(image_shape[0]),
    }


def normalized_hand_shape(points):
    """Remove whole-hand motion and scale before comparing recovered tracks."""
    shape = np.asarray(points, dtype=np.float32).reshape(21, 2).copy()
    palm_center = shape[[0, 5, 9, 13, 17]].mean(axis=0)
    shape -= palm_center
    scale = float(np.linalg.norm(shape[9] - shape[0]))
    if scale < 1e-6:
        scale = float(np.linalg.norm(np.ptp(shape, axis=0)))
    return shape / max(scale, 1e-6)


class HandDetector:
    def __init__(self, model_path: Path):
        self.ready = False
        self.error = None
        self.running_mode = "VIDEO"
        self._landmarker = self._recovery = self._mp = None
        self._timestamp_lock = threading.Lock()
        self._last_timestamp_ms = -1
        self._last_diagnostics = {}
        self._view = None
        self._shape = None
        self._search_index = self._misses = 0
        self._recovery_tracking = False
        self._pose_recovery_requested = False
        self._unqualified_frames = 0
        self._pending_recovery = None
        self._pending_recovery_count = 0
        self._accepted_points = None
        self._accepted_view = None
        self._last_world_landmarks = None
        self._face_guard = None
        if not model_path.exists():
            self.error = f"Missing MediaPipe hand model: {model_path.name}"
            return
        try:
            import mediapipe as mp
            self._mp = mp
            common = dict(base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
                          num_hands=1,
                          min_tracking_confidence=0.60)
            self._landmarker = mp.tasks.vision.HandLandmarker.create_from_options(
                mp.tasks.vision.HandLandmarkerOptions(
                    **common, min_hand_detection_confidence=0.55, min_hand_presence_confidence=0.55,
                    running_mode=mp.tasks.vision.RunningMode.VIDEO))
            self._recovery = mp.tasks.vision.HandLandmarker.create_from_options(
                mp.tasks.vision.HandLandmarkerOptions(
                    **common, min_hand_detection_confidence=0.20, min_hand_presence_confidence=0.30,
                    running_mode=mp.tasks.vision.RunningMode.IMAGE))
            self.ready = True
            if model_path.with_name("blaze_face_short_range.tflite").exists():
                self._face_guard = FaceGuard(model_path.with_name("blaze_face_short_range.tflite"))
        except Exception as error:
            self.error = f"MediaPipe initialization failed: {error}"
            self.close()

    def _run_view(self, rgb, view, *, video):
        x, y, width, height, turns, enhance = view
        crop = rgb[y:y + height, x:x + width]
        if not video:
            # A bright window can hide an underexposed hand in the full-frame
            # exposure statistic. Enhance only an actually dark recovery crop.
            import cv2
            if enhance or float(np.median(cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY))) < 85:
                lab = cv2.cvtColor(crop, cv2.COLOR_RGB2LAB)
                lab[:, :, 0] = cv2.createCLAHE(clipLimit=1.8, tileGridSize=(8, 8)).apply(lab[:, :, 0])
                crop = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB,
                               data=np.ascontiguousarray(np.rot90(crop, turns)))
        if video:
            with self._timestamp_lock:
                current = int(time.monotonic_ns() // 1_000_000)
                timestamp = max(current, self._last_timestamp_ms + 1)
                self._last_timestamp_ms = timestamp
            self._last_diagnostics.update(timestamp_ms=timestamp, timestamp_adjusted=timestamp != current)
            result = self._landmarker.detect_for_video(image, timestamp)
        else:
            result = self._recovery.detect(image)
        candidates = []
        for index, hand in enumerate(getattr(result, "hand_landmarks", []) or []):
            points = np.asarray([[p.x, p.y] for p in hand], dtype=np.float32)
            if points.shape != (21, 2) or not np.isfinite(points).all():
                continue
            points = restore_points(points, view, rgb.shape)
            categories = getattr(result, "handedness", []) or []
            category = categories[index][0] if index < len(categories) and categories[index] else None
            world_hands = getattr(result, "hand_world_landmarks", []) or []
            world = None
            if index < len(world_hands):
                candidate_world = np.asarray(
                    [[p.x, p.y, p.z] for p in world_hands[index]], dtype=np.float32
                )
                if candidate_world.shape == (21, 3) and np.isfinite(candidate_world).all():
                    world = restore_world_points(candidate_world, view[4])
            candidates.append((float(np.prod(np.ptp(points, axis=0))), points, category, world))
        return max(candidates, key=lambda row: row[0]) if candidates else None

    def _search_views(self, shape):
        height, width = shape[:2]
        views = [(0, 0, width, height, turn, enhance)
                 for turn, enhance in ((3, 0), (3, 1), (1, 0), (1, 1), (2, 0), (0, 1))]
        side = max(32, int(min(width, height) * 0.60))
        # Search only one alternative per frame; cycle overlapping regions.
        for turn in (0, 3, 1):
            for fx, fy in ((.5, .5), (0, .5), (1, .5), (0, 0), (1, 0), (0, 1), (1, 1)):
                views.append((round((width-side)*fx), round((height-side)*fy), side, side, turn, 1))
        return views

    def detect(self, rgb_image, candidate_score=None):
        if not self.ready:
            return None
        self._last_world_landmarks = None
        started = time.perf_counter()
        height, width = rgb_image.shape[:2]
        if self._shape != (height, width):
            self._shape = (height, width)
            self._view = None
            self._misses = self._search_index = 0
            self._recovery_tracking = False
            self._pose_recovery_requested = False
            self._unqualified_frames = 0
            self._pending_recovery = None
            self._pending_recovery_count = 0
            self._accepted_points = None
            self._accepted_view = None
        full = (0, 0, width, height, 0, 0)
        view = self._view or full
        # A classifier veto is not evidence that the tracking coordinates jumped.
        # Reset smoothing only when a different image view actually replaces it.
        reacquired = False
        if self._pose_recovery_requested:
            self._recovery_tracking = True
            self._pose_recovery_requested = False
        self._last_diagnostics = {"running_mode": "VIDEO", "candidate_count": 0,
                                  "analysis_width": width, "analysis_height": height,
                                  "reacquired": reacquired,
                                  "recovery_attempted": False, "rejection_reason": None}
        face_guard = getattr(self, "_face_guard", None)
        boxes = face_guard.detect(rgb_image) if face_guard else []
        self._last_diagnostics["face_guard_ready"] = bool(face_guard and face_guard.detector)
        self._last_diagnostics["face_count"] = len(boxes)
        selected = self._run_view(rgb_image, view, video=not self._recovery_tracking)
        if selected is not None and overlaps_face_as_small_hand(selected[1], rgb_image.shape, boxes):
            selected = None
            self._last_diagnostics["rejection_reason"] = "hand_candidate_overlaps_face"
        def score(row):
            if row is None:
                return -1.0
            metrics = hand_pixel_metrics(row[1], rgb_image.shape)
            if metrics["palm_scale_px"] < 8 or metrics["hand_span_px"] < 24 or metrics["hand_short_side_px"] < 10:
                return -0.5
            return candidate_score(row[1]) if candidate_score else 1.0
        selected_score = score(selected)
        if self._recovery_tracking:
            self._last_diagnostics["running_mode"] = "IMAGE_RECOVERY"
        if selected_score < 1.0 and (time.perf_counter() - started) < .055:
            views = self._search_views(rgb_image.shape)
            if self._view is not None and self._unqualified_frames < 2:
                # Retry the last view first. Side-pointing hands are often
                # visible only after a 90-degree detector rotation; abandoning
                # that view after one intermittent miss made the search cycle
                # through unrelated rotations for several seconds.
                recovery_view = (*view[:5], 1 - view[5])
            else:
                recovery_view = views[self._search_index % len(views)]
                self._search_index += 1
            self._last_diagnostics["recovery_attempted"] = True
            alternative = self._run_view(rgb_image, recovery_view, video=False)
            if alternative is not None and overlaps_face_as_small_hand(alternative[1], rgb_image.shape, boxes):
                alternative = None
            alternative_score = score(alternative)
            if alternative_score > selected_score:
                selected, view, selected_score = alternative, recovery_view, alternative_score
                # An occluded side view can make VIDEO tracking drift to a
                # plausible but incorrect open hand. Keep redetecting the
                # recovered view rather than inheriting that tracking state.
                self._recovery_tracking = True
                self._last_diagnostics["reacquired"] = view != (self._view or full)
        self._unqualified_frames = 0 if selected_score >= 1.0 else self._unqualified_frames + 1
        if selected is None:
            self._pending_recovery = None
            self._pending_recovery_count = 0
            self._misses += 1
            if self._misses >= 3:
                self._view = None
                self._recovery_tracking = False
                self._accepted_points = None
                self._accepted_view = None
            return None
        area, points, category, *extra = selected
        world_landmarks = extra[0] if extra else None
        metrics = hand_pixel_metrics(points, rgb_image.shape)
        # Judge source pixels, not an arbitrary percentage of the camera frame.
        # Upscaling or zooming a crop must not bypass the information floor.
        usable = (metrics["palm_scale_px"] >= 8 and metrics["hand_span_px"] >= 24
                  and metrics["hand_short_side_px"] >= 10)
        self._last_diagnostics.update(metrics)
        self._last_diagnostics.update(
            candidate_count=1, selected_bbox_area=area,
            selected_handedness=getattr(category, "category_name", None),
            selected_handedness_confidence=getattr(category, "score", None),
            detection_rotation_degrees=int(view[4] * 90), detection_view=list(view[:4]), recovery_contrast=bool(view[5]),
            small_hand=metrics["palm_scale_px"] < 20,
            rejection_reason=None if usable else "insufficient_hand_pixels")
        if not usable:
            self._misses += 1
            if self._misses >= 3:
                self._view = None
                self._accepted_points = None
                self._accepted_view = None
            return None
        self._misses = 0
        self._view = view
        self._search_index = 0
        if self._recovery_tracking or getattr(self, "_pending_recovery", None) is not None:
            pending = getattr(self, "_pending_recovery", None)
            same_accepted_view = (
                getattr(self, "_accepted_points", None) is not None
                and getattr(self, "_accepted_view", None) == tuple(view[:5])
            )
            previous = pending if pending is not None else (
                self._accepted_points if same_accepted_view else None
            )
            # A genuine pointing hand may be moving across the scene. Compare
            # hand shape rather than absolute screen coordinates so translation
            # and moderate distance changes do not prevent reacquisition.
            current_shape = normalized_hand_shape(points)
            previous_shape = normalized_hand_shape(previous) if previous is not None else None
            consistent = previous_shape is not None and float(
                np.median(np.linalg.norm(current_shape - previous_shape, axis=1))
            ) < .15
            previous_count = getattr(self, "_pending_recovery_count", 0) if pending is not None else (
                1 if same_accepted_view else 0
            )
            self._pending_recovery_count = previous_count + 1 if consistent else 1
            self._pending_recovery = points.copy()
            # Once IMAGE mode has found a qualified view, let VIDEO tracking
            # verify that view on the next frame. Requiring three consecutive
            # IMAGE detections made low-resolution side-pointing hands nearly
            # impossible to reacquire because IMAGE detection is intermittent.
            # The temporal command gate still requires stable observations.
            self._recovery_tracking = False
            if self._pending_recovery_count < 2:
                self._last_diagnostics["rejection_reason"] = "confirming_recovered_hand"
                return None
            self._pending_recovery = None
            self._pending_recovery_count = 0
        else:
            self._pending_recovery = None
            self._pending_recovery_count = 0
        if metrics["palm_scale_px"] < 24 and metrics["hand_span_px"] < 90:
            pixels = points * [width, height]
            side = min(min(width, height), max(96, int(np.ptp(pixels, axis=0).max() * 2.6)))
            center = (pixels.min(axis=0) + pixels.max(axis=0)) / 2
            x = int(np.clip(center[0] - side / 2, 0, width - side))
            y = int(np.clip(center[1] - side / 2, 0, height - side))
            self._view = (x, y, side, side, view[4], view[5])
        self._accepted_points = points.copy()
        self._accepted_view = tuple(self._view[:5])
        self._last_world_landmarks = (
            None if world_landmarks is None else np.asarray(world_landmarks, dtype=np.float32).copy()
        )
        return points

    def diagnostics(self):
        return dict(self._last_diagnostics)

    def world_landmarks(self):
        points = getattr(self, "_last_world_landmarks", None)
        return None if points is None else np.asarray(points, dtype=np.float32).copy()

    def request_pose_recovery(self):
        # Do not keep tracking a hallucinated finger arrangement indefinitely.
        # The caller still rejects the current frame; recovery uses new pixels.
        self._pose_recovery_requested = True

    def close(self):
        if getattr(self, "_face_guard", None) is not None:
            self._face_guard.close()
        for name in ("_landmarker", "_recovery"):
            model = getattr(self, name, None)
            if model is not None:
                model.close()
            setattr(self, name, None)
        self.ready = False
