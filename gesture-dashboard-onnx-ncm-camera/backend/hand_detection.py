"""Native-detail hand tracking with a bounded orientation/crop recovery pass."""
from __future__ import annotations

import threading
import time
from pathlib import Path

import numpy as np
from .face_guard import FaceGuard, face_context_observation
from .body_guard import (BodyGuard, POSE_MODEL_NAME, DORSAL_CONTEXT_VETOES,
                         RELAXED_ELBOW_MIN_ANGLE_DEG, body_hand_context, dorsal_body_rejection,
                         retain_relaxed_posture)

DETECTOR_SETTINGS = {
    "maximum_hands": 2,
    "live_hand_selection": "foreground-hand tracking; confirmed hand switch; retain clear owner near equal distance",
    "live_minimum_palm_pixels": 18,
    "live_minimum_dorsal_palm_pixels": 24,
    "running_mode": "VIDEO with IMAGE recovery",
    "minimum_detection_confidence": 0.55,
    "minimum_presence_confidence": 0.55,
    "minimum_tracking_confidence": 0.60,
    "recovery_detection_confidence": 0.20,
    "recovery_presence_confidence": 0.30,
    "face_guard": "disabled; gestures may overlap the face",
    "body_guard": "Dorsal requires confident arm and waist, above-waist presentation; relaxed or unknown posture rejected",
    "relaxed_elbow_min_angle_deg": RELAXED_ELBOW_MIN_ANGLE_DEG,
    "body_pose_refresh_ms": 150,
    "body_pose_occlusion_hold_ms": 400,
    "recovery_confirmation_frames": 2,
    "minimum_palm_pixels": 8,
    "minimum_hand_span_pixels": 24,
    "minimum_hand_short_side_pixels": 10,
    "maximum_analysis_dimension": 960,
    "maximum_detection_passes_per_frame": 2,
    "recovery_start_budget_ms": 55,
    "distance_method": "projected image/world hand scale when coherent; palm-size fallback; no absolute depth",
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
    def __init__(self, model_path: Path, *, use_body_context=False):
        self.ready = False
        self.error = None
        self.running_mode = "VIDEO"
        self._landmarker = self._recovery = self._mp = None
        self._timestamp_lock = threading.Lock()
        self._last_timestamp_ms = -1
        self._frame_time = None
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
        self._body_guard = None
        if not model_path.exists():
            self.error = f"Missing MediaPipe hand model: {model_path.name}"
            return
        try:
            import mediapipe as mp
            self._mp = mp
            common = dict(base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
                          num_hands=2,
                          min_tracking_confidence=0.60)
            self._landmarker = mp.tasks.vision.HandLandmarker.create_from_options(
                mp.tasks.vision.HandLandmarkerOptions(
                    **common, min_hand_detection_confidence=0.55, min_hand_presence_confidence=0.55,
                    running_mode=mp.tasks.vision.RunningMode.VIDEO))
            self._recovery = mp.tasks.vision.HandLandmarker.create_from_options(
                mp.tasks.vision.HandLandmarkerOptions(
                    **common, min_hand_detection_confidence=0.20, min_hand_presence_confidence=0.30,
                    running_mode=mp.tasks.vision.RunningMode.IMAGE))
            if use_body_context:
                self._body_guard = BodyGuard(model_path.with_name(POSE_MODEL_NAME))
                # Missing body evidence disables Dorsal through its strict
                # verifier, while hand-only commands remain usable.
            self.ready = True
            if not use_body_context and model_path.with_name("blaze_face_short_range.tflite").exists():
                self._face_guard = FaceGuard(model_path.with_name("blaze_face_short_range.tflite"))
        except Exception as error:
            self.error = f"MediaPipe initialization failed: {error}"
            self.close()

    def _run_candidates(self, rgb, view, *, video):
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
                frame_time = getattr(self, "_frame_time", None)
                current = (int(frame_time * 1000) if frame_time is not None
                           else int(time.monotonic_ns() // 1_000_000))
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
        return candidates

    def _run_view(self, rgb, view, *, video):
        # Legacy single-image collection helpers still consume one observation.
        candidates = self._run_candidates(rgb, view, video=video)
        return max(candidates, key=lambda row: row[0]) if candidates else None

    def detect_candidates(self, rgb_image, *, state, candidate_score):
        """Live path: always scan the full frame before bounded crop recovery.

        Recovery state belongs to the caller's session. A crop can improve one
        hand, but can never hide the other from the next full-frame observation.
        """
        if not self.ready:
            return []
        self._frame_time = state.get("frame_time")
        started = time.perf_counter()
        height, width = rgb_image.shape[:2]
        if state.get("shape") != (height, width):
            frame_time = state.get("frame_time")
            state.clear()
            state["shape"] = (height, width)
            if frame_time is not None:
                state["frame_time"] = frame_time
        state["frames"] = state.get("frames", 0) + 1
        full = (0, 0, width, height, 0, 0)
        same_context = getattr(self, "_multi_context", None) is state
        self._multi_context = state
        self._last_diagnostics = dict(running_mode="VIDEO" if same_context else "IMAGE",
                                      candidate_count=0, analysis_width=width, analysis_height=height,
                                      recovery_attempted=False, rejection_reason=None)
        # Face position/contact is not a live gesture veto. Keep explicit
        # disabled diagnostics for existing viewers without loading a model.
        self._last_diagnostics.update(
            face_contact=dict(enabled=False, ready=False, face_count=0, error=None),
            face_guard_ready=False, face_guard_error=None, face_count=0,
            face_contact_rejected_count=0, near_face_candidate_count=0)
        body_guard = getattr(self, "_body_guard", None)
        poses = None
        self._last_diagnostics["body_guard"] = dict(
            ready=bool(body_guard and body_guard.detector), checked=False, body_count=0,
            error=getattr(body_guard, "error", None))

        def observations(view, video):
            nonlocal poses
            rows = []
            candidates = self._run_candidates(rgb_image, view, video=video)
            for area, points, category, world in candidates:
                metrics = hand_pixel_metrics(points, rgb_image.shape)
                reason = None
                if (metrics["palm_scale_px"] < 8 or metrics["hand_span_px"] < 24
                        or metrics["hand_short_side_px"] < 10):
                    reason = reason or "insufficient_hand_pixels"
                inside = np.mean(np.all((points >= -.05) & (points <= 1.05), axis=1))
                if inside < .90:
                    reason = reason or "incomplete_hand"
                face_context = dict(enabled=False, near_face=False,
                                    contact_confirmed=False, suppressed=False)
                context = None
                if not reason and body_guard is not None:
                    if poses is None:
                        poses, diagnostics = body_guard.detect(rgb_image, state=state)
                        self._last_diagnostics["body_guard"] = dict(diagnostics, checked=True)
                    if self._last_diagnostics["body_guard"]["ready"]:
                        context = body_hand_context(points, rgb_image.shape, poses)
                        current_indices = self._last_diagnostics["body_guard"].get("presentation_body_indices")
                        if context is not None and current_indices is not None:
                            context["presentation_current"] = context["body_index"] in current_indices
                row = dict(points=points, world_landmarks=world, metrics=metrics, area=area, view=view,
                           handedness=getattr(category, "category_name", None),
                           handedness_confidence=float(getattr(category, "score", 0) or 0),
                           rejection_reason=reason, face_context=face_context,
                           body_context=context, recovered=view != full)
                row["recovery_score"] = -1.0 if reason else float(candidate_score(row))
                if not reason:
                    reason = dorsal_body_rejection(context, row.get("resolved_label"))
                    if reason:
                        row["rejection_reason"] = reason
                        row["recovery_score"] = -1.0
                if reason:
                    self._last_diagnostics["rejection_reason"] = reason
                rows.append(row)
            return rows

        rows = observations(full, same_context)
        requested = state.pop("pose_recovery_requested", False)
        context_vetoes = DORSAL_CONTEXT_VETOES
        needs_recovery = (requested or not rows or any(
                              row["recovery_score"] < 1 and row["rejection_reason"] not in context_vetoes
                              for row in rows)
                          or state.get("view") is not None
                          or len(rows) < state.get("expected_hands", 0)
                          or (len(rows) < 2 and state["frames"] % 3 == 0))
        if needs_recovery and time.perf_counter() - started < .055:
            views = self._search_views(rgb_image.shape)
            view = state.get("view")
            if view is None:
                index = state.get("search_index", 0)
                view = views[index % len(views)]
                state["search_index"] = index + 1
            self._last_diagnostics["recovery_attempted"] = True
            alternatives = observations(view, False)
            improved = False
            matched = set()
            # Match across passes only. Never collapse two hands found together
            # by one pass merely because they overlap on screen.
            original_count = len(rows)
            for alternative in alternatives:
                center = alternative["points"][[0, 5, 9, 13, 17]].mean(axis=0)
                distances = []
                for i, row in enumerate(rows[:original_count]):
                    if i in matched:
                        continue
                    other = row["points"][[0, 5, 9, 13, 17]].mean(axis=0)
                    scale = max(row["metrics"]["palm_scale_px"], alternative["metrics"]["palm_scale_px"], 1)
                    distances.append((float(np.linalg.norm((center - other) * [width, height]) / scale), i))
                distance, index = min(distances) if distances else (float("inf"), -1)
                if distance < .80:
                    matched.add(index)
                    # Changing the detector crop must not rescue a hand that
                    # lacks Dorsal body/presentation evidence in this frame.
                    if rows[index]["rejection_reason"] in context_vetoes:
                        continue
                    if alternative["rejection_reason"] in context_vetoes:
                        rows[index] = alternative
                        improved = True
                    elif alternative["recovery_score"] > rows[index]["recovery_score"]:
                        rows[index] = alternative
                        improved = True
                elif not alternative["rejection_reason"]:
                    rows.append(alternative)
                    improved = True
            if improved:
                state["view"], state["view_misses"] = view, 0
            else:
                state["view_misses"] = state.get("view_misses", 0) + 1
                if state["view_misses"] >= 2:
                    state["view"] = None
        frame_time = state.get("frame_time")
        retain_relaxed_posture(rows, rgb_image.shape, state=state,
                               now=time.monotonic() if frame_time is None else float(frame_time))
        self._last_diagnostics["candidate_count"] = len(rows)
        if any(not row["rejection_reason"] for row in rows):
            self._last_diagnostics["rejection_reason"] = None
        elif rows:
            self._last_diagnostics["rejection_reason"] = rows[0]["rejection_reason"]
        return rows

    def _search_views(self, shape):
        height, width = shape[:2]
        side = max(32, int(min(width, height) * 0.60))
        positions = ((.5, .5), (0, .5), (1, .5), (0, 0), (1, 0), (0, 1), (1, 1))

        # Keep exactly the same bounded recovery candidate set, but put the
        # common distant side-pointing case first.  MediaPipe often needs a
        # +/-90-degree view for Left/Right, and the guide asks users to keep
        # their hand near the centre.  Reordering is safer than lowering any
        # detector threshold or adding more opportunities for a false match.
        full_side_views = [
            (0, 0, width, height, 3, 0),
            (0, 0, width, height, 1, 0),
        ]
        full_remaining_views = [
            (0, 0, width, height, 3, 1),
            (0, 0, width, height, 1, 1),
            (0, 0, width, height, 2, 0),
            (0, 0, width, height, 0, 1),
        ]
        crop_views = [
            (
                round((width - side) * fx),
                round((height - side) * fy),
                side,
                side,
                turn,
                1,
            )
            for fx, fy in positions
            for turn in (3, 1, 0)
        ]
        centre_side_views = crop_views[:2]
        centre_upright_view = crop_views[2:3]
        other_crop_views = crop_views[3:]
        return [
            *full_side_views,
            *centre_side_views,
            *full_remaining_views[:2],
            *centre_upright_view,
            *full_remaining_views[2:],
            *other_crop_views,
        ]

    def detect(self, rgb_image, candidate_score=None):
        self._frame_time = None
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
        self._last_diagnostics["face_guard_error"] = getattr(face_guard, "error", None)
        self._last_diagnostics["face_count"] = len(boxes)
        selected = self._run_view(rgb_image, view, video=not self._recovery_tracking)
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
        if getattr(self, "_body_guard", None) is not None:
            self._body_guard.close()
        if getattr(self, "_face_guard", None) is not None:
            self._face_guard.close()
        for name in ("_landmarker", "_recovery"):
            model = getattr(self, name, None)
            if model is not None:
                model.close()
            setattr(self, name, None)
        self.ready = False
