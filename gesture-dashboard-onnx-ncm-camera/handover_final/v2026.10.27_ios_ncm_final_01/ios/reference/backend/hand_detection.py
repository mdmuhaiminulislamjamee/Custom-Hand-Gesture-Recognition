"""Native-detail hand tracking with a bounded orientation/crop recovery pass."""
from __future__ import annotations

import threading
import time
from pathlib import Path

import numpy as np

DETECTOR_SETTINGS = {
    "maximum_hands": 2,
    "live_hand_selection": "spatial track continuity; three-frame handedness consensus; filtered foreground hand switch",
    "live_minimum_palm_pixels": 18,
    "live_minimum_dorsal_palm_pixels": 24,
    "running_mode": "VIDEO with two independent recovery streams and IMAGE acquisition",
    "minimum_detection_confidence": 0.55,
    "minimum_presence_confidence": 0.55,
    "minimum_tracking_confidence": 0.60,
    "recovery_detection_confidence": 0.20,
    "recovery_presence_confidence": 0.30,
    "face_guard": "disabled; gestures may overlap the face",
    "body_guard": "disabled; all gestures use hand landmarks only",
    "recovery_policy": "independent persistent VIDEO recovery views for both hands; bidirectional original/contrast crop search; full-frame scan every frame",
    "recovery_location_hold_ms": 650,
    "recovery_confirmation_frames": 2,
    "minimum_palm_pixels": 8,
    "minimum_hand_span_pixels": 24,
    "minimum_hand_short_side_pixels": 10,
    "maximum_analysis_dimension": 960,
    "maximum_detection_passes_per_frame": 3,
    "recovery_start_budget_ms": 55,
    "peer_recovery_start_budget_ms": 85,
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
    def __init__(self, model_path: Path):
        self.ready = False
        self.error = None
        self.running_mode = "VIDEO"
        self._landmarker = self._recovery = self._mp = None
        self._recovery_video = None
        self._recovery_video_context = None
        self._recovery_videos = {}
        self._recovery_stream_context = None
        self._live_recovery_context = None
        self._model_path = model_path
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
            self.ready = True
        except Exception as error:
            self.error = f"MediaPipe initialization failed: {error}"
            self.close()

    def _run_candidates(self, rgb, view, *, video):
        x, y, width, height, turns, enhance = view
        crop = rgb[y:y + height, x:x + width]
        live_context = getattr(self, "_live_recovery_context", None)
        if not video:
            # A bright window can hide an underexposed hand in the full-frame
            # exposure statistic. Enhance only an actually dark recovery crop.
            import cv2
            if enhance or (live_context is None and
                           float(np.median(cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY))) < 85):
                lab = cv2.cvtColor(crop, cv2.COLOR_RGB2LAB)
                lab[:, :, 0] = cv2.createCLAHE(clipLimit=1.8, tileGridSize=(8, 8)).apply(lab[:, :, 0])
                crop = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB,
                               data=np.ascontiguousarray(np.rot90(crop, turns)))
        track_recovery = bool(not video and live_context is not None and (
            live_context.get("view") == view
            or (getattr(self, "_recovery_stream_context", None) is live_context
                and tuple(view) in getattr(self, "_recovery_videos", {}))
            or any(item.get("view") == view for item in live_context.get("recent_hands", []))
            or view == self._startup_recovery_view(rgb.shape)))
        if video or track_recovery:
            landmarker = self._landmarker
            if track_recovery:
                streams = getattr(self, "_recovery_videos", {})
                if getattr(self, "_recovery_stream_context", None) is not live_context:
                    for previous in streams.values():
                        previous.close()
                    streams = {}
                    self._recovery_stream_context = live_context
                key = tuple(view)
                landmarker = streams.pop(key, None)
                if landmarker is None:
                    # Two independent crop coordinate systems must not reset
                    # one another. Bound the pool while preserving the most
                    # recently used tracking stream during exploratory search.
                    if len(streams) >= 2:
                        streams.pop(next(iter(streams))).close()
                    landmarker = self._mp.tasks.vision.HandLandmarker.create_from_options(
                        self._mp.tasks.vision.HandLandmarkerOptions(
                            base_options=self._mp.tasks.BaseOptions(model_asset_path=str(self._model_path)),
                            num_hands=2, min_hand_detection_confidence=.20,
                            min_hand_presence_confidence=.30, min_tracking_confidence=.60,
                            running_mode=self._mp.tasks.vision.RunningMode.VIDEO))
                streams[key] = landmarker
                self._recovery_videos = streams
                self._recovery_video = landmarker
                self._recovery_video_context = (live_context, view)
                self._last_diagnostics["recovery_tracking"] = True
            with self._timestamp_lock:
                frame_time = getattr(self, "_frame_time", None)
                current = (int(frame_time * 1000) if frame_time is not None
                           else int(time.monotonic_ns() // 1_000_000))
                timestamp = max(current, self._last_timestamp_ms + 1)
                self._last_timestamp_ms = timestamp
            self._last_diagnostics.update(timestamp_ms=timestamp, timestamp_adjusted=timestamp != current)
            result = landmarker.detect_for_video(image, timestamp)
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
        self._last_diagnostics["body_guard"] = dict(
            enabled=False, ready=False, checked=False, body_count=0, error=None)

        def observations(view, video):
            rows = []
            self._live_recovery_context = state if view != full else None
            try:
                candidates = self._run_candidates(rgb_image, view, video=video)
            finally:
                self._live_recovery_context = None
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
                row = dict(points=points, world_landmarks=world, metrics=metrics, area=area, view=view,
                           handedness=getattr(category, "category_name", None),
                           handedness_confidence=float(getattr(category, "score", 0) or 0),
                           rejection_reason=reason, face_context=face_context,
                           recovered=view != full)
                row["recovery_score"] = -1.0 if reason else float(candidate_score(row))
                if (not reason and row.get("resolved_label") == "dorsal"
                        and not np.all((points >= .01) & (points <= .99))):
                    reason = "dorsal_hand_cropped"
                    row["rejection_reason"] = reason
                    row["recovery_score"] = -1.0
                if reason:
                    self._last_diagnostics["rejection_reason"] = reason
                rows.append(row)
            return rows

        rows = observations(full, same_context)
        if any(not row["rejection_reason"] and row["recovery_score"] >= 1 for row in rows):
            state["ever_qualified"] = True
        requested = state.pop("pose_recovery_requested", False)
        visited = set()
        self._last_diagnostics["detection_passes"] = 1
        # One recovery pass can maintain one hand while the other requires a
        # separate view. Stop as soon as two fresh hands are present; never
        # replace a missing observation with old landmark coordinates.
        for recovery_pass in range(2):
            needs_recovery = (requested or len(rows) < 2 or any(
                                  row["recovery_score"] < 1 for row in rows)
                              or (recovery_pass == 0 and state.get("view") is not None))
            budget = .055 if recovery_pass == 0 else .085
            if not needs_recovery or time.perf_counter() - started >= budget:
                break
            state["visited_views"] = visited
            view = self._live_recovery_view(rgb_image.shape, state, rows)
            if view in visited:
                break
            visited.add(view)
            self._last_diagnostics["recovery_attempted"] = True
            self._last_diagnostics["detection_passes"] += 1
            self._last_diagnostics.update(recovery_view=list(view[:4]),
                                          recovery_rotation_degrees=int(view[4] * 90),
                                          recovery_contrast=bool(view[5]))
            alternatives = observations(view, False)
            if any(not row["rejection_reason"] and row["recovery_score"] >= 1
                   for row in alternatives):
                state["ever_qualified"] = True
                state["view_last_seen"] = state.get("frame_time", time.monotonic())
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
                    if (alternative["recovery_score"] > rows[index]["recovery_score"]
                            or (alternative["recovery_score"] == rows[index]["recovery_score"]
                                and not rows[index]["recovered"]
                                and view == state.get("view"))):
                        rows[index] = alternative
                        improved = True
                elif not alternative["rejection_reason"]:
                    rows.append(alternative)
                    improved = True
            if improved and any(not row["rejection_reason"] and row["recovery_score"] >= 0
                                for row in alternatives):
                state["view"], state["view_misses"] = view, 0
            else:
                state["view_misses"] = state.get("view_misses", 0) + 1
                if state["view_misses"] >= 2:
                    state["view"] = None
            self._remember_hands(state, rows)
            requested = False
            if len(rows) >= 2:
                break
        state.pop("visited_views", None)
        self._remember_hands(state, rows)
        self._last_diagnostics["candidate_count"] = len(rows)
        if any(not row["rejection_reason"] for row in rows):
            self._last_diagnostics["rejection_reason"] = None
        elif rows:
            self._last_diagnostics["rejection_reason"] = rows[0]["rejection_reason"]
        return rows

    @staticmethod
    def _hand_crop(points, shape, turns=0):
        """Keep native hand pixels and enough wrist context for palm detection."""
        height, width = shape[:2]
        pixels = np.asarray(points) * [width, height]
        center = (pixels.min(axis=0) + pixels.max(axis=0)) / 2
        side = max(96, float(np.ptp(pixels, axis=0).max()) * 2.6)
        crop_width, crop_height = min(width, round(side)), min(height, round(side))
        x = round(float(np.clip(center[0] - crop_width / 2, 0, width - crop_width)))
        y = round(float(np.clip(center[1] - crop_height / 2, 0, height - crop_height)))
        return (x, y, crop_width, crop_height, turns, 1)

    @staticmethod
    def _missing_peer_crop(points, shape, turns=0, *, opposite=True, enhance=1):
        """Zoom the side of the torso opposite a sole detected hand.

        Search either side: a person need not be centred in the camera frame.
        """
        height, width = shape[:2]
        side = max(96, round(min(width, height) * .68))
        side = min(side, width, height)
        palm = np.asarray(points)[[0, 5, 9, 13, 17]].mean(axis=0)
        palm_x, palm_y = float(palm[0] * width), float(palm[1] * height)
        direction = 1.0 if palm_x < width / 2 else -1.0
        if not opposite:
            direction *= -1
        target_x = palm_x + direction * max(width * .22, side * .55)
        x = round(float(np.clip(target_x - side / 2, 0, width - side)))
        y = round(float(np.clip(palm_y - side / 2, 0, height - side)))
        return (x, y, side, side, turns, enhance)

    @staticmethod
    def _remember_hands(state, rows):
        now = state.get("frame_time")
        now = time.monotonic() if now is None else float(now)
        recent = [item for item in state.get("recent_hands", [])
                  if 0 <= now - item["seen"] <= .65]
        for row in rows:
            if row.get("rejection_reason") in {"insufficient_hand_pixels", "incomplete_hand"}:
                continue
            # This history only chooses where to look in the next image. It is
            # never returned as a hand observation or used to execute commands.
            center = row["points"][[0, 5, 9, 13, 17]].mean(axis=0)
            recent = [old for old in recent if np.linalg.norm(
                center - old["points"][[0, 5, 9, 13, 17]].mean(axis=0)) > .10]
            recent.append(dict(points=row["points"].copy(), seen=now, turns=row["view"][4],
                               view=row["view"]))
        state["recent_hands"] = recent[-2:]

    def _live_recovery_view(self, shape, state, rows):
        """Track a productive view continuously; search again on fresh misses."""
        now = state.get("frame_time")
        now = time.monotonic() if now is None else float(now)
        current = [dict(points=row["points"], turns=row["view"][4]) for row in rows
                   if row.get("rejection_reason") not in {"insufficient_hand_pixels", "incomplete_hand"}]
        hints = [item for item in state.get("recent_hands", [])
                 if 0 <= now - item["seen"] <= .65]
        visited = state.get("visited_views", set())
        attempt = state.get("recovery_attempts", 0)
        state["recovery_attempts"] = attempt + 1
        missing_hints = [item for item in hints if all(np.linalg.norm(
            item["points"][[0, 5, 9, 13, 17]].mean(axis=0)
            - row["points"][[0, 5, 9, 13, 17]].mean(axis=0)) > .10 for row in current)]
        if len(current) == 1 and missing_hints:
            hint = missing_hints[attempt % len(missing_hints)]
            previous = hint.get("view")
            if previous is not None and previous != (0, 0, shape[1], shape[0], 0, 0):
                if previous not in visited:
                    return previous
            crop = self._hand_crop(hint["points"], shape, hint["turns"])
            if crop not in visited:
                return crop
        productive = state.get("view")
        if (len(current) == 1 and productive is not None and productive not in visited
                and 0 <= now - state.get("view_last_seen", -float("inf")) <= .65
                and state.get("view_misses", 0) < 2):
            x, y, width, height = productive[:4]
            pixels = current[0]["points"] * [shape[1], shape[0]]
            inside = np.all((pixels >= [x, y]) & (pixels <= [x + width, y + height]), axis=1)
            if float(inside.mean()) >= .90:
                # Keep the established hand's better finger view as well as
                # searching for a peer. The next bounded pass can search the
                # other side; discovering a peer must not destabilize this hand.
                return productive
        if not current:
            for hint in reversed(hints):
                previous = hint.get("view")
                if (previous is not None and previous not in visited
                        and previous != (0, 0, shape[1], shape[0], 0, 0)):
                    return previous
                crop = self._hand_crop(hint["points"], shape, hint["turns"])
                if crop not in visited:
                    return crop
        if len(current) == 1:
            # Do not spend the only extra pass zooming the hand MediaPipe has
            # already found.  Search the adjacent torso region, alternating
            # upright and side rotations for an edge-on peer gesture.
            search = state.get("peer_search_index", 0)
            state["peer_search_index"] = search + 1
            turns = (0, 3, 1)[(search // 2) % 3]
            return self._missing_peer_crop(current[0]["points"], shape, turns,
                opposite=search % 2 == 0, enhance=(search // 6) % 2)
        # The full frame was already scanned this frame. Keep the second pass
        # on a productive view so IMAGE re-detection against clothing does not
        # repeatedly break an otherwise stable VIDEO track. Only new pixels
        # can refresh this timestamp; a stale crop expires after 650 ms.
        if (state.get("view") is not None
                and 0 <= now - state.get("view_last_seen", -float("inf")) <= .65
                and state.get("view_misses", 0) < 2 and state["view"] not in visited):
            return state["view"]
        if not state.get("ever_qualified", False) and attempt % 4 != 3:
            upright = self._startup_recovery_view(shape)
            if upright not in visited:
                return upright
        # Prefer a missing hand's last location over the hand already visible.
        hints = missing_hints + current
        # Revisit the upright acquisition stream while the user raises their
        # hand. Exhausting all off-centre rotations first can otherwise spend
        # seconds away from a command that appears just after startup.
        if not hints and attempt % 2 == 0 and self._startup_recovery_view(shape) not in visited:
            return self._startup_recovery_view(shape)
        if hints and attempt % 2 == 0:
            index = state.get("local_search_index", 0)
            state["local_search_index"] = index + 1
            hint = hints[index % len(hints)]
            return self._hand_crop(hint["points"], shape, hint["turns"])
        # Recheck an effective view, but keep exploring if a hand moves or a
        # different orientation is needed. No crop can pin the global search.
        if state.get("view") is not None and attempt % 4 == 1 and state["view"] not in visited:
            return state["view"]
        index = state.get("search_index", 0)
        state["search_index"] = index + 1
        height, width = shape[:2]
        enhanced_full = (0, 0, width, height, 0, 1)
        # Upright commands against a torso need a useful crop on a cold start,
        # before any hand location exists. Try it before the rotation search.
        upright = self._startup_recovery_view(shape)
        # Native contrast can outperform CLAHE on a backlit palm. Cover both
        # sides even when the person is off-centre and no hand history exists.
        side = min(width, height, max(96, round(min(width, height) * .68)))
        acquisition = [(round((width - side) * fx), 0, side, side, turn, 0)
                       for turn in (0, 3, 1) for fx in (0, .5, 1)]
        views = [enhanced_full, upright,
                 *acquisition,
                 *(view for view in self._search_views(shape) if view != enhanced_full)]
        while views[index % len(views)] in visited:
            index += 1
        state["search_index"] = index + 1
        return views[index % len(views)]

    @staticmethod
    def _startup_recovery_view(shape):
        height, width = shape[:2]
        side = max(1, round(min(width, height) * .95))
        return ((width - side) // 2, (height - side) // 2, side, side, 0, 1)

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
        if getattr(self, "_face_guard", None) is not None:
            self._face_guard.close()
        streams = getattr(self, "_recovery_videos", {})
        for model in streams.values():
            model.close()
        self._recovery_videos = {}
        self._recovery_stream_context = None
        if streams:
            self._recovery_video = None
        for name in ("_landmarker", "_recovery", "_recovery_video"):
            model = getattr(self, name, None)
            if model is not None:
                model.close()
            setattr(self, name, None)
        self._recovery_video_context = self._live_recovery_context = None
        self.ready = False
