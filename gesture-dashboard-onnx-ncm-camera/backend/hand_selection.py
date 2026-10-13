"""Automatic hand ownership for a single RGB camera, local to each session.

Palm size is a relative closeness cue, never a metric depth measurement. Gesture
qualification precedes that cue, and a track's evidence never transfers to a
different hand. No finger motion is required for a held static gesture.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np

from .config import RuntimeConfig
from .runtime import TemporalGate
from .body_guard import RESTING_HAND_REASON, DORSAL_BELOW_WAIST_REASON


@dataclass(frozen=True)
class SelectionPolicy:
    minimum_palm_pixels: float = 18.0
    minimum_span_pixels: float = 40.0
    minimum_short_side_pixels: float = 12.0
    minimum_dorsal_palm_pixels: float = 24.0
    closer_palm_ratio: float = 1.35
    switch_palm_ratio: float = 1.60
    choice_hold_seconds: float = 0.25
    choice_frames: int = 3
    release_seconds: float = 0.35
    missing_hold_seconds: float = 0.60
    track_expiry_seconds: float = 0.80


@dataclass
class HandTrack:
    id: int
    points: np.ndarray
    handedness: str | None
    last_seen: float
    palm_size: float
    segments: np.ndarray
    gate: TemporalGate
    invalid_since: float | None = None


def palm_center(row):
    return np.asarray(row["points"])[[0, 5, 9, 13, 17]].mean(axis=0)


def gate_arguments(details, quality):
    """Use the production geometry/open-set rules for every hand candidate."""
    pose = details.get("pose_validation") or {}
    direction = details.get("directional") or {}
    negative = float((quality.get("online_adapter") or {}).get("negative_support", 0))
    supported = bool(pose.get("geometry_supported", False)) and negative <= 0
    return dict(
        known_gesture_mass=float(quality.get("known_gesture_mass", 1.0)),
        pose_valid=bool(pose.get("valid", True)),
        rejection_reason=str(pose.get("reason", "pose geometry rejected")),
        geometry_supported=supported,
        directional_recovery=bool(supported and direction.get("strong_geometry", False)
                                  and direction.get("model_supported_pose", False)),
        sideways_palm_recovery=bool(pose.get("sideways_palm_recovery", False)),
        dorsal_range_recovery=bool(pose.get("dorsal_range_recovery", False)),
    )


class AutomaticHandSelector:
    def __init__(self, config: RuntimeConfig, policy: SelectionPolicy | None = None):
        self.config = config
        self.policy = policy or SelectionPolicy()
        self.reset()

    def reset(self):
        self.tracks: dict[int, HandTrack] = {}
        self.detector_state = {}
        self.active_id = None
        self._next_id = 1
        self._pending_id = None
        self._pending_since = 0.0
        self._pending_frames = 0
        self._image_shape = None

    def _clear_pending(self):
        self._pending_id = None
        self._pending_frames = 0

    def _confirm_choice(self, track_id, now):
        if self._pending_id != track_id:
            self._pending_id = track_id
            self._pending_since = now
            self._pending_frames = 0
        self._pending_frames += 1
        return (self._pending_frames >= self.policy.choice_frames
                and now - self._pending_since >= self.policy.choice_hold_seconds)

    def _match_cost(self, track, row, dimensions):
        old_center = track.points[[0, 5, 9, 13, 17]].mean(axis=0)
        center = palm_center(row)
        scale = max(track.palm_size, row["metrics"]["palm_scale_px"], 1.0)
        distance = float(np.linalg.norm((center - old_center) * dimensions) / scale)
        ratio = row["metrics"]["palm_scale_px"] / max(track.palm_size, 1.0)
        if distance > 2.0 or not 0.45 < ratio < 2.2:
            return 100.0
        old_shape = (track.points - old_center) * dimensions / max(track.palm_size, 1.0)
        new_shape = (row["points"] - center) * dimensions / max(row["metrics"]["palm_scale_px"], 1.0)
        shape = float(np.median(np.linalg.norm(old_shape - new_shape, axis=1)))
        mismatch = bool(track.handedness and row.get("handedness")
                        and track.handedness != row["handedness"])
        return distance + 0.35 * shape + 0.25 * abs(np.log(ratio)) + (0.65 if mismatch else 0)

    def _associate(self, rows, dimensions):
        # Exhaustive one-to-one assignment is tiny for at most two observations.
        ids = list(self.tracks)
        costs = {(i, key): self._match_cost(self.tracks[key], row, dimensions)
                 for i, row in enumerate(rows) for key in ids}
        assignments = []
        for keys in product([None, *ids], repeat=len(rows)):
            used = [key for key in keys if key is not None]
            if len(set(used)) != len(used):
                continue
            cost = sum(2.2 if key is None else costs[i, key] for i, key in enumerate(keys))
            assignments.append((cost, keys))
        assignments.sort(key=lambda item: item[0])
        best = assignments[0]
        ambiguous = (len(rows) == 2 and len(assignments) > 1
                     and all(key is not None for key in best[1])
                     and all(key is not None for key in assignments[1][1])
                     and assignments[1][0] - best[0] < 0.20)
        return best[1], ambiguous

    def _closer(self, near, far, ratio):
        """Require agreement between palm segments to reduce orientation bias."""
        segment_ratios = near.segments / np.maximum(far.segments, 1.0)
        spread = float(np.percentile(segment_ratios, 80) / max(np.percentile(segment_ratios, 20), .01))
        return (near.palm_size >= far.palm_size * ratio
                and np.min(near.segments) >= 8 and np.min(far.segments) >= 8
                and spread <= 1.40)

    def update(self, rows, image_shape, now):
        shape = tuple(image_shape[:2])
        if self._image_shape is not None and self._image_shape != shape:
            next_id = self._next_id
            self.reset()
            # The runtime sees a new identity and clears its smoothing/hold even
            # when an existing stream changes resolution or its analysis crop.
            self._next_id = next_id
        self._image_shape = shape
        dimensions = np.array([shape[1], shape[0]])
        self.tracks = {key: track for key, track in self.tracks.items()
                       if now - track.last_seen < self.policy.track_expiry_seconds}
        if self.active_id not in self.tracks:
            self.active_id = None
        previews = [dict(track_id=None, selected=False, landmarks=row["points"].round(6).tolist(),
                         handedness=row.get("handedness"), palm_scale_px=row["metrics"]["palm_scale_px"],
                         gesture=self.config.reject_label, confidence=0.0, eligible=False,
                         reason=row.get("rejection_reason"), body_context=row.get("body_context"),
                         face_context=row.get("face_context")) for row in rows]

        def result(selected, reason, message):
            if selected is not None:
                previews[selected]["selected"] = True
            return (None if selected is None else rows[selected], dict(
                reason=reason, message=message,
                selected_track_id=None if selected is None else previews[selected]["track_id"],
                active_track_id=self.active_id, candidate_count=len(rows),
                distance_method="relative apparent palm size; not measured depth",
                closer_palm_ratio=self.policy.closer_palm_ratio,
            ), previews)

        usable = [i for i, row in enumerate(rows) if not row.get("rejection_reason")]
        if rows and not usable:
            self._clear_pending()
            for track in self.tracks.values():
                track.gate.reset()
            reason = rows[0]["rejection_reason"]
            messages = {
                RESTING_HAND_REASON: "Resting hand: no gesture. Lift your hand away from your hip or thigh.",
                DORSAL_BELOW_WAIST_REASON: "Dorsal below the waist: no gesture. Raise the hand to command.",
                "body_context_unavailable": "Body posture check unavailable; gestures are paused for this frame.",
                "insufficient_hand_pixels": "Move your hand closer and keep the whole hand visible.",
                "incomplete_hand": "Move your hand closer and keep the whole hand visible.",
            }
            message = messages.get(reason, "Hold a clear gesture with the whole hand visible.")
            return result(None, reason, message)
        if len(rows) > 2:
            self._clear_pending()
            for track in self.tracks.values():
                track.gate.reset()
            return result(None, "too_many_candidates", "Hand tracks are unclear. Separate your hands.")
        if len(usable) == 2:
            a, b = (rows[i] for i in usable)
            separation = np.linalg.norm((palm_center(a) - palm_center(b)) * dimensions)
            if separation < .85 * max(a["metrics"]["palm_scale_px"], b["metrics"]["palm_scale_px"]):
                self._clear_pending()
                for track in self.tracks.values():
                    track.gate.reset()
                return result(None, "hands_overlap", "Separate your hands so both palms can be tracked.")
        keys, ambiguous = self._associate([rows[i] for i in usable], dimensions)
        if ambiguous:
            self._clear_pending()
            for track in self.tracks.values():
                track.gate.reset()
            return result(None, "ambiguous_tracks", "Hold briefly while the hand tracks separate.")

        current = {}
        decisions = {}
        for i, key in zip(usable, keys):
            row = rows[i]
            metrics = row["metrics"]
            if key is None:
                key = self._next_id
                self._next_id += 1
                self.tracks[key] = HandTrack(key, row["points"].copy(), row.get("handedness"), now,
                                             metrics["palm_scale_px"], np.array(metrics["palm_segments_px"]),
                                             TemporalGate(self.config))
            track = self.tracks[key]
            dt = max(.001, now - track.last_seen)
            if dt >= .35:
                track.gate.reset()
            alpha = 1 - .65 ** (min(dt, .5) * self.config.target_fps)
            track.palm_size += alpha * (metrics["palm_scale_px"] - track.palm_size)
            track.segments += alpha * (np.array(metrics["palm_segments_px"]) - track.segments)
            track.points, track.last_seen = row["points"].copy(), now
            if row.get("handedness_confidence", 0) >= .75:
                track.handedness = row.get("handedness")
            current[key] = i
            row["track_id"] = key
            previews[i]["track_id"] = key
            size_ok = (metrics["palm_scale_px"] >= self.policy.minimum_palm_pixels
                       and metrics["hand_span_px"] >= self.policy.minimum_span_pixels
                       and metrics["hand_short_side_px"] >= self.policy.minimum_short_side_pixels)
            args = row.get("gate_args", {})
            label = self.config.class_names[int(np.argmax(row["probabilities"]))] if "probabilities" in row else None
            dorsal_ok = label != "dorsal" or metrics["palm_scale_px"] >= self.policy.minimum_dorsal_palm_pixels
            if not size_ok or not dorsal_ok:
                reason = "hand_too_small" if not size_ok else "dorsal_hand_too_small"
                decision = track.gate.reject(reason)
            elif "probabilities" not in row:
                decision = track.gate.reject("unusable_hand")
            else:
                decision = track.gate.update(row["probabilities"], now=now, **args)
            valid = decision.predicted_gesture != self.config.reject_label
            track.invalid_since = None if valid else (now if track.invalid_since is None else track.invalid_since)
            decisions[key] = decision
            previews[i].update(gesture=decision.predicted_gesture, confidence=decision.confidence,
                               eligible=decision.execute, reason=decision.reason)
        for key, track in self.tracks.items():
            if key not in current:
                track.gate.reset()

        ready = [key for key, decision in decisions.items() if decision.execute]
        # Preserve ownership through brief occlusions, without emitting stale poses.
        if self.active_id is not None:
            active = self.tracks[self.active_id]
            if self.active_id not in current:
                self._clear_pending()
                if now - active.last_seen < self.policy.missing_hold_seconds:
                    return result(None, "active_hand_missing", "Active hand hidden; waiting for it to return.")
                self.active_id = None
            elif active.invalid_since is not None:
                if now - active.invalid_since < self.policy.release_seconds:
                    self._clear_pending()
                    return result(None, "active_hand_releasing", "Waiting for the active gesture to release.")
                self.active_id = None
            else:
                challengers = [key for key in ready if key != self.active_id
                               and self._closer(self.tracks[key], active, self.policy.switch_palm_ratio)]
                if challengers:
                    challenger = challengers[0]
                    if self._confirm_choice(challenger, now):
                        self.active_id = challenger
                        self._clear_pending()
                        return result(current[challenger], "closer_hand_selected", "Selected the clearly closer gesture hand.")
                    return result(None, "confirming_closer_hand", "Confirming the closer hand; hold your gesture.")
                self._clear_pending()
                return result(current[self.active_id], "active_hand_locked", "Following the active hand.")

        if len(ready) == 1:
            chosen = ready[0]
            # A second hand still building a valid gesture must get time to qualify.
            competitors = [key for key, d in decisions.items() if key != chosen
                           and d.predicted_gesture != self.config.reject_label]
            if competitors:
                self._clear_pending()
                return result(None, "confirming_both_hands", "Checking both hands; hold your gesture.")
        elif len(ready) == 2:
            ordered = sorted(ready, key=lambda key: self.tracks[key].palm_size, reverse=True)
            if not self._closer(self.tracks[ordered[0]], self.tracks[ordered[1]], self.policy.closer_palm_ratio):
                self._clear_pending()
                return result(None, "ambiguous_hands", "Both hands have gestures. Bring the intended hand forward or relax the other.")
            chosen = ordered[0]
        else:
            self._clear_pending()
            small = any(p["reason"] in {"hand_too_small", "dorsal_hand_too_small"} for p in previews)
            # Preserve a single usable observation for feedback and the existing
            # temporal classifier, which still prevents an unconfirmed action.
            if len(rows) == 1 and current and not small:
                return result(0, "confirming_gesture", "Hold a clear gesture to activate this hand.")
            return result(None, "hand_too_small" if small else "waiting_for_gesture",
                          "Move your hand closer for reliable finger detail." if small
                          else "Hold one clear gesture; checking the visible hands.")

        if len(rows) > 1 and not self._confirm_choice(chosen, now):
            return result(None, "confirming_hand_choice", "Confirming the gesture hand; keep holding.")
        self.active_id = chosen
        self._clear_pending()
        return result(current[chosen], "closer_hand_selected" if len(ready) > 1 else "gesture_hand_selected",
                      "Selected the clearly closer gesture hand." if len(ready) > 1 else "Following the hand with a clear gesture.")
