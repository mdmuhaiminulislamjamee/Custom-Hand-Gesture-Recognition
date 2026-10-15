"""Automatic hand ownership for a single RGB camera, local to each session.

Apparent hand scale is a relative closeness cue, never a metric depth measurement.
Compare both physical hands before gesture qualification. The foreground hand
owns recognition after a brief distance hold. When the hands return to the same
distance, a clearly tracked owner keeps control; a change of foreground hand
clears the previous gesture history.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np

from .config import RuntimeConfig
from .runtime import TemporalGate
from .body_guard import RESTING_HAND_REASON


@dataclass(frozen=True)
class SelectionPolicy:
    minimum_palm_pixels: float = 18.0
    minimum_span_pixels: float = 40.0
    minimum_short_side_pixels: float = 12.0
    minimum_dorsal_palm_pixels: float = 24.0
    closer_palm_ratio: float = 1.15
    closer_projected_world_ratio: float = 1.10
    equal_projected_world_ratio: float = 1.05
    overlap_palm_ratio: float = 0.50
    choice_hold_seconds: float = 0.25
    choice_frames: int = 3
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
        expanded_thumb_direction_recovery=bool(supported and pose.get("expanded_thumb_direction_recovery", False)),
        continued_direction_recovery=bool(supported and direction.get("continued_direction", False)),
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
        self._empty_since = None
        self._ownership_reset = False

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
        if mismatch and track.id == self.active_id and row.get("handedness_confidence", 0) >= .75:
            # A newcomer at the missing owner's screen position must not
            # inherit its identity simply because its palm has a similar shape.
            return 100.0
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
        # A locked hand can return at another screen position after a pose
        # rejection or a quick movement. Reacquire its unique confident side
        # without transferring control to the opposite/previously tracked hand.
        if self.active_id in self.tracks and self.active_id not in best[1]:
            owner = self.tracks[self.active_id]
            returning = [i for i, row in enumerate(rows)
                         if owner.handedness and row.get("handedness") == owner.handedness
                         and row.get("handedness_confidence", 0) >= .70]
            if len(returning) == 1 and best[1][returning[0]] is None:
                i = returning[0]
                keys = list(best[1])
                keys[i] = self.active_id
                rows[i]["identity_reacquired"] = True
                return tuple(keys), False
        ambiguous = (len(rows) == 2 and len(assignments) > 1
                     and all(key is not None for key in best[1])
                     and all(key is not None for key in assignments[1][1])
                     and assignments[1][0] - best[0] < 0.20)
        return best[1], ambiguous

    def _closer_metrics(self, near, far, ratio):
        """Require agreement between palm segments to reduce orientation bias."""
        near_segments = np.asarray(near["palm_segments_px"])
        far_segments = np.asarray(far["palm_segments_px"])
        segment_ratios = near_segments / np.maximum(far_segments, 1.0)
        spread = float(np.percentile(segment_ratios, 80) / max(np.percentile(segment_ratios, 20), .01))
        return (near["palm_scale_px"] >= far["palm_scale_px"] * ratio
                and np.min(near_segments) >= 8 and np.min(far_segments) >= 8
                and spread <= 1.40)

    def _distance_is_consistent(self, near, far):
        ratios = np.asarray(near["palm_segments_px"]) / np.maximum(far["palm_segments_px"], 1.)
        return bool(float(np.percentile(ratios, 80) / max(np.percentile(ratios, 20), .01)) <= 1.40)

    @staticmethod
    def _projected_world_scale(row, dimensions):
        """Estimate image pixels per world-landmark unit, independent of pose.

        World Z is relative to each hand and cannot give camera distance. The
        corresponding image and world XY segments can, however, reveal how
        large each hand projects after its rotation and finger pose are removed.
        Reject incoherent landmarks rather than treating them as depth evidence.
        """
        world = row.get("world_landmarks")
        if world is None:
            return None
        image = np.asarray(row["points"], dtype=float)
        world = np.asarray(world, dtype=float)
        if (image.shape != (21, 2) or world.shape != (21, 3)
                or not np.isfinite(image).all() or not np.isfinite(world).all()):
            return None
        pairs = ((0, 5), (0, 9), (0, 13), (0, 17), (5, 9), (9, 13),
                 (13, 17), (5, 17), (5, 6), (6, 7), (7, 8), (9, 10),
                 (10, 11), (11, 12), (13, 14), (14, 15), (17, 18))
        ratios = []
        for a, b in pairs:
            world_length = float(np.linalg.norm(world[a, :2] - world[b, :2]))
            pixel_length = float(np.linalg.norm((image[a] - image[b]) * dimensions))
            if world_length >= .006 and pixel_length >= 5:
                ratios.append(pixel_length / world_length)
        if len(ratios) < 7:
            return None
        low, middle, high = np.percentile(ratios, (20, 50, 80))
        if low <= 0 or high / low > 1.45:
            return None
        return float(middle)

    def _compare_two_hands(self, rows, physical, dimensions):
        """Return (nearer index, reason, relative scale, measurement method)."""
        a, b = physical
        scales = [self._projected_world_scale(rows[i], dimensions) for i in physical]
        if all(scale is not None for scale in scales):
            near, far = sorted(range(2), key=lambda i: scales[i], reverse=True)
            ratio = scales[near] / scales[far]
            if ratio >= self.policy.closer_projected_world_ratio:
                return physical[near], None, ratio, "projected world landmarks"
            reason = "hands_same_distance" if ratio <= self.policy.equal_projected_world_ratio else "hand_distance_uncertain"
            return None, reason, ratio, "projected world landmarks"
        ordered = sorted(physical, key=lambda i: rows[i]["metrics"]["palm_scale_px"], reverse=True)
        near, far = (rows[i]["metrics"] for i in ordered)
        ratio = near["palm_scale_px"] / max(far["palm_scale_px"], 1.)
        if self._closer_metrics(near, far, self.policy.closer_palm_ratio):
            return ordered[0], None, ratio, "apparent palm size"
        similar = ratio < self.policy.closer_palm_ratio and self._distance_is_consistent(near, far)
        return None, "hands_same_distance" if similar else "hand_distance_uncertain", ratio, "apparent palm size"

    def _physical_hand(self, row):
        """Idle/context-blocked hands still count for the distance comparison."""
        if row.get("rejection_reason") in {"face_detected_as_hand", "incomplete_hand", "insufficient_hand_pixels"}:
            return False
        metrics = row["metrics"]
        values = np.asarray(metrics["palm_segments_px"], dtype=float)
        return bool(values.shape == (4,) and np.isfinite(values).all() and values.min() >= 8
                    and metrics["palm_scale_px"] >= self.policy.minimum_palm_pixels
                    and metrics["hand_span_px"] >= self.policy.minimum_span_pixels
                    and metrics["hand_short_side_px"] >= self.policy.minimum_short_side_pixels)

    def _release_ownership(self):
        self.active_id = None
        self._ownership_reset = True
        self._clear_pending()
        for track in self.tracks.values():
            track.gate.reset()

    def update(self, rows, image_shape, now):
        self._ownership_reset = False
        shape = tuple(image_shape[:2])
        if self._image_shape is not None and self._image_shape != shape:
            next_id = self._next_id
            self.reset()
            # The runtime sees a new identity and clears its smoothing/hold even
            # when an existing stream changes resolution or its analysis crop.
            self._next_id = next_id
        self._image_shape = shape
        dimensions = np.array([shape[1], shape[0]])
        self._empty_since = None if rows else (now if self._empty_since is None else self._empty_since)
        if self.active_id is not None and self._empty_since is not None and now - self._empty_since >= self.policy.track_expiry_seconds:
            self._release_ownership()
        self.tracks = {key: track for key, track in self.tracks.items()
                       if key == self.active_id or now - track.last_seen < self.policy.track_expiry_seconds}
        previews = [dict(track_id=None, selected=False, landmarks=row["points"].round(6).tolist(),
                         handedness=row.get("handedness"), palm_scale_px=row["metrics"]["palm_scale_px"],
                         gesture=self.config.reject_label, confidence=0.0, eligible=False,
                         reason=row.get("rejection_reason"), body_context=row.get("body_context"),
                         face_context=row.get("face_context")) for row in rows]

        distance_ratio = None
        distance_method = "relative apparent hand scale; not measured depth"
        switch_pending = False

        def result(selected, reason, message):
            # The inactive hand may build its own gate internally, but its
            # gesture must not appear as another recognized/eligible command.
            for i, preview in enumerate(previews):
                if i != selected:
                    preview.update(gesture=self.config.reject_label, confidence=0., eligible=False)
            if selected is not None:
                previews[selected]["selected"] = True
                if switch_pending:
                    previews[selected]["eligible"] = False
            return (None if selected is None else rows[selected], dict(
                reason=reason, message=message,
                selected_track_id=None if selected is None else previews[selected]["track_id"],
                active_track_id=self.active_id, candidate_count=len(rows),
                ownership_reset=self._ownership_reset,
                ownership_locked=self.active_id is not None,
                switch_pending=switch_pending,
                distance_method=distance_method,
                distance_scale_ratio=distance_ratio,
                closer_palm_ratio=self.policy.closer_palm_ratio,
                closer_projected_world_ratio=self.policy.closer_projected_world_ratio,
            ), previews)

        usable = [i for i, row in enumerate(rows) if not row.get("rejection_reason")]
        physical = [i for i, row in enumerate(rows) if self._physical_hand(row)]
        # Track physical hands even when face/body/gesture checks pause their
        # commands. Otherwise the owner freezes at its last allowed position
        # and every later gesture can be rejected as "active hand missing".
        current = {}
        trackable = sorted(set(usable + physical)) if len(rows) <= 2 else []
        for row in rows:
            row["identity_reacquired"] = False
        keys, ambiguous = self._associate([rows[i] for i in trackable], dimensions)
        if len(physical) == 2:
            first, second = (rows[i] for i in physical)
            palm_separation = float(np.linalg.norm(
                (palm_center(first) - palm_center(second)) * dimensions))
            larger_palm = max(first["metrics"]["palm_scale_px"],
                              second["metrics"]["palm_scale_px"])
            if palm_separation < self.policy.overlap_palm_ratio * larger_palm:
                _, overlap_reason, _, _ = self._compare_two_hands(rows, physical, dimensions)
                if overlap_reason:
                    ambiguous = True
        if (len(physical) == 2
                and all(rows[i].get("handedness_confidence", 0) >= .75 for i in physical)
                and rows[physical[0]].get("handedness") == rows[physical[1]].get("handedness")):
            ambiguous = True
        for i, key in zip(trackable, () if ambiguous else keys):
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
            if row["identity_reacquired"]:
                track.gate.reset()
            alpha = 1 - .65 ** (min(dt, .5) * self.config.target_fps)
            track.palm_size += alpha * (metrics["palm_scale_px"] - track.palm_size)
            track.segments += alpha * (np.array(metrics["palm_segments_px"]) - track.segments)
            track.points, track.last_seen = row["points"].copy(), now
            if row.get("handedness_confidence", 0) >= .75:
                track.handedness = row.get("handedness")
            row["track_id"] = key
            previews[i]["track_id"] = key
            if i in usable:
                current[key] = i
            else:
                track.gate.reset()
                track.invalid_since = now
        if (self.active_id is not None and self.active_id not in current
                and now - self.tracks[self.active_id].last_seen >= self.policy.track_expiry_seconds):
            # A different visible hand must not inherit the owner's gesture
            # immediately. Once that owner has truly disappeared, release its
            # lock and build a fresh hold for the observed hand on later frames.
            self._release_ownership()
            return result(None, "active_hand_expired",
                          "Previous hand left; hold a clear gesture to select the visible hand.")
        nearer_index = None
        retained_owner = False
        if len(physical) == 2:
            nearer_index, distance_reason, distance_ratio, distance_method = self._compare_two_hands(
                rows, physical, dimensions)
            # Small scale changes near a decision threshold must not make an
            # established, clearly tracked owner disappear for one frame.
            closer_ratio = (self.policy.closer_projected_world_ratio
                            if distance_method == "projected world landmarks"
                            else self.policy.closer_palm_ratio)
            retained_owner = (distance_reason is not None and not ambiguous
                              and self.active_id in current
                              and (distance_reason == "hands_same_distance"
                                   or distance_ratio < closer_ratio))
            if distance_reason and not retained_owner and not ambiguous:
                self._clear_pending()
                for track in self.tracks.values():
                    track.gate.reset()
                return result(None, distance_reason,
                              "Hands are at similar distance. Bring one hand clearly forward to select it."
                              if distance_reason == "hands_same_distance" else
                              "Hand distance is uncertain. Keep both hands visible and bring one clearly forward.")

        if rows and not usable:
            self._clear_pending()
            for track in self.tracks.values():
                track.gate.reset()
            reason = rows[0]["rejection_reason"]
            messages = {
                RESTING_HAND_REASON: "Relaxed Dorsal below the waist: no gesture. Raise your hand or bend the arm to gesture.",
                "insufficient_hand_pixels": "Move your hand closer and keep the whole hand visible.",
                "incomplete_hand": "Move your hand closer and keep the whole hand visible.",
                "face_contact_estimated": "Hand appears to touch your face. Move it away to resume gestures.",
                "face_contact_pending": "Checking face contact; move your hand away to resume gestures.",
                "face_overlap_depth_unavailable": "Hand overlaps your face; separation is uncertain. Move it away to resume.",
                "face_detected_as_hand": "Face region rejected as a hand: no gesture.",
                "face_context_unavailable": "Face check unavailable; gestures are paused for this frame.",
            }
            message = messages.get(reason, "Hold a clear gesture with the whole hand visible.")
            return result(None, reason, message)
        if len(rows) > 2:
            self._clear_pending()
            for track in self.tracks.values():
                track.gate.reset()
            return result(None, "too_many_candidates", "Hand tracks are unclear. Separate your hands.")
        if ambiguous:
            self._clear_pending()
            for track in self.tracks.values():
                track.gate.reset()
            return result(None, "ambiguous_tracks", "Hold briefly while the hand tracks separate.")

        decisions = {}
        for key, i in current.items():
            row = rows[i]
            metrics = row["metrics"]
            track = self.tracks[key]
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
        # Distance takes precedence over the previous owner and over gesture
        # confidence. A nearer hand with an unclear pose may pause its own
        # command, but must never leave the farther hand in control.
        nearer_key = next((key for key, i in current.items() if i == nearer_index), None)
        switch_confirmed = False
        if (nearer_index is not None and not ambiguous and self.active_id is not None
                and self.active_id != nearer_key):
            if nearer_key is None:
                self._clear_pending()
                return result(None, "nearer_hand_unusable", "The closer hand is not fully visible.")
            if not self._confirm_choice(nearer_key, now):
                switch_pending = True
                if self.active_id in current:
                    return result(current[self.active_id], "confirming_hand_switch",
                                  "Following the selected hand while checking the closer hand.")
                return result(None, "active_hand_missing", "Selected hand hidden; checking the closer hand.")
            switch_confirmed = True
            self._release_ownership()
            previews[nearer_index]["eligible"] = False
        if self.active_id is not None:
            active = self.tracks[self.active_id]
            if self.active_id not in current:
                self._clear_pending()
                return result(None, "active_hand_missing", "Selected hand hidden; waiting briefly to reacquire it.")
            elif active.invalid_since is not None:
                self._clear_pending()
                return result(current[self.active_id], "active_hand_unrecognized",
                              "Following the selected hand; checking its gesture.")
            else:
                self._clear_pending()
                return result(current[self.active_id], "active_hand_locked",
                              "Following the selected hand while distances are close."
                              if retained_owner else "Following the closer hand.")

        if nearer_index is not None:
            # Select the physical foreground hand after a short, stable depth
            # comparison. The main classifier then receives its landmarks even
            # while the gesture gate is still building a hold.
            if nearer_key is None:
                self._clear_pending()
                return result(None, "nearer_hand_unusable", "The closer hand is not fully visible.")
            chosen = nearer_key
        elif len(ready) == 1:
            chosen = ready[0]
            # A second hand still building a valid gesture must get time to qualify.
            competitors = [key for key, d in decisions.items() if key != chosen
                           and d.predicted_gesture != self.config.reject_label]
            if competitors:
                self._clear_pending()
                return result(None, "confirming_both_hands", "Checking both hands; hold your gesture.")
        elif len(ready) == 2:
            self._clear_pending()
            return result(None, "hand_distance_uncertain", "Hand distance is uncertain. Bring one clearly visible hand forward.")
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

        if len(physical) > 1 and not switch_confirmed and not self._confirm_choice(chosen, now):
            return result(None, "confirming_hand_choice", "Confirming the gesture hand; keep holding.")
        self.active_id = chosen
        self._clear_pending()
        return result(current[chosen], "closer_hand_selected" if nearer_index is not None else "gesture_hand_selected",
                      "Following the closer hand." if nearer_index is not None
                      else "Following the first hand with a clear gesture.")
