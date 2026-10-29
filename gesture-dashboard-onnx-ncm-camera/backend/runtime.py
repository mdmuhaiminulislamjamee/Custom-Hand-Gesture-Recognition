from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
import math
import time

import numpy as np

from .config import (DISABLED_GESTURES, DORSAL_REJECTION_REASON,
                     OPEN_HAND_LOWER_ARC_REJECTION_REASON, OPEN_PALM_ANGLE_REJECTION_REASON, RuntimeConfig,
                     STOP_INVERTED_ANGLE_REJECTION_REASON)


class InferenceStartLimiter:
    """Serialize backend-wide inference starts without locking model execution."""

    HARD_MAXIMUM_FPS = 10.0

    def __init__(
        self,
        frame_interval_ms: float,
        *,
        clock: Callable[[], float] = time.perf_counter,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self._clock = clock
        self._sleep = sleep
        self._lock = asyncio.Lock()
        self._last_start_at: float | None = None
        self._interval_seconds = self._effective_interval(frame_interval_ms)

    @classmethod
    def _effective_interval(cls, frame_interval_ms: float) -> float:
        requested_seconds = float(frame_interval_ms) / 1000.0
        if not math.isfinite(requested_seconds) or requested_seconds <= 0.0:
            raise ValueError("Inference frame interval must be a positive finite value.")
        return max(requested_seconds, 1.0 / cls.HARD_MAXIMUM_FPS)

    @property
    def interval_seconds(self) -> float:
        return self._interval_seconds

    async def configure(self, frame_interval_ms: float) -> None:
        """Apply a new interval while retaining the last admitted start time."""

        interval_seconds = self._effective_interval(frame_interval_ms)
        async with self._lock:
            self._interval_seconds = interval_seconds

    async def wait_for_start(self) -> float:
        """Wait for and record the next globally admissible inference start."""

        async with self._lock:
            if self._last_start_at is not None:
                earliest_start = self._last_start_at + self._interval_seconds
                while True:
                    remaining = earliest_start - self._clock()
                    if remaining <= 0.0:
                        break
                    await self._sleep(remaining)
            started_at = self._clock()
            self._last_start_at = started_at
            return started_at


def probability_ema(probability_rows: np.ndarray, alpha: float = 0.65) -> np.ndarray:
    rows = np.asarray(probability_rows, dtype=np.float64)
    if rows.ndim != 2 or not len(rows):
        raise ValueError("EMA requires at least one probability row.")
    if not 0.0 < float(alpha) <= 1.0:
        raise ValueError("EMA alpha must be in (0, 1].")
    output: list[np.ndarray] = []
    state = rows[0].copy()
    state /= max(float(state.sum()), 1e-12)
    output.append(state.copy())
    for row in rows[1:]:
        state = float(alpha) * row + (1.0 - float(alpha)) * state
        state = np.clip(state, 1e-12, None)
        state /= state.sum()
        output.append(state.copy())
    return np.vstack(output)


def trailing_equal_count(values: list[int] | np.ndarray) -> int:
    values = list(values)
    if not values:
        return 0
    final_value = values[-1]
    count = 0
    for value in reversed(values):
        if value != final_value:
            break
        count += 1
    return count


@dataclass(slots=True)
class TemporalDecision:
    execute: bool
    predicted_gesture: str
    confidence: float
    stable_frames: int
    frames_used: int
    reason: str
    probabilities: dict[str, float]
    held_seconds: float
    probability_margin: float = 0.0
    known_gesture_mass: float = 1.0


@dataclass(slots=True)
class TemporalGate:
    config: RuntimeConfig
    max_history: int = 30
    _history: deque[np.ndarray] = field(default_factory=lambda: deque(maxlen=30))
    _label_started_at: float | None = None
    _last_label: str | None = None
    _rejected_frames: int = 0
    _last_update_at: float | None = None

    def reset(self) -> None:
        self._history.clear()
        self._label_started_at = None
        self._last_label = None
        self._rejected_frames = 0
        self._last_update_at = None

    def reject(
        self,
        reason: str,
        *,
        probabilities: np.ndarray | None = None,
        known_gesture_mass: float = 0.0,
    ) -> TemporalDecision:
        self._rejected_frames += 1
        # Allow a 1-frame grace dropout for a held gesture before discarding historical hold state
        if self._rejected_frames >= 2 or not self._history:
            self._history.clear()
            self._label_started_at = None
            self._last_label = None
            self._last_update_at = None
        values = (
            np.asarray(probabilities, dtype=np.float64).reshape(-1)
            if probabilities is not None
            else np.zeros(len(self.config.class_names), dtype=np.float64)
        )
        if len(values) != len(self.config.class_names):
            values = np.zeros(len(self.config.class_names), dtype=np.float64)
        total = float(values.sum())
        if total > 1e-12:
            values = values / total
        return TemporalDecision(
            execute=False,
            predicted_gesture=self.config.reject_label,
            confidence=0.0,
            stable_frames=0,
            frames_used=0,
            reason=reason,
            probabilities={
                name: float(values[index])
                for index, name in enumerate(self.config.class_names)
            },
            held_seconds=0.0,
            probability_margin=0.0,
            known_gesture_mass=float(max(0.0, min(1.0, known_gesture_mass))),
        )

    def update(
        self,
        probabilities: np.ndarray,
        now: float | None = None,
        *,
        known_gesture_mass: float = 1.0,
        pose_valid: bool = True,
        rejection_reason: str | None = None,
        geometry_supported: bool = False,
        directional_recovery: bool = False,
        sideways_palm_recovery: bool = False,
        stop_inverted_recovery: bool = False,
        horizontal_direction_recovery: bool = False,
        dorsal_range_recovery: bool = False,
        expanded_thumb_direction_recovery: bool = False,
        continued_direction_recovery: bool = False,
        peace_pair_recovery: bool = False,
    ) -> TemporalDecision:
        now = time.monotonic() if now is None else float(now)
        row = np.asarray(probabilities, dtype=np.float64).reshape(-1)
        if len(row) != len(self.config.class_names):
            raise ValueError("Probability count does not match the configured class order.")
        if (
            not np.isfinite(row).all()
            or (row < 0.0).any()
            or float(row.sum()) <= 1e-12
        ):
            return self.reject("invalid classifier probabilities")
        if not np.isfinite(float(known_gesture_mass)):
            return self.reject("invalid known-gesture mass")
        row = np.clip(row, 1e-12, None)
        row /= row.sum()
        if self.config.class_names[int(row.argmax())] in DISABLED_GESTURES:
            # A hard negative clears the hold immediately, including its grace
            # frame. It must never become a second-best command after masking.
            self.reset()
            return self.reject(DORSAL_REJECTION_REASON, probabilities=row,
                               known_gesture_mass=known_gesture_mass)
        order = np.sort(row)
        raw_confidence = float(order[-1])
        probability_margin = float(order[-1] - order[-2]) if len(order) > 1 else raw_confidence
        if not pose_valid:
            if rejection_reason in {DORSAL_REJECTION_REASON, OPEN_HAND_LOWER_ARC_REJECTION_REASON,
                                    STOP_INVERTED_ANGLE_REJECTION_REASON, OPEN_PALM_ANGLE_REJECTION_REASON}:
                self.reset()
            return self.reject(
                rejection_reason or "pose geometry rejected",
                probabilities=row,
                known_gesture_mass=known_gesture_mass,
            )
        geometry_gesture = self.config.class_names[int(row.argmax())]
        recovery_mass_ok = bool(
            known_gesture_mass >= self.config.geometry_recovery_mass_floor
        )
        expanded_thumb_fallback = bool(
            expanded_thumb_direction_recovery and geometry_supported and directional_recovery
            # The HUD rounds mass to two decimals: a displayed 0.01 may be
            # as low as 0.005. This exception requires the strict thumb and
            # single-index silhouette in addition to a longer hold.
            and geometry_gesture in {"left", "right"} and known_gesture_mass >= .005
        )
        continued_direction_fallback = bool(
            continued_direction_recovery and geometry_supported and directional_recovery
            and geometry_gesture in {"left", "right", "up", "down"}
        )
        horizontal_fallback = bool(
            horizontal_direction_recovery and geometry_supported and directional_recovery
            and geometry_gesture in {"left", "right"}
        )
        directional_fallback = bool(
            directional_recovery and geometry_supported
            and geometry_gesture in {"left", "right", "up", "down"}
            # Preserve the tightly bounded camera-domain Down silhouettes that
            # motivated the geometry fallback. Other directions need a small
            # amount of classifier vocabulary evidence to reject held rock/
            # peace hard negatives that can project like a pointing finger.
            and (geometry_gesture == "down" or recovery_mass_ok
                 or expanded_thumb_fallback or continued_direction_fallback or horizontal_fallback)
        )
        supported_hand = bool(
            geometry_supported
            and geometry_gesture in {"open_palm", "like", "fist", "thumb_down"}
            # The shipped classifier learned mostly upright palms. Allow a
            # small camera-angle mass miss only after strict palm geometry and
            # visible-surface validation have identified Open Palm.
            and ((known_gesture_mass >= 0.12 or sideways_palm_recovery)
                 if geometry_gesture == "open_palm"
                 else recovery_mass_ok)
        )
        peace_fallback = bool(
            geometry_supported
            and geometry_gesture == "peace"
            # The reported clear two-finger pose displays 0.01 mass. Only a
            # separately verified pair may recover that miss; ordinary Peace
            # geometry still needs the established vocabulary evidence.
            and (recovery_mass_ok or (peace_pair_recovery and known_gesture_mass >= .005))
        )
        stop_fallback = bool(
            stop_inverted_recovery and geometry_supported
            and geometry_gesture == "stop_inverted"
        )
        if known_gesture_mass < self.config.known_mass_floor and not (
            directional_fallback or supported_hand or peace_fallback or stop_fallback
        ):
            return self.reject(
                "outside the active gesture vocabulary",
                probabilities=row,
                known_gesture_mass=known_gesture_mass,
            )
        if raw_confidence < self.config.confidence_floor:
            return self.reject(
                "below confidence floor",
                probabilities=row,
                known_gesture_mass=known_gesture_mass,
            )
        if probability_margin < self.config.probability_margin_floor:
            return self.reject(
                "ambiguous probability margin",
                probabilities=row,
                known_gesture_mass=known_gesture_mass,
            )
        # The backend-safe profile starts a fresh hold whenever the top label
        # changes.  The responsive camera profile instead keeps a three-frame
        # vote window, matching the mobile runtime's ordinary two-of-three
        # confirmation while still discarding stale history after an outage.
        label_changed = bool(
            self._history
            and int(self._history[-1].argmax()) != int(row.argmax())
        )
        stale_history = bool(
            self._last_update_at is not None and now - self._last_update_at >= 0.5
        )
        if self._history and (
            stale_history
            or (label_changed and not self.config.responsive_confirmation)
        ):
            self.reset()
        if self._last_update_at is not None and now <= self._last_update_at:
            return self.reject("duplicate or out-of-order frame timestamp")
        self._last_update_at = now
        self._rejected_frames = 0
        self._history.append(row)
        ema_rows = probability_ema(np.vstack(self._history), self.config.ema_alpha)
        labels = ema_rows.argmax(axis=1)
        final_probabilities = (
            row if self.config.responsive_confirmation else ema_rows[-1]
        )
        final_index = int(np.argmax(final_probabilities))
        gesture = self.config.class_names[final_index]
        confidence = float(final_probabilities[final_index])
        if self.config.responsive_confirmation:
            window_size = max(1, int(self.config.confirmation_window_frames))
            recent_rows = list(self._history)[-window_size:]
            stable_frames = sum(
                int(np.argmax(value)) == final_index for value in recent_rows
            )
        else:
            stable_frames = trailing_equal_count(labels)

        if gesture != self._last_label:
            self._last_label = gesture
            self._label_started_at = now
        label_started_at = (
            now if self._label_started_at is None else self._label_started_at
        )
        held_seconds = max(0.0, now - label_started_at)
        required_hold = self.config.minimum_hold_seconds
        if directional_fallback and known_gesture_mass < self.config.known_mass_floor:
            required_hold = max(required_hold, 0.50 if expanded_thumb_fallback or horizontal_fallback else 0.40)
        if supported_hand and known_gesture_mass < self.config.known_mass_floor:
            required_hold = max(required_hold, 0.45)
        if peace_fallback and known_gesture_mass < self.config.known_mass_floor:
            required_hold = max(required_hold, 0.50)
        if stop_fallback and known_gesture_mass < self.config.known_mass_floor:
            required_hold = max(required_hold, 0.45)
        strong_confirmation = bool(
            self.config.responsive_confirmation
            and known_gesture_mass >= self.config.known_mass_floor
            and raw_confidence >= self.config.strong_confirmation_confidence
            and probability_margin >= self.config.strong_confirmation_margin
        )
        required_stable_frames = (
            max(1, int(self.config.confirmation_votes_required))
            if self.config.responsive_confirmation
            else self.config.stable_frames_required
        )
        execute = bool(
            confidence >= self.config.confidence_floor
            and (
                strong_confirmation
                or (
                    stable_frames >= required_stable_frames
                    and held_seconds >= required_hold
                )
            )
        )
        if confidence < self.config.confidence_floor:
            reason = "below confidence floor"
        elif strong_confirmation:
            reason = "strong gesture confirmed immediately"
        elif stable_frames < required_stable_frames:
            reason = "insufficient consecutive stable frames"
        elif held_seconds < required_hold:
            reason = "gesture hold requirement not reached"
        else:
            reason = (
                "stable directional geometry"
                if directional_fallback and known_gesture_mass < self.config.known_mass_floor
                else "stable Peace finger geometry"
                if peace_fallback and known_gesture_mass < self.config.known_mass_floor
                else "stable Stop inverted hand geometry"
                if stop_fallback and known_gesture_mass < self.config.known_mass_floor
                else "stable and confident"
            )
        return TemporalDecision(
            execute=execute,
            predicted_gesture=gesture,
            confidence=confidence,
            stable_frames=stable_frames,
            frames_used=len(ema_rows),
            reason=reason,
            probabilities={
                name: float(final_probabilities[index])
                for index, name in enumerate(self.config.class_names)
            },
            held_seconds=held_seconds,
            probability_margin=probability_margin,
            known_gesture_mass=float(max(0.0, min(1.0, known_gesture_mass))),
        )
