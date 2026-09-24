from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
import math
import time

import numpy as np

from .config import RuntimeConfig


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
        # Do not let a stale high-confidence EMA survive an invalid/no-hand pose.
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
        order = np.sort(row)
        raw_confidence = float(order[-1])
        probability_margin = float(order[-1] - order[-2]) if len(order) > 1 else raw_confidence
        if not pose_valid:
            return self.reject(
                rejection_reason or "pose geometry rejected",
                probabilities=row,
                known_gesture_mass=known_gesture_mass,
            )
        geometry_gesture = self.config.class_names[int(row.argmax())]
        recovery_mass_ok = bool(
            known_gesture_mass >= self.config.geometry_recovery_mass_floor
        )
        dorsal_fallback = bool(
            geometry_supported and geometry_gesture == "dorsal" and recovery_mass_ok
        )
        directional_fallback = bool(
            directional_recovery and geometry_supported
            and geometry_gesture in {"left", "right", "up", "down"}
            # Preserve the tightly bounded camera-domain Down silhouettes that
            # motivated the geometry fallback. Other directions need a small
            # amount of classifier vocabulary evidence to reject held rock/
            # peace hard negatives that can project like a pointing finger.
            and (geometry_gesture == "down" or recovery_mass_ok)
        )
        supported_hand = bool(
            geometry_supported
            and geometry_gesture in {"open_palm", "like", "fist", "thumb_down"}
            and recovery_mass_ok
        )
        if known_gesture_mass < self.config.known_mass_floor and not (
            dorsal_fallback or directional_fallback or supported_hand
        ):
            return self.reject(
                "outside the ten-gesture vocabulary",
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
        # Start a fresh hold for a confident pose change or capture outage;
        # the previous EMA must never execute an action for the new pose.
        if self._history and (
            int(self._history[-1].argmax()) != int(row.argmax())
            or (self._last_update_at is not None and now - self._last_update_at >= 0.5)
        ):
            self.reset()
        if self._last_update_at is not None and now <= self._last_update_at:
            return self.reject("duplicate or out-of-order frame timestamp")
        self._last_update_at = now
        self._rejected_frames = 0
        self._history.append(row)
        ema_rows = probability_ema(np.vstack(self._history), self.config.ema_alpha)
        labels = ema_rows.argmax(axis=1)
        final_index = int(labels[-1])
        gesture = self.config.class_names[final_index]
        confidence = float(ema_rows[-1, final_index])
        stable_frames = trailing_equal_count(labels)

        if gesture != self._last_label:
            self._last_label = gesture
            self._label_started_at = now
        label_started_at = (
            now if self._label_started_at is None else self._label_started_at
        )
        held_seconds = max(0.0, now - label_started_at)
        required_hold = self.config.minimum_hold_seconds
        if dorsal_fallback and known_gesture_mass < self.config.known_mass_floor:
            required_hold = max(required_hold, 0.35)
        if directional_fallback and known_gesture_mass < self.config.known_mass_floor:
            required_hold = max(required_hold, 0.40)
        if supported_hand and known_gesture_mass < self.config.known_mass_floor:
            required_hold = max(required_hold, 0.45)
        execute = bool(
            confidence >= self.config.confidence_floor
            and stable_frames >= self.config.stable_frames_required
            and held_seconds >= required_hold
        )
        if confidence < self.config.confidence_floor:
            reason = "below confidence floor"
        elif stable_frames < self.config.stable_frames_required:
            reason = "insufficient consecutive stable frames"
        elif held_seconds < required_hold:
            reason = "gesture hold requirement not reached"
        else:
            reason = (
                "stable Dorsal geometry"
                if dorsal_fallback and known_gesture_mass < self.config.known_mass_floor
                else "stable directional geometry"
                if directional_fallback and known_gesture_mass < self.config.known_mass_floor
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
                name: float(ema_rows[-1, index])
                for index, name in enumerate(self.config.class_names)
            },
            held_seconds=held_seconds,
            probability_margin=probability_margin,
            known_gesture_mass=float(max(0.0, min(1.0, known_gesture_mass))),
        )
