from __future__ import annotations

from dataclasses import dataclass
import time


FOLLOW_SEQUENCE = ("open_palm",)
FOLLOW_MESSAGES = ("Hold Open Palm to enable object tracking",)


@dataclass(slots=True)
class FollowObjectUpdate:
    state: str
    step_index: int
    completed: bool
    reset: bool
    active: bool
    message: str
    expected_gesture: str | None
    hold_progress: float
    confidence: float
    source: str


@dataclass(slots=True)
class FollowObjectStateMachine:
    timeout_seconds: float = 20.0
    hold_seconds: float = 2.5
    session_timeout_seconds: float = 120.0
    _step_index: int = 0
    _active: bool = False
    _started_at: float | None = None
    _last_transition_at: float | None = None
    _candidate_started_at: float | None = None
    _candidate_gesture: str | None = None

    sequence = FOLLOW_SEQUENCE
    state_names = ("wait_open_palm",)

    @property
    def active(self) -> bool:
        return self._active

    @property
    def state(self) -> str:
        return self.state_names[self._step_index] if self._active else "inactive"

    @property
    def expected_gesture(self) -> str | None:
        return self.sequence[self._step_index] if self._active else None

    def start(self, now: float | None = None) -> FollowObjectUpdate:
        now = time.monotonic() if now is None else float(now)
        self._step_index = 0
        self._active = True
        self._started_at = now
        self._last_transition_at = now
        self._candidate_started_at = None
        self._candidate_gesture = None
        return self.snapshot(message=FOLLOW_MESSAGES[0])

    def stop(self) -> FollowObjectUpdate:
        self.reset()
        return self.snapshot(message="Follow Object stopped.")

    def reset(self) -> None:
        self._step_index = 0
        self._active = False
        self._started_at = None
        self._last_transition_at = None
        self._candidate_started_at = None
        self._candidate_gesture = None

    def snapshot(
        self,
        *,
        message: str | None = None,
        completed: bool = False,
        reset: bool = False,
        confidence: float = 0.0,
        source: str = "waiting",
        hold_progress: float = 0.0,
    ) -> FollowObjectUpdate:
        expected = self.expected_gesture
        default_message = (
            FOLLOW_MESSAGES[self._step_index]
            if self._active
            else "Press Begin Follow Object to start the dedicated sequence."
        )
        return FollowObjectUpdate(
            state="completed" if completed else self.state,
            step_index=len(self.sequence) if completed else self._step_index,
            completed=completed,
            reset=reset,
            active=self._active,
            message=message or default_message,
            expected_gesture=expected,
            hold_progress=float(max(0.0, min(1.0, hold_progress))),
            confidence=float(confidence),
            source=source,
        )

    def observe(
        self,
        gesture: str,
        stable: bool,
        confidence: float = 0.0,
        source: str = "runtime",
        now: float | None = None,
    ) -> FollowObjectUpdate:
        now = time.monotonic() if now is None else float(now)
        if not self._active:
            return self.snapshot()

        if self._started_at is not None and now - self._started_at > self.session_timeout_seconds:
            self.reset()
            return self.snapshot(
                message="Object tracking timed out. Start again with Open Palm.", reset=True
            )

        if (
            self._step_index > 0
            and self._last_transition_at is not None
            and now - self._last_transition_at > self.timeout_seconds
        ):
            self.start(now=now)
            return self.snapshot(
                message="Step timed out. Begin again with Open Palm.",
                reset=True,
                source="timeout",
            )

        expected = self.sequence[self._step_index]
        if not stable or gesture != expected:
            self._candidate_started_at = None
            self._candidate_gesture = None
            return self.snapshot(
                confidence=confidence,
                source=source,
                message=f"{FOLLOW_MESSAGES[self._step_index]} — hold {expected} steadily.",
            )

        if self._candidate_gesture != gesture or self._candidate_started_at is None:
            self._candidate_gesture = gesture
            self._candidate_started_at = now
        held = max(0.0, now - self._candidate_started_at)
        progress = 1.0 if self.hold_seconds <= 0 else held / self.hold_seconds
        if held < self.hold_seconds:
            return self.snapshot(
                confidence=confidence,
                source=source,
                hold_progress=progress,
                message=f"{FOLLOW_MESSAGES[self._step_index]} — keep holding {gesture}.",
            )

        self._step_index += 1
        self._last_transition_at = now
        self._candidate_started_at = None
        self._candidate_gesture = None
        if self._step_index == len(self.sequence):
            self.reset()
            return self.snapshot(
                completed=True,
                message="Object tracking enabled",
                confidence=confidence,
                source=source,
                hold_progress=1.0,
            )
        return self.snapshot(
            confidence=confidence,
            source=source,
            message=FOLLOW_MESSAGES[self._step_index],
        )
