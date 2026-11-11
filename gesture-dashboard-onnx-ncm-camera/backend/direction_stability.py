"""Bounded continuation of confirmed pointing, without delaying new commands."""
from __future__ import annotations

from dataclasses import dataclass

DIRECTIONS = frozenset(("left", "right", "up", "down"))


@dataclass(slots=True)
class DirectionStability:
    label: str | None = None
    confirmed_at: float = 0.0
    invalid_frames: int = 0

    def reset(self) -> None:
        self.label = None
        self.confirmed_at = 0.0
        self.invalid_frames = 0

    def boundary_candidate(self, label: str, qualified: bool, directional: dict) -> bool:
        alignment = float((directional.get("axis_alignment") or {}).get(self.label, -1.0))
        return bool(self.label and qualified and label in DIRECTIONS and label != self.label
                    and alignment >= .65 and directional.get("axis_dominance", 1.0) < .80)

    def update(self, *, label: str, qualified: bool, confirmed: bool,
               now: float, directional: dict, hard_rejection: bool,
               confidence: float, margin: float, known_mass: float,
               raw_label: str) -> str | None:
        """Return a display override; overrides never authorize a new action.

        Neutral evidence still reaches the caller's ordinary release controller.
        A grace result never refreshes the original confirmation timestamp.
        """
        previous = self.label
        alignment = float((directional.get("axis_alignment") or {}).get(previous, -1.0))
        boundary = self.boundary_candidate(label, qualified, directional)
        projected = bool(directional.get("continued_direction", False))
        # Keep the existing, independently validated 3-D single-index turn.
        if (projected and qualified and directional.get("world_index_only", False)
                and 0 <= now - self.confirmed_at <= 1.000001):
            self.invalid_frames = 0
            return None
        if qualified and label in DIRECTIONS and not boundary and not projected:
            self.invalid_frames = 0
            if confirmed:
                self.label, self.confirmed_at = label, now
            return None
        if hard_rejection or (qualified and label not in DIRECTIONS):
            self.reset()
            return None
        if previous is None:
            return None
        self.invalid_frames += 1
        elapsed = now - self.confirmed_at
        within_grace = 0 <= elapsed <= .200001 and self.invalid_frames < 2
        safe = bool(directional.get("continuation_safe", False)
                    and alignment >= .65 and confidence >= .65
                    and margin >= .05 and known_mass >= .60 and raw_label in DIRECTIONS)
        if within_grace and safe:
            return previous
        if not within_grace:
            self.reset()
        # A borderline neighbouring direction/foreshortened projection is not
        # fresh entry evidence while the previous direction is being released.
        return "no_gesture" if boundary or projected else None
