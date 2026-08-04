"""Deterministic continuous-fill state for the TM02 sink-side pot."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PotFillState:
    """Task-owned fill state independent of simulator rendering."""

    active_steps: int = 0
    progress: float = 0.0
    filled: bool = False


def advance_pot_fill(
    state: PotFillState,
    *,
    water_reaches_pot: bool,
    required_steps: int,
    filled_threshold: float,
) -> PotFillState:
    """Advance one control step and preserve accumulated water while paused."""

    steps_required = int(required_steps)
    threshold = float(filled_threshold)
    if steps_required <= 0:
        raise ValueError("TM02 pot_fill_required_steps must be positive")
    if not 0.0 < threshold <= 1.0:
        raise ValueError("TM02 pot_filled_threshold must be within (0, 1]")

    active_steps = max(0, int(state.active_steps))
    if bool(water_reaches_pot) and active_steps < steps_required:
        active_steps += 1
    progress = min(1.0, float(active_steps) / float(steps_required))
    return PotFillState(
        active_steps=active_steps,
        progress=progress,
        filled=bool(state.filled or progress >= threshold),
    )


__all__ = ["PotFillState", "advance_pot_fill"]
