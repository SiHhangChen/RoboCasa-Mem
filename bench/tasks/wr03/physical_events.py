"""Per-control-step physical event observer for WR-03."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from bench.tasks.wr03.domain.contracts import CABINET_LEVELS, CAN_ID
from bench.tasks.wr03.domain.semantics import (
    SemanticState,
    latch_storage_hidden,
    record_retrieval_door_open,
    record_retrieval_pick,
    record_storage,
    record_storage_door_closed,
)


@dataclass(frozen=True)
class PhysicalSample:
    step_id: int
    can_in_lower: bool
    can_in_upper: bool
    lower_door_closed: bool
    upper_door_closed: bool
    lower_door_open: bool
    upper_door_open: bool

    def can_in_level(self, level: str) -> bool:
        return self.can_in_lower if level == "lower" else self.can_in_upper

    def door_closed(self, level: str) -> bool:
        return (
            self.lower_door_closed if level == "lower" else self.upper_door_closed
        )

    def door_open(self, level: str) -> bool:
        return self.lower_door_open if level == "lower" else self.upper_door_open

    @property
    def can_outside_both(self) -> bool:
        return not self.can_in_lower and not self.can_in_upper

    @property
    def both_doors_closed(self) -> bool:
        return self.lower_door_closed and self.upper_door_closed


@dataclass(frozen=True)
class ObserverUpdate:
    step_id: int
    semantic_events: tuple[str, ...]
    duplicate_step: bool
    diagnostics: Mapping[str, Any]


class WR03PhysicalObserver:
    """Infer the history-dependent WR-03 protocol from simulator state only."""

    STORAGE = "storage"
    CLOSE = "close"
    HIDDEN = "hidden"
    RETRIEVE = "retrieve"
    COMPLETE = "complete"

    def __init__(self, state: SemanticState, *, dwell_steps: int = 3) -> None:
        if int(dwell_steps) < 1:
            raise ValueError("WR-03 physical-event dwell must be positive")
        self.state = state
        self.dwell_steps = int(dwell_steps)
        self._previous: PhysicalSample | None = None
        self._counts: dict[str, int] = {}
        self._phase = self.STORAGE
        self._phase_history: list[dict[str, int | str]] = []
        self._stage_first_steps: dict[str, int] = {}

    def reset(self, sample: PhysicalSample) -> ObserverUpdate:
        self._previous = sample
        self._counts.clear()
        self._phase = self.STORAGE
        self._phase_history = [{"phase": self._phase, "step": int(sample.step_id)}]
        self._stage_first_steps.clear()
        return self._report(sample.step_id, (), duplicate_step=False)

    def observe(self, sample: PhysicalSample) -> ObserverUpdate:
        if self._previous is None:
            return self.reset(sample)
        if int(sample.step_id) <= int(self._previous.step_id):
            return self._report(sample.step_id, (), duplicate_step=True)

        self._synchronize_phase(sample.step_id)
        events: list[str] = []

        if self._phase == self.STORAGE:
            levels = [
                level for level in CABINET_LEVELS if sample.can_in_level(level)
            ]
            if len(levels) == 1 and self._stable(f"stored:{levels[0]}", True):
                level = levels[0]
                if record_storage(self.state, CAN_ID, level):
                    events.append(f"stored:{level}")
                    self._mark("stored", sample.step_id)
                    self._transition(self.CLOSE, sample.step_id)

        elif self._phase == self.CLOSE:
            stored = self.state.stored_level
            if stored is not None and not sample.can_in_level(stored):
                self._counts.clear()
            else:
                for level in CABINET_LEVELS:
                    if (
                        level not in self.state.closed_levels_after_storage
                        and self._stable(f"closed:{level}", sample.door_closed(level))
                        and record_storage_door_closed(self.state, level)
                    ):
                        events.append(f"closed:{level}")
                if (
                    stored is not None
                    and sample.can_in_level(stored)
                    and sample.both_doors_closed
                    and latch_storage_hidden(
                        self.state,
                        can_still_stored=True,
                        cabinets_closed=True,
                    )
                ):
                    events.append("memory_barrier")
                    self._mark("memory_barrier", sample.step_id)
                    self._transition(self.HIDDEN, sample.step_id)

        elif self._phase == self.HIDDEN:
            opened = [level for level in CABINET_LEVELS if sample.door_open(level)]
            if len(opened) == 1 and self._stable(f"opened:{opened[0]}", True):
                level = opened[0]
                record_retrieval_door_open(self.state, level)
                events.append(f"retrieval_door:{level}")
                self._mark("retrieval_door", sample.step_id)
                self._transition(self.RETRIEVE, sample.step_id)

        elif self._phase == self.RETRIEVE:
            if self._stable("extracted", sample.can_outside_both):
                if record_retrieval_pick(
                    self.state,
                    CAN_ID,
                    physically_extracted=True,
                ):
                    events.append("retrieved")
                    self._mark("retrieved", sample.step_id)
                    self._transition(self.COMPLETE, sample.step_id)

        self._previous = sample
        return self._report(sample.step_id, tuple(events), duplicate_step=False)

    def _synchronize_phase(self, step_id: int) -> None:
        if self.state.retrieved_item_id is not None:
            target = self.COMPLETE
        elif self.state.retrieval_level_opened is not None:
            target = self.RETRIEVE
        elif self.state.storage_hidden:
            target = self.HIDDEN
        elif self.state.stored_level is not None:
            target = self.CLOSE
        else:
            target = self.STORAGE
        if target != self._phase:
            self._transition(target, step_id)

    def _stable(self, name: str, condition: bool) -> bool:
        count = self._counts.get(name, 0) + 1 if condition else 0
        self._counts[name] = count
        return count >= self.dwell_steps

    def _transition(self, phase: str, step_id: int) -> None:
        if phase == self._phase:
            return
        self._phase = str(phase)
        self._counts.clear()
        self._phase_history.append({"phase": self._phase, "step": int(step_id)})

    def _mark(self, name: str, step_id: int) -> None:
        self._stage_first_steps.setdefault(str(name), int(step_id))

    def _report(
        self,
        step_id: int,
        events: tuple[str, ...],
        *,
        duplicate_step: bool,
    ) -> ObserverUpdate:
        return ObserverUpdate(
            step_id=int(step_id),
            semantic_events=events,
            duplicate_step=bool(duplicate_step),
            diagnostics={
                "phase": self._phase,
                "phase_history": list(self._phase_history),
                "dwell_steps": self.dwell_steps,
                "dwell_counts": dict(self._counts),
                "stage_first_steps": dict(self._stage_first_steps),
            },
        )


__all__ = ["ObserverUpdate", "PhysicalSample", "WR03PhysicalObserver"]
