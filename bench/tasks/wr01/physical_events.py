"""Per-control-step physical event reconstruction for WR-01.

The expert runner records semantic operations explicitly, but a recorded action
stream does not contain those callbacks.  This observer reconstructs the
history-dependent memory protocol from the simulator state so policy and
action-only replay use the same success contract as the expert run.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from bench.tasks.wr01.domain.contracts import CABINET_LEVELS, ITEM_IDS
from bench.tasks.wr01.domain.semantics import (
    SemanticState,
    evaluate_success,
    latch_storage_hidden,
    record_retrieval_door_open,
    record_retrieval_pick,
    record_storage,
)


@dataclass(frozen=True)
class PhysicalSample:
    step_id: int
    red_can_in_lower: bool
    red_can_in_upper: bool
    blue_can_in_lower: bool
    blue_can_in_upper: bool
    lower_door_closed: bool
    upper_door_closed: bool
    lower_door_open: bool
    upper_door_open: bool

    def item_in_level(self, item_id: str, level: str) -> bool:
        if item_id == "red_can":
            return self.red_can_in_lower if level == "lower" else self.red_can_in_upper
        if item_id == "blue_can":
            return self.blue_can_in_lower if level == "lower" else self.blue_can_in_upper
        raise ValueError(f"unknown WR-01 item: {item_id!r}")

    def item_levels(self, item_id: str) -> tuple[str, ...]:
        return tuple(level for level in CABINET_LEVELS if self.item_in_level(item_id, level))

    def door_closed(self, level: str) -> bool:
        return self.lower_door_closed if level == "lower" else self.upper_door_closed

    def door_open(self, level: str) -> bool:
        return self.lower_door_open if level == "lower" else self.upper_door_open

    @property
    def both_doors_closed(self) -> bool:
        return self.lower_door_closed and self.upper_door_closed

    def item_outside_all_cabinets(self, item_id: str) -> bool:
        return not self.item_levels(item_id)


@dataclass(frozen=True)
class ObserverUpdate:
    step_id: int
    semantic_events: tuple[str, ...]
    duplicate_step: bool
    diagnostics: Mapping[str, Any]


class WR01PhysicalObserver:
    """Single-pass WR-01 protocol observer driven only by physical state."""

    STORAGE = "storage"
    CLOSE = "close"
    HIDDEN = "hidden"
    RETRIEVE = "retrieve"
    COMPLETE = "complete"

    _STAGES = ("storage", "hidden", "retrieval_door", "retrieved", "success")

    def __init__(self, state: SemanticState, *, dwell_steps: int = 3) -> None:
        if int(dwell_steps) < 1:
            raise ValueError("WR-01 physical-event dwell must be positive")
        self.state = state
        self.dwell_steps = int(dwell_steps)
        self._previous: PhysicalSample | None = None
        self._counts: dict[str, int] = {}
        self._phase = self.STORAGE
        self._phase_history: list[dict[str, int | str]] = []
        self._stage_first_steps: dict[str, int] = {}

    @property
    def phase(self) -> str:
        return self._phase

    @property
    def stage_first_steps(self) -> dict[str, int]:
        return dict(self._stage_first_steps)

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
            for item_id in ITEM_IDS:
                if item_id in self.state.stored_levels_by_item:
                    continue
                levels = sample.item_levels(item_id)
                if len(levels) == 1 and self._stable(f"stored:{item_id}:{levels[0]}", True):
                    if record_storage(self.state, item_id, levels[0]):
                        events.append(f"stored:{item_id}:{levels[0]}")
                        self._mark("storage", sample.step_id)
                    self._counts.clear()
                    break
            if len(self.state.stored_levels_by_item) == len(ITEM_IDS):
                self._transition(self.CLOSE, sample.step_id)

        elif self._phase == self.CLOSE:
            for level in CABINET_LEVELS:
                if self._stable(f"closed:{level}", sample.door_closed(level)):
                    events.append(f"closed:{level}")
            expected = all(
                item in self.state.stored_levels_by_item for item in ITEM_IDS
            )
            if expected and sample.both_doors_closed and latch_storage_hidden(
                self.state,
                all_stored=all(
                    len(sample.item_levels(item)) == 1
                    and sample.item_in_level(
                        item, self.state.stored_levels_by_item[item]
                    )
                    for item in ITEM_IDS
                ),
                cabinets_closed=True,
            ):
                events.append("memory_barrier")
                self._mark("hidden", sample.step_id)
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
            target = self.state.target_item_id
            if self._stable(
                "retrieved",
                sample.item_outside_all_cabinets(target),
            ) and record_retrieval_pick(self.state, target):
                events.append("retrieved")
                self._mark("retrieved", sample.step_id)
                self._transition(self.COMPLETE, sample.step_id)

        elif self._phase == self.COMPLETE:
            if self._stable(
                "success",
                evaluate_success(
                    self.state,
                    _physical_state(sample, self.state),
                ).success,
            ):
                self._mark("success", sample.step_id)

        self._previous = sample
        return self._report(sample.step_id, tuple(events), duplicate_step=False)

    def _synchronize_phase(self, step_id: int) -> None:
        if self.state.retrieved_item_id is not None:
            target = self.COMPLETE
        elif self.state.retrieval_level_opened is not None:
            target = self.RETRIEVE
        elif self.state.storage_hidden:
            target = self.HIDDEN
        elif len(self.state.stored_levels_by_item) == len(ITEM_IDS):
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


def _physical_state(sample: PhysicalSample, state: SemanticState) -> Any:
    from bench.tasks.wr01.domain.semantics import PhysicalState

    distractor = next(item for item in ITEM_IDS if item != state.target_item_id)
    level = state.stored_levels_by_item.get(distractor)
    return PhysicalState(
        distractor_still_stored=bool(
            level is not None and sample.item_in_level(distractor, level)
        )
    )


__all__ = ["ObserverUpdate", "PhysicalSample", "WR01PhysicalObserver"]
