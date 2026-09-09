"""Oracle subtask scheduler for isolated WA-01 low-level evaluation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from bench.evaluation.subtask import SubtaskTransition
from bench.tasks.wa01.evaluation import evaluate_initialized_runtime


SCHEDULE_ID = "wa01_oracle_v1"


@dataclass(frozen=True, slots=True)
class _SubtaskSpec:
    stage: str
    label: str


@dataclass(slots=True)
class WA01OracleSubtaskScheduler:
    """Switch policy prompts from simulator-verified WA-01 stage events.

    The scheduler sees the initialized task runtime, including its private
    evaluator state.  That state is used only for scheduling and reporting; it
    is never included in the policy observation.
    """

    runtime: Any
    schedule_id: str = SCHEDULE_ID
    _index: int = field(init=False, default=0)
    _specs: tuple[_SubtaskSpec, ...] = field(init=False, default=())
    _transitions: list[dict[str, Any]] = field(init=False, default_factory=list)
    _finished: bool = field(init=False, default=False)
    _failed: bool = field(init=False, default=False)
    _failure_reason: str | None = field(init=False, default=None)
    _prompts: tuple[str, ...] = field(init=False, default=())

    def __post_init__(self) -> None:
        if str(self.schedule_id) != SCHEDULE_ID:
            raise ValueError(
                f"unsupported WA-01 subtask schedule {self.schedule_id!r}; "
                f"expected {SCHEDULE_ID!r}"
            )
        self.schedule_id = str(self.schedule_id)
        self._specs = (
            _SubtaskSpec("sugar", "add one sugar cube to a coffee cup"),
            _SubtaskSpec("one_ready", "carry one coffee cup to the dining counter"),
            _SubtaskSpec(
                "ready",
                "carry the remaining coffee cup to the dining counter",
            ),
            # The final prompt is resolved from the privileged cup-to-slot
            # assignment once both cups are ready.
            _SubtaskSpec("success", ""),
        )
        self._prompts = tuple(spec.label for spec in self._specs)
        state = evaluate_initialized_runtime(self.runtime)
        if state.failure_code:
            self._failed = True
            self._failure_reason = str(state.failure_code)
        # The sugared coffee is unknown at reset for unmarked episodes.  The
        # final left/right prompt is resolved at the ready -> selection handoff.

    @property
    def current_subtask(self) -> str:
        return self._specs[self._index].stage

    @property
    def current_prompt(self) -> str:
        prompt = self._prompts[self._index]
        if not prompt:
            raise RuntimeError("WA-01 selection prompt has not been resolved")
        return prompt

    @property
    def finished(self) -> bool:
        return self._finished

    @property
    def failed(self) -> bool:
        return self._failed

    def after_step(
        self,
        runtime: Any,
        info: Mapping[str, Any],
    ) -> SubtaskTransition:
        del info
        if runtime is not self.runtime:
            raise ValueError("WA-01 scheduler received a different runtime")
        if self._finished or self._failed:
            return SubtaskTransition(
                complete=self._finished,
                failed=self._failed,
                current_subtask=self.current_subtask,
                environment_step=_environment_step(runtime),
                failure_reason=self._failure_reason,
            )

        state = evaluate_initialized_runtime(runtime)
        step = _environment_step(runtime)
        if state.failure_code:
            self._failed = True
            self._failure_reason = str(state.failure_code)
            event = SubtaskTransition(
                failed=True,
                current_subtask=self.current_subtask,
                environment_step=step,
                failure_reason=self._failure_reason,
            )
            self._transitions.append(event.to_dict())
            return event

        if not _stage_completed(state, self._specs[self._index].stage):
            return SubtaskTransition(
                current_subtask=self.current_subtask,
                environment_step=step,
            )

        previous = self.current_subtask
        if self._index == len(self._specs) - 1:
            self._finished = True
            event = SubtaskTransition(
                complete=True,
                previous_subtask=previous,
                current_subtask=previous,
                prompt=self.current_prompt,
                environment_step=step,
            )
            self._transitions.append(event.to_dict())
            return event

        self._index += 1
        if self.current_subtask == "success":
            # Re-resolve at handoff time so the target is based on the actual
            # detected sugared cup, while the slot mapping remains privileged.
            self._prompts = (*self._prompts[:-1], self._selection_prompt(state))
        event = SubtaskTransition(
            changed=True,
            previous_subtask=previous,
            current_subtask=self.current_subtask,
            prompt=self.current_prompt,
            environment_step=step,
        )
        self._transitions.append(event.to_dict())
        return event

    def to_metadata(self) -> Mapping[str, Any]:
        return {
            "schedule_id": self.schedule_id,
            "subtasks": [spec.stage for spec in self._specs],
            "current_subtask": self.current_subtask,
            "current_prompt": self.current_prompt,
            "finished": bool(self._finished),
            "failed": bool(self._failed),
            "failure_reason": self._failure_reason,
            "transitions": list(self._transitions),
        }

    def _selection_prompt(self, state: Any) -> str:
        details = dict(getattr(state, "details", {}) or {})
        sugar = details.get("sugar")
        placement = details.get("dining_placement")
        if not isinstance(sugar, Mapping) or not isinstance(placement, Mapping):
            raise RuntimeError("WA-01 evaluator omitted privileged placement details")
        coffee_id = sugar.get("sugared_coffee_id")
        slot_by_coffee = placement.get("planned_slot_by_coffee")
        if not isinstance(slot_by_coffee, Mapping):
            raise RuntimeError("WA-01 evaluator omitted planned slot assignment")
        slot = str(slot_by_coffee.get(coffee_id, "")).strip().lower()
        if slot not in {"left", "right"}:
            raise RuntimeError(
                f"cannot resolve WA-01 selection slot for coffee {coffee_id!r}: {slot!r}"
            )
        return f"pick up the coffee cup in the {slot} dining placement zone"


def build_subtask_scheduler(runtime: Any, schedule_id: str) -> WA01OracleSubtaskScheduler:
    return WA01OracleSubtaskScheduler(runtime=runtime, schedule_id=str(schedule_id))


def _stage_completed(state: Any, stage: str) -> bool:
    try:
        return bool(state.stages[stage].qualified)
    except KeyError as exc:
        raise RuntimeError(f"WA-01 evaluator has no stage {stage!r}") from exc


def _environment_step(runtime: Any) -> int | None:
    env = getattr(runtime, "env", None)
    value = getattr(env, "timestep", None)
    return None if value is None else int(value)


__all__ = ["SCHEDULE_ID", "WA01OracleSubtaskScheduler", "build_subtask_scheduler"]
