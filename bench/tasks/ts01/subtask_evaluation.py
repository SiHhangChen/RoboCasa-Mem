"""Oracle subtask scheduler for isolated TS01 low-level evaluation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from bench.evaluation.subtask import SubtaskTransition
from bench.tasks.ts01.evaluation import evaluate_initialized_runtime


SCHEDULE_ID = "ts01_oracle_v1"


@dataclass(frozen=True, slots=True)
class _SubtaskSpec:
    stage: str
    label: str


@dataclass(slots=True)
class TS01OracleSubtaskScheduler:
    """Switch from sugar placement to button pressing using simulator state."""

    runtime: Any
    schedule_id: str = SCHEDULE_ID
    _index: int = field(init=False, default=0)
    _specs: tuple[_SubtaskSpec, ...] = field(init=False, default=())
    _transitions: list[dict[str, Any]] = field(init=False, default_factory=list)
    _finished: bool = field(init=False, default=False)
    _failed: bool = field(init=False, default=False)
    _failure_reason: str | None = field(init=False, default=None)

    def __post_init__(self) -> None:
        if str(self.schedule_id) != SCHEDULE_ID:
            raise ValueError(
                f"unsupported TS01 subtask schedule {self.schedule_id!r}; "
                f"expected {SCHEDULE_ID!r}"
            )
        self.schedule_id = str(self.schedule_id)
        self._specs = (
            _SubtaskSpec("add_sugar", "add a sugar cube to the hot chocolate"),
            _SubtaskSpec("press_button", "press the completion button"),
        )
        state = evaluate_initialized_runtime(self.runtime)
        if state.failure_code:
            self._failed = True
            self._failure_reason = str(state.failure_code)

    @property
    def current_subtask(self) -> str:
        return self._specs[self._index].stage

    @property
    def current_prompt(self) -> str:
        return self._specs[self._index].label

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
            raise ValueError("TS01 scheduler received a different runtime")
        step = _environment_step(runtime)
        if self._finished or self._failed:
            return SubtaskTransition(
                complete=self._finished,
                failed=self._failed,
                current_subtask=self.current_subtask,
                prompt=self.current_prompt,
                environment_step=step,
                failure_reason=self._failure_reason,
            )

        state = evaluate_initialized_runtime(runtime)
        if state.failure_code:
            self._failed = True
            self._failure_reason = str(state.failure_code)
            event = SubtaskTransition(
                failed=True,
                current_subtask=self.current_subtask,
                prompt=self.current_prompt,
                environment_step=step,
                failure_reason=self._failure_reason,
            )
            self._transitions.append(event.to_dict())
            return event

        if not _subtask_completed(state, self.current_subtask):
            return SubtaskTransition(
                current_subtask=self.current_subtask,
                prompt=self.current_prompt,
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


def build_subtask_scheduler(runtime: Any, schedule_id: str) -> TS01OracleSubtaskScheduler:
    return TS01OracleSubtaskScheduler(runtime=runtime, schedule_id=str(schedule_id))


def _subtask_completed(state: Any, subtask: str) -> bool:
    if subtask == "add_sugar":
        return bool(dict(getattr(state, "details", {}) or {}).get("sugar_goal_ready"))
    if subtask == "press_button":
        try:
            return bool(state.stages["success"].qualified)
        except KeyError as exc:
            raise RuntimeError("TS01 evaluator has no success stage") from exc
    raise RuntimeError(f"unknown TS01 subtask {subtask!r}")


def _environment_step(runtime: Any) -> int | None:
    env = getattr(runtime, "env", None)
    value = getattr(env, "timestep", None)
    return None if value is None else int(value)


__all__ = ["SCHEDULE_ID", "TS01OracleSubtaskScheduler", "build_subtask_scheduler"]
