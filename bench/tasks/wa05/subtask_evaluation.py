"""Oracle subtask scheduler for isolated WA-05 low-level evaluation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from bench.evaluation.subtask import SubtaskTransition
from bench.tasks.wa05.evaluation import evaluate_initialized_runtime


SCHEDULE_ID = "wa05_oracle_v1"
_PICK_CONFIRM_STEPS = 3


@dataclass(frozen=True, slots=True)
class _SubtaskSpec:
    stage: str
    label: str


@dataclass(slots=True)
class WA05OracleSubtaskScheduler:
    """Switch prompts using privileged physical events, never policy inputs."""

    runtime: Any
    schedule_id: str = SCHEDULE_ID
    _index: int = field(init=False, default=0)
    _specs: tuple[_SubtaskSpec, ...] = field(init=False, default=())
    _prompts: list[str] = field(init=False, default_factory=list)
    _transitions: list[dict[str, Any]] = field(init=False, default_factory=list)
    _finished: bool = field(init=False, default=False)
    _failed: bool = field(init=False, default=False)
    _failure_reason: str | None = field(init=False, default=None)
    _initial_cup_position: np.ndarray | None = field(init=False, default=None)
    _pick_confirm_steps: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        if str(self.schedule_id) != SCHEDULE_ID:
            raise ValueError(
                f"unsupported WA-05 subtask schedule {self.schedule_id!r}; "
                f"expected {SCHEDULE_ID!r}"
            )
        self.schedule_id = str(self.schedule_id)
        self._specs = (
            _SubtaskSpec("pick_cup", "pick up the cup"),
            _SubtaskSpec("fill_cup", "fill the cup with water"),
            _SubtaskSpec("place_cup", "place the cup in the service area"),
            _SubtaskSpec("press_button", ""),
        )
        self._prompts = [spec.label for spec in self._specs]
        env = getattr(self.runtime, "env", None)
        position = getattr(env, "object_position", None)
        if callable(position):
            self._initial_cup_position = np.asarray(
                position("cup_A"), dtype=float
            ).reshape(3).copy()
        state = evaluate_initialized_runtime(self.runtime)
        if state.failure_code:
            self._failed = True
            self._failure_reason = str(state.failure_code)

    @property
    def current_subtask(self) -> str:
        return self._specs[self._index].stage

    @property
    def current_prompt(self) -> str:
        prompt = self._prompts[self._index]
        if not prompt:
            raise RuntimeError("WA-05 answer-button prompt has not been resolved")
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
            raise ValueError("WA-05 scheduler received a different runtime")
        step = _environment_step(runtime)
        if self._finished or self._failed:
            return SubtaskTransition(
                complete=self._finished,
                failed=self._failed,
                current_subtask=self.current_subtask,
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
                environment_step=step,
                failure_reason=self._failure_reason,
            )
            self._transitions.append(event.to_dict())
            return event

        if not self._current_subtask_completed(state):
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
        if self.current_subtask == "press_button":
            self._prompts[self._index] = _button_prompt(state)
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

    def _current_subtask_completed(self, state: Any) -> bool:
        if self.current_subtask == "pick_cup":
            held = _cup_is_held_and_lifted(
                self.runtime,
                initial_position=self._initial_cup_position,
            )
            self._pick_confirm_steps = self._pick_confirm_steps + 1 if held else 0
            return self._pick_confirm_steps >= _PICK_CONFIRM_STEPS
        if self.current_subtask == "fill_cup":
            return _stage_completed(state, "cup_filled")
        if self.current_subtask == "place_cup":
            return _stage_completed(state, "cup_staged")
        if self.current_subtask == "press_button":
            return _stage_completed(state, "success")
        raise RuntimeError(f"unknown WA-05 subtask {self.current_subtask!r}")


def build_subtask_scheduler(runtime: Any, schedule_id: str) -> WA05OracleSubtaskScheduler:
    return WA05OracleSubtaskScheduler(runtime=runtime, schedule_id=str(schedule_id))


def _cup_is_held_and_lifted(
    runtime: Any,
    *,
    initial_position: np.ndarray | None,
) -> bool:
    env = getattr(runtime, "env", None)
    if env is None or initial_position is None:
        return False
    try:
        from bench.execution.manipulation.object_interactions import (
            get_object_hold_status,
        )

        hold = get_object_hold_status(env=env, object_name="cup_A")
        if not bool(hold.get("held", False)):
            return False
        current = np.asarray(env.object_position("cup_A"), dtype=float).reshape(3)
        lift = float(current[2] - initial_position[2])
        lateral = float(np.linalg.norm(current[:2] - initial_position[:2]))
        upright = getattr(env, "cup_is_upright", None)
        return bool(
            (not callable(upright) or upright("cup_A"))
            and (lift >= 0.02 or lateral >= 0.04)
        )
    except Exception:
        return False


def _stage_completed(state: Any, stage: str) -> bool:
    try:
        return bool(state.stages[stage].qualified)
    except KeyError as exc:
        raise RuntimeError(f"WA-05 evaluator has no stage {stage!r}") from exc


def _button_prompt(state: Any) -> str:
    temperature = str(
        dict(getattr(state, "details", {}) or {}).get("filled_temperature", "")
    ).strip().lower()
    if temperature == "hot":
        return "press the red button for hot water"
    if temperature == "cold":
        return "press the blue button for cold water"
    raise RuntimeError("WA-05 evaluator did not resolve the filled water temperature")


def _environment_step(runtime: Any) -> int | None:
    env = getattr(runtime, "env", None)
    value = getattr(env, "timestep", None)
    return None if value is None else int(value)


__all__ = ["SCHEDULE_ID", "WA05OracleSubtaskScheduler", "build_subtask_scheduler"]
