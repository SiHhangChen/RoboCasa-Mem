"""Authoritative ordered TS01 evaluation state machine.

The simulator owns containment debouncing and exposes the number of sugar
cubes that are physically confirmed inside the saucepan.  This module owns the
task protocol on top of that signal: seed-dependent sugar prerequisites,
rising-edge qualification, terminal over-add, completion-button ordering, and
the padded three-sugar view consumed by aggregate reports.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from bench.tasks.ts01.domain.contracts import VALID_TARGET_SUGAR_COUNTS


SUGAR_STAGE_NAMES = ("sugar_1", "sugar_2", "sugar_3")
STAGE_ORDER = (*SUGAR_STAGE_NAMES, "button", "success")
TASK_EVALUATOR_METADATA_KEY = "task_evaluator"


def _target_count(value: Any) -> int:
    try:
        target = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"target_sugar_count must be an integer, got {value!r}") from exc
    if target not in VALID_TARGET_SUGAR_COUNTS:
        raise ValueError(
            "target_sugar_count must be one of "
            f"{VALID_TARGET_SUGAR_COUNTS}, got {value!r}"
        )
    return target


@dataclass(frozen=True, slots=True)
class TS01EvaluationSnapshot:
    """One debounced physical observation presented to the task evaluator."""

    step_id: int
    target_sugar_count: int
    confirmed_sugar_count: int
    objects_released: bool
    completion_button_pressed: bool
    details: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        step = int(self.step_id)
        count = int(self.confirmed_sugar_count)
        if step < 0:
            raise ValueError("step_id must be non-negative")
        if count < 0:
            raise ValueError("confirmed_sugar_count must be non-negative")
        object.__setattr__(self, "step_id", step)
        object.__setattr__(
            self,
            "target_sugar_count",
            _target_count(self.target_sugar_count),
        )
        object.__setattr__(self, "confirmed_sugar_count", count)
        object.__setattr__(self, "objects_released", bool(self.objects_released))
        object.__setattr__(
            self,
            "completion_button_pressed",
            bool(self.completion_button_pressed),
        )
        object.__setattr__(self, "details", dict(self.details or {}))

    @property
    def sugar_goal_ready(self) -> bool:
        return bool(
            self.confirmed_sugar_count == self.target_sugar_count
            and self.objects_released
        )

    @property
    def over_added(self) -> bool:
        return self.confirmed_sugar_count > self.target_sugar_count

    def raw_stage_values(self) -> dict[str, bool]:
        values = {
            name: bool(
                index <= self.target_sugar_count
                and self.confirmed_sugar_count >= index
            )
            for index, name in enumerate(SUGAR_STAGE_NAMES, start=1)
        }
        values["button"] = bool(self.completion_button_pressed)
        values["success"] = bool(
            self.completion_button_pressed and self.sugar_goal_ready
        )
        return values


@dataclass(frozen=True, slots=True)
class StageEvalResult:
    """Observed versus prerequisite-qualified state for one TS01 stage."""

    applicable: bool
    observed: bool
    qualified: bool
    first_observed_step: int | None
    qualified_step: int | None
    invalidated: bool
    reason: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "applicable": bool(self.applicable),
            "observed": bool(self.observed),
            "qualified": bool(self.qualified),
            "first_observed_step": self.first_observed_step,
            "qualified_step": self.qualified_step,
            "invalidated": bool(self.invalidated),
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True)
class TS01EvalResult:
    """Immutable verdict returned by the TS01 evaluator."""

    target_sugar_count: int
    success: bool
    terminal: bool
    failure_code: str | None
    stages: Mapping[str, StageEvalResult]
    details: Mapping[str, Any] = field(default_factory=dict)

    def to_diagnostics(self) -> dict[str, Any]:
        payload = dict(self.details or {})
        stages: dict[str, dict[str, Any]] = {}
        required_stage = f"sugar_{self.target_sugar_count}"
        required_qualified = bool(self.stages[required_stage].qualified)
        over_added = self.failure_code == "over_add"

        for name in STAGE_ORDER:
            stage = self.stages[name]
            values = stage.to_dict()
            if name in SUGAR_STAGE_NAMES and not stage.applicable:
                if over_added:
                    effective_success = False
                    completion_mode = "blocked_by_over_add"
                elif required_qualified:
                    effective_success = True
                    completion_mode = "not_required"
                else:
                    effective_success = False
                    completion_mode = "waiting_for_required_sugar_goal"
                derived_from: str | None = required_stage
            else:
                effective_success = bool(stage.qualified)
                completion_mode = "physical" if stage.qualified else None
                derived_from = None
            values.update(
                {
                    "effective_success": effective_success,
                    "completion_mode": completion_mode,
                    "derived_from": derived_from,
                }
            )
            stages[name] = values

        payload.update(
            {
                "success": bool(self.success),
                "terminal": bool(self.terminal),
                "failure_code": self.failure_code,
                "target_sugar_count": int(self.target_sugar_count),
                "stages": stages,
                "stage_checks": {
                    name: bool(values["qualified"])
                    for name, values in stages.items()
                },
                "effective_stage_checks": {
                    name: bool(values["effective_success"])
                    for name, values in stages.items()
                },
                "stage_first_observed_steps": {
                    name: values["first_observed_step"]
                    for name, values in stages.items()
                },
                "stage_qualified_steps": {
                    name: values["qualified_step"]
                    for name, values in stages.items()
                },
            }
        )
        return payload


@dataclass
class _MutableStage:
    observed: bool = False
    qualified: bool = False
    first_observed_step: int | None = None
    qualified_step: int | None = None


class TS01Evaluator:
    """Strict, history-aware evaluator for one seed-resolved TS01 episode."""

    def __init__(self, *, target_sugar_count: int) -> None:
        self.target_sugar_count = _target_count(target_sugar_count)
        self._stages = {name: _MutableStage() for name in STAGE_ORDER}
        self._rejected: dict[str, list[dict[str, Any]]] = {
            name: [] for name in STAGE_ORDER
        }
        self._previous_raw = {name: False for name in STAGE_ORDER}
        self._last_step_id: int | None = None
        self._terminal_failure_code: str | None = None
        self._cached_result: TS01EvalResult | None = None

    @property
    def stage_prerequisites(self) -> dict[str, str | None]:
        prerequisites: dict[str, str | None] = {
            "sugar_1": None,
            "sugar_2": "sugar_1",
            "sugar_3": "sugar_2",
            "button": f"sugar_{self.target_sugar_count}",
            "success": "button",
        }
        return prerequisites

    def reset(self, snapshot: TS01EvaluationSnapshot) -> TS01EvalResult:
        self._validate_configuration(snapshot)
        self._stages = {name: _MutableStage() for name in STAGE_ORDER}
        self._rejected = {name: [] for name in STAGE_ORDER}
        self._terminal_failure_code = "over_add" if snapshot.over_added else None
        raw = snapshot.raw_stage_values()
        if self._terminal_failure_code is None:
            for name, value in raw.items():
                if value:
                    self._mark_observed(name, snapshot.step_id)
                    self._reject(name, snapshot.step_id, "present_at_reset")
        self._previous_raw = raw
        self._last_step_id = snapshot.step_id
        self._cached_result = self._build_result(snapshot)
        return self._cached_result

    def observe(self, snapshot: TS01EvaluationSnapshot) -> TS01EvalResult:
        self._validate_configuration(snapshot)
        if self._last_step_id is None:
            return self.reset(snapshot)
        if snapshot.step_id == self._last_step_id:
            if self._cached_result is None:
                raise RuntimeError("TS01 evaluator has no cached result")
            return self._cached_result
        if snapshot.step_id < self._last_step_id:
            raise ValueError(
                "TS01 evaluation step moved backwards: "
                f"{snapshot.step_id} < {self._last_step_id}"
            )
        if self._cached_result is not None and self._cached_result.terminal:
            return self._cached_result

        raw = snapshot.raw_stage_values()

        # Over-add takes precedence over all observations in the same step.
        # This prevents a count jump from being credited as an earlier stage.
        if snapshot.over_added:
            self._terminal_failure_code = "over_add"
            self._previous_raw = raw
            self._last_step_id = snapshot.step_id
            self._cached_result = self._build_result(snapshot)
            return self._cached_result

        qualified_before = {
            name: self._stages[name].qualified for name in STAGE_ORDER
        }

        for name, value in raw.items():
            if value:
                self._mark_observed(name, snapshot.step_id)

        for index, name in enumerate(SUGAR_STAGE_NAMES, start=1):
            if index > self.target_sugar_count:
                continue
            if not raw[name] or self._previous_raw[name] or self._stages[name].qualified:
                continue
            prerequisite = self.stage_prerequisites[name]
            if prerequisite is None or qualified_before[prerequisite]:
                self._qualify(name, snapshot.step_id)
            else:
                self._reject(
                    name,
                    snapshot.step_id,
                    f"prerequisite_not_qualified:{prerequisite}",
                )

        if (
            raw["button"]
            and not self._previous_raw["button"]
            and not self._stages["button"].qualified
        ):
            prerequisite = self.stage_prerequisites["button"]
            if prerequisite is None or not qualified_before[prerequisite]:
                self._reject(
                    "button",
                    snapshot.step_id,
                    f"prerequisite_not_qualified:{prerequisite}",
                )
            elif not snapshot.sugar_goal_ready:
                self._reject("button", snapshot.step_id, "sugar_goal_not_ready")
            else:
                self._qualify("button", snapshot.step_id)
                self._mark_observed("success", snapshot.step_id)
                self._qualify("success", snapshot.step_id)

        self._previous_raw = raw
        self._last_step_id = snapshot.step_id
        self._cached_result = self._build_result(snapshot)
        return self._cached_result

    def result(self) -> TS01EvalResult:
        if self._cached_result is None:
            raise RuntimeError("TS01 evaluator has not been reset")
        return self._cached_result

    def _validate_configuration(self, snapshot: TS01EvaluationSnapshot) -> None:
        if snapshot.target_sugar_count != self.target_sugar_count:
            raise ValueError(
                "TS01 evaluation snapshot target_sugar_count changed within "
                f"an episode: {snapshot.target_sugar_count} != "
                f"{self.target_sugar_count}"
            )

    def _stage_applicable(self, name: str) -> bool:
        if name not in SUGAR_STAGE_NAMES:
            return True
        return SUGAR_STAGE_NAMES.index(name) < self.target_sugar_count

    def _mark_observed(self, name: str, step_id: int) -> None:
        stage = self._stages[name]
        if not stage.observed:
            stage.observed = True
            stage.first_observed_step = int(step_id)

    def _qualify(self, name: str, step_id: int) -> None:
        stage = self._stages[name]
        if stage.qualified:
            return
        stage.qualified = True
        stage.qualified_step = int(step_id)

    def _reject(self, name: str, step_id: int, reason: str) -> None:
        entry = {"step": int(step_id), "reason": str(reason)}
        if entry not in self._rejected[name]:
            self._rejected[name].append(entry)

    def _stage_result(self, name: str) -> StageEvalResult:
        applicable = self._stage_applicable(name)
        stage = self._stages[name]
        rejected = self._rejected[name]
        if not applicable:
            invalidated = False
            reason = "not_applicable_for_target_count"
        elif stage.qualified:
            invalidated = False
            reason = None
        elif rejected:
            invalidated = True
            reason = str(rejected[-1]["reason"])
        else:
            prerequisite = self.stage_prerequisites[name]
            if prerequisite is not None and not self._stages[prerequisite].qualified:
                invalidated = False
                reason = f"waiting_for_prerequisite:{prerequisite}"
            else:
                invalidated = False
                reason = "not_observed"
        return StageEvalResult(
            applicable=applicable,
            observed=stage.observed,
            qualified=stage.qualified,
            first_observed_step=stage.first_observed_step,
            qualified_step=stage.qualified_step,
            invalidated=invalidated,
            reason=reason,
        )

    def _build_result(self, snapshot: TS01EvaluationSnapshot) -> TS01EvalResult:
        stages = {name: self._stage_result(name) for name in STAGE_ORDER}
        details = dict(snapshot.details)
        evaluation = dict(details.get("evaluation") or {})
        evaluation.update(
            {
                "schema_version": 1,
                "step_id": int(snapshot.step_id),
                "stage_order": [
                    *SUGAR_STAGE_NAMES[: self.target_sugar_count],
                    "button",
                    "success",
                ],
                "reporting_stage_order": list(STAGE_ORDER),
                "stage_prerequisites": self.stage_prerequisites,
                "qualification_policy": "rising_edge_after_prerequisite",
                "raw_stage_checks": snapshot.raw_stage_values(),
                "rejected_stage_observations": {
                    name: list(values)
                    for name, values in self._rejected.items()
                    if values
                },
            }
        )
        details.update(
            {
                "evaluation": evaluation,
                "confirmed_sugar_count": int(snapshot.confirmed_sugar_count),
                "objects_released": bool(snapshot.objects_released),
                "completion_button_pressed": bool(
                    snapshot.completion_button_pressed
                ),
                "sugar_goal_ready": bool(snapshot.sugar_goal_ready),
            }
        )
        if self._terminal_failure_code == "over_add":
            details["over_add"] = {
                "target_sugar_count": int(self.target_sugar_count),
                "confirmed_sugar_count": int(snapshot.confirmed_sugar_count),
                "step": int(snapshot.step_id),
            }
        success = bool(stages["success"].qualified)
        return TS01EvalResult(
            target_sugar_count=self.target_sugar_count,
            success=success,
            terminal=bool(success or self._terminal_failure_code is not None),
            failure_code=self._terminal_failure_code,
            stages=stages,
            details=details,
        )


def evaluate_initialized_runtime(runtime: Any) -> TS01EvalResult:
    """Evaluate one initialized task runtime through its task-local evaluator."""

    initialized = getattr(runtime, "initialized", None) or runtime
    metadata = getattr(initialized, "metadata", {}) or {}
    evaluator = metadata.get(TASK_EVALUATOR_METADATA_KEY)
    env = (
        getattr(initialized, "env", None)
        or getattr(runtime, "env", None)
        or initialized
    )
    if evaluator is None:
        evaluator = getattr(env, "_ts01_evaluator", None)
    if not isinstance(evaluator, TS01Evaluator):
        raise RuntimeError("TS01 runtime has no initialized task evaluator")
    snapshot_builder = getattr(env, "evaluation_snapshot", None)
    if not callable(snapshot_builder):
        raise AttributeError("TS01 environment has no evaluation_snapshot()")
    return evaluator.observe(snapshot_builder())


def success_for_runtime(
    runtime: Any,
    info: Mapping[str, Any] | None = None,
) -> bool:
    del info
    return bool(evaluate_initialized_runtime(runtime).success)


def diagnostics_for_runtime(runtime: Any) -> Mapping[str, Any]:
    return evaluate_initialized_runtime(runtime).to_diagnostics()


__all__ = [
    "SUGAR_STAGE_NAMES",
    "STAGE_ORDER",
    "TASK_EVALUATOR_METADATA_KEY",
    "StageEvalResult",
    "TS01EvalResult",
    "TS01EvaluationSnapshot",
    "TS01Evaluator",
    "diagnostics_for_runtime",
    "evaluate_initialized_runtime",
    "success_for_runtime",
]
