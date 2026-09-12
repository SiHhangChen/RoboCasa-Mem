"""Authoritative trajectory-aware TS02 evaluation state machine.

The physical observer owns drawer and button event detection. This module
owns the benchmark protocol on top of those debounced facts and projects the
result onto TS02's five public report stages.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from bench.evaluation.report_profiles import TS02_REPORT_PROFILE


STAGE_ORDER = TS02_REPORT_PROFILE.stage_ids
STAGE_PREREQUISITES: dict[str, str | None] = {
    "one_drawer_checked": None,
    "target_found": "one_drawer_checked",
    "search_stopped_after_target": "target_found",
    "button_pressed": "search_stopped_after_target",
    "correct_press_count": "button_pressed",
}
REPORT_STAGE_ORDER = STAGE_ORDER
REPORT_STAGE_PROFILE_ID = TS02_REPORT_PROFILE.profile_id
TASK_EVALUATOR_METADATA_KEY = "task_evaluator"


def _step(value: Any, *, field_name: str) -> int:
    try:
        step = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be an integer, got {value!r}") from exc
    if step < 0:
        raise ValueError(f"{field_name} must be non-negative")
    return step


@dataclass(frozen=True, slots=True)
class TS02EvaluationSnapshot:
    """One task-owned trajectory snapshot presented to the evaluator."""

    step_id: int
    fixture_names: tuple[str, ...]
    target_fixture: str
    completed_fixture_history: tuple[str, ...] = ()
    completed_fixture_steps: Mapping[str, int] = field(default_factory=dict)
    target_found: bool = False
    search_after_target_detected: bool = False
    valid_button_press_steps: tuple[int, ...] = ()
    actual_button_press_count: int = 0
    required_button_press_count: int = 0
    domain_success: bool = False
    failure_reason: str | None = None
    details: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        step_id = _step(self.step_id, field_name="step_id")
        fixture_names = tuple(str(name) for name in self.fixture_names)
        target_fixture = str(self.target_fixture)
        if len(fixture_names) < 2 or len(set(fixture_names)) != len(fixture_names):
            raise ValueError("fixture_names must contain at least two unique names")
        if target_fixture not in fixture_names:
            raise ValueError("target_fixture must name one of fixture_names")

        history = tuple(str(name) for name in self.completed_fixture_history)
        unknown_history = sorted(set(history) - set(fixture_names))
        if unknown_history:
            raise ValueError(
                "completed_fixture_history contains unknown drawers: "
                + ", ".join(unknown_history)
            )
        completed_steps = {
            str(name): _step(value, field_name=f"completed_fixture_steps[{name!r}]")
            for name, value in self.completed_fixture_steps.items()
        }
        unknown_steps = sorted(set(completed_steps) - set(fixture_names))
        if unknown_steps:
            raise ValueError(
                "completed_fixture_steps contains unknown drawers: "
                + ", ".join(unknown_steps)
            )
        missing_steps = sorted(set(history) - set(completed_steps))
        if missing_steps:
            raise ValueError(
                "completed_fixture_steps is missing completed drawers: "
                + ", ".join(missing_steps)
            )
        if any(value > step_id for value in completed_steps.values()):
            raise ValueError("completed fixture step cannot exceed step_id")

        button_steps = tuple(
            _step(value, field_name="valid_button_press_steps entry")
            for value in self.valid_button_press_steps
        )
        if any(value > step_id for value in button_steps):
            raise ValueError("valid button press step cannot exceed step_id")
        actual_count = _step(
            self.actual_button_press_count,
            field_name="actual_button_press_count",
        )
        required_count = _step(
            self.required_button_press_count,
            field_name="required_button_press_count",
        )
        expected_required_count = len(history) if bool(self.target_found) else 0
        if required_count != expected_required_count:
            raise ValueError(
                "TS02 required_button_press_count must be 0 before target_found "
                "and equal the completed search length afterwards"
            )
        if len(button_steps) > actual_count:
            raise ValueError(
                "valid_button_press_steps cannot exceed actual_button_press_count"
            )

        object.__setattr__(self, "step_id", step_id)
        object.__setattr__(self, "fixture_names", fixture_names)
        object.__setattr__(self, "target_fixture", target_fixture)
        object.__setattr__(self, "completed_fixture_history", history)
        object.__setattr__(self, "completed_fixture_steps", completed_steps)
        object.__setattr__(self, "target_found", bool(self.target_found))
        object.__setattr__(
            self,
            "search_after_target_detected",
            bool(self.search_after_target_detected),
        )
        object.__setattr__(self, "valid_button_press_steps", button_steps)
        object.__setattr__(self, "actual_button_press_count", actual_count)
        object.__setattr__(self, "required_button_press_count", required_count)
        object.__setattr__(self, "domain_success", bool(self.domain_success))
        object.__setattr__(
            self,
            "failure_reason",
            None if self.failure_reason is None else str(self.failure_reason),
        )
        object.__setattr__(self, "details", dict(self.details or {}))

    @property
    def target_completion_step(self) -> int | None:
        return self.completed_fixture_steps.get(self.target_fixture)

    @property
    def search_stopped_after_target(self) -> bool:
        history = self.completed_fixture_history
        return bool(
            self.target_found
            and history
            and history[-1] == self.target_fixture
            and not self.search_after_target_detected
        )

    def raw_stage_values(self) -> dict[str, bool]:
        """Return current detector values before ordered qualification."""

        return {
            "one_drawer_checked": bool(self.completed_fixture_history),
            "target_found": bool(
                self.target_found
                and self.target_fixture in self.completed_fixture_history
            ),
            "search_stopped_after_target": self.search_stopped_after_target,
            "button_pressed": bool(self.valid_button_press_steps),
            "correct_press_count": bool(self.domain_success),
        }

    def stage_observation_steps(self) -> dict[str, int | None]:
        history = self.completed_fixture_history
        first_fixture = history[0] if history else None
        return {
            "one_drawer_checked": (
                self.completed_fixture_steps.get(first_fixture)
                if first_fixture is not None
                else None
            ),
            "target_found": self.target_completion_step,
            "search_stopped_after_target": self.target_completion_step,
            "button_pressed": (
                self.valid_button_press_steps[0]
                if self.valid_button_press_steps
                else None
            ),
            "correct_press_count": (
                self.valid_button_press_steps[-1]
                if self.valid_button_press_steps
                else None
            ),
        }


@dataclass(frozen=True, slots=True)
class StageEvalResult:
    """Observed and currently qualified state for one TS02 stage."""

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
class TS02EvalResult:
    """Immutable verdict returned by the TS02 evaluator."""

    success: bool
    terminal: bool
    failure_code: str | None
    stages: Mapping[str, StageEvalResult]
    details: Mapping[str, Any] = field(default_factory=dict)

    def to_diagnostics(self) -> dict[str, Any]:
        payload = dict(self.details or {})
        stages = {name: self.stages[name].to_dict() for name in STAGE_ORDER}
        payload.update(
            {
                "success": bool(self.success),
                "terminal": bool(self.terminal),
                "failure_code": self.failure_code,
                "stages": stages,
                "stage_checks": {
                    name: bool(values["qualified"])
                    for name, values in stages.items()
                },
                "stage_observed": {
                    name: bool(values["observed"])
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

    def to_report_stages(self) -> dict[str, Any]:
        stages = {
            name: {
                "applicable": True,
                "passed": bool(self.stages[name].qualified),
                "first_pass_step": (
                    self.stages[name].qualified_step
                    if self.stages[name].qualified
                    else None
                ),
            }
            for name in REPORT_STAGE_ORDER
        }
        return {
            "schema_version": "membench.report_stages/v1",
            "task_id": TS02_REPORT_PROFILE.task_id,
            "profile_id": REPORT_STAGE_PROFILE_ID,
            "order": list(REPORT_STAGE_ORDER),
            "stages": stages,
        }


class TS02Evaluator:
    """Idempotent evaluator for one seed-resolved TS02 trajectory."""

    def __init__(
        self,
        *,
        fixture_names: Sequence[str],
        target_fixture: str,
    ) -> None:
        names = tuple(str(name) for name in fixture_names)
        target = str(target_fixture)
        if len(names) < 2 or len(set(names)) != len(names):
            raise ValueError("fixture_names must contain at least two unique names")
        if target not in names:
            raise ValueError("target_fixture must name one of fixture_names")
        self.fixture_names = names
        self.target_fixture = target
        self._first_observed_steps: dict[str, int | None] = {
            name: None for name in STAGE_ORDER
        }
        self._first_qualified_steps: dict[str, int | None] = {
            name: None for name in STAGE_ORDER
        }
        self._ever_qualified = {name: False for name in STAGE_ORDER}
        self._last_step_id: int | None = None
        self._cached_result: TS02EvalResult | None = None

    def reset(self, snapshot: TS02EvaluationSnapshot) -> TS02EvalResult:
        self._validate_configuration(snapshot)
        self._first_observed_steps = {name: None for name in STAGE_ORDER}
        self._first_qualified_steps = {name: None for name in STAGE_ORDER}
        self._ever_qualified = {name: False for name in STAGE_ORDER}
        self._last_step_id = snapshot.step_id
        self._cached_result = self._evaluate(snapshot)
        return self._cached_result

    def observe(self, snapshot: TS02EvaluationSnapshot) -> TS02EvalResult:
        self._validate_configuration(snapshot)
        if self._last_step_id is None:
            return self.reset(snapshot)
        if snapshot.step_id == self._last_step_id:
            if self._cached_result is None:
                raise RuntimeError("TS02 evaluator has no cached result")
            return self._cached_result
        if snapshot.step_id < self._last_step_id:
            raise ValueError(
                "TS02 evaluation step moved backwards: "
                f"{snapshot.step_id} < {self._last_step_id}"
            )
        self._last_step_id = snapshot.step_id
        self._cached_result = self._evaluate(snapshot)
        return self._cached_result

    def result(self) -> TS02EvalResult:
        if self._cached_result is None:
            raise RuntimeError("TS02 evaluator has not been reset")
        return self._cached_result

    def _validate_configuration(self, snapshot: TS02EvaluationSnapshot) -> None:
        if (
            snapshot.fixture_names != self.fixture_names
            or snapshot.target_fixture != self.target_fixture
        ):
            raise ValueError(
                "TS02 evaluation snapshot fixture configuration changed within "
                "an episode"
            )

    def _evaluate(self, snapshot: TS02EvaluationSnapshot) -> TS02EvalResult:
        raw = snapshot.raw_stage_values()
        observation_steps = snapshot.stage_observation_steps()
        currently_qualified: dict[str, bool] = {}

        for name in STAGE_ORDER:
            if raw[name] and self._first_observed_steps[name] is None:
                self._first_observed_steps[name] = (
                    observation_steps[name]
                    if observation_steps[name] is not None
                    else snapshot.step_id
                )
            prerequisite = STAGE_PREREQUISITES[name]
            qualified = bool(
                raw[name]
                and (
                    prerequisite is None
                    or currently_qualified.get(prerequisite, False)
                )
            )
            currently_qualified[name] = qualified
            if qualified:
                self._ever_qualified[name] = True
                if self._first_qualified_steps[name] is None:
                    self._first_qualified_steps[name] = (
                        observation_steps[name]
                        if observation_steps[name] is not None
                        else snapshot.step_id
                    )

        stages = {
            name: self._stage_result(
                name,
                snapshot=snapshot,
                raw=raw,
                currently_qualified=currently_qualified,
            )
            for name in STAGE_ORDER
        }
        details = dict(snapshot.details)
        evaluation = dict(details.get("evaluation") or {})
        evaluation.update(
            {
                "schema_version": 1,
                "step_id": int(snapshot.step_id),
                "stage_order": list(STAGE_ORDER),
                "stage_prerequisites": dict(STAGE_PREREQUISITES),
                "qualification_policy": (
                    "ordered_trajectory_verdict_with_revocable_search_stop"
                ),
                "raw_stage_checks": raw,
            }
        )
        details.update(
            {
                "evaluation": evaluation,
                "fixture_names": list(snapshot.fixture_names),
                "target_fixture": snapshot.target_fixture,
                "completed_fixture_history": list(
                    snapshot.completed_fixture_history
                ),
                "completed_fixture_steps": dict(snapshot.completed_fixture_steps),
                "target_found": bool(snapshot.target_found),
                "search_after_target_detected": bool(
                    snapshot.search_after_target_detected
                ),
                "valid_button_press_steps": list(
                    snapshot.valid_button_press_steps
                ),
                "completion_button_press_count": int(
                    snapshot.actual_button_press_count
                ),
                "required_button_presses": int(
                    snapshot.required_button_press_count
                ),
                "domain_success": bool(snapshot.domain_success),
            }
        )
        success = bool(stages["correct_press_count"].qualified)
        return TS02EvalResult(
            success=success,
            terminal=bool(success or snapshot.failure_reason is not None),
            failure_code=(None if success else snapshot.failure_reason),
            stages=stages,
            details=details,
        )

    def _stage_result(
        self,
        name: str,
        *,
        snapshot: TS02EvaluationSnapshot,
        raw: Mapping[str, bool],
        currently_qualified: Mapping[str, bool],
    ) -> StageEvalResult:
        qualified = bool(currently_qualified[name])
        prerequisite = STAGE_PREREQUISITES[name]
        invalidated = bool(self._ever_qualified[name] and not qualified)
        if qualified:
            reason = None
        elif prerequisite is not None and not currently_qualified[prerequisite]:
            reason = f"waiting_for_prerequisite:{prerequisite}"
        elif name == "search_stopped_after_target":
            reason = (
                "search_continued_after_target"
                if snapshot.search_after_target_detected
                else "target_not_last_checked"
            )
            invalidated = bool(
                invalidated or snapshot.search_after_target_detected
            )
        elif name == "correct_press_count" and (
            snapshot.actual_button_press_count
            != snapshot.required_button_press_count
        ):
            reason = "completion_button_count_does_not_match_search_count"
            invalidated = bool(invalidated or snapshot.actual_button_press_count > 0)
        elif not raw[name]:
            reason = snapshot.failure_reason or "not_observed"
        else:
            reason = f"prerequisite_not_qualified:{prerequisite}"
        return StageEvalResult(
            applicable=True,
            observed=self._first_observed_steps[name] is not None,
            qualified=qualified,
            first_observed_step=self._first_observed_steps[name],
            qualified_step=self._first_qualified_steps[name],
            invalidated=invalidated,
            reason=reason,
        )


def evaluate_initialized_runtime(runtime: Any) -> TS02EvalResult:
    """Evaluate one initialized runtime through its task-local evaluator."""

    initialized = getattr(runtime, "initialized", None) or runtime
    metadata = getattr(initialized, "metadata", {}) or {}
    evaluator = metadata.get(TASK_EVALUATOR_METADATA_KEY)
    env = (
        getattr(initialized, "env", None)
        or getattr(runtime, "env", None)
        or initialized
    )
    if evaluator is None:
        evaluator = getattr(env, "_ts02_evaluator", None)
    if not isinstance(evaluator, TS02Evaluator):
        raise RuntimeError("TS02 runtime has no initialized task evaluator")
    snapshot_builder = getattr(env, "evaluation_snapshot", None)
    if not callable(snapshot_builder):
        raise AttributeError("TS02 environment has no evaluation_snapshot()")
    return evaluator.observe(snapshot_builder())


def success_for_runtime(
    runtime: Any,
    info: Mapping[str, Any] | None = None,
) -> bool:
    del info
    return bool(evaluate_initialized_runtime(runtime).success)


def diagnostics_for_runtime(runtime: Any) -> Mapping[str, Any]:
    return evaluate_initialized_runtime(runtime).to_diagnostics()


def report_stages_for_runtime(runtime: Any) -> Mapping[str, Any]:
    return evaluate_initialized_runtime(runtime).to_report_stages()


__all__ = [
    "REPORT_STAGE_ORDER",
    "REPORT_STAGE_PROFILE_ID",
    "STAGE_ORDER",
    "STAGE_PREREQUISITES",
    "TASK_EVALUATOR_METADATA_KEY",
    "StageEvalResult",
    "TS02EvalResult",
    "TS02EvaluationSnapshot",
    "TS02Evaluator",
    "diagnostics_for_runtime",
    "evaluate_initialized_runtime",
    "report_stages_for_runtime",
    "success_for_runtime",
]
