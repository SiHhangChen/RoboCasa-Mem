"""Authoritative ordered WA-05 evaluation state machine.

The environment owns physical fill, staging, and answer-button detection.
This module qualifies those facts into the three public benchmark stages.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from bench.evaluation.report_profiles import WA05_REPORT_PROFILE
from bench.tasks.wa05.domain.contracts import CUP_IDS, SERVICE_SLOTS, TEMPERATURES


STAGE_ORDER = WA05_REPORT_PROFILE.stage_ids
STAGE_PREREQUISITES: dict[str, str | None] = {
    "cup_filled": None,
    "cup_staged": "cup_filled",
    "success": "cup_staged",
}
REPORT_STAGE_ORDER = STAGE_ORDER
REPORT_STAGE_PROFILE_ID = WA05_REPORT_PROFILE.profile_id
TASK_EVALUATOR_METADATA_KEY = "task_evaluator"


def _temperature(value: Any, *, field_name: str) -> str | None:
    if value is None:
        return None
    temperature = str(value).strip().lower()
    if temperature not in TEMPERATURES:
        raise ValueError(f"{field_name} must be one of {TEMPERATURES}")
    return temperature


@dataclass(frozen=True, slots=True)
class WA05EvaluationSnapshot:
    """One debounced physical and semantic observation."""

    step_id: int
    cup_id: str
    service_slot: str
    filled_temperature: str | None = None
    staged_before_answer: bool = False
    pressed_temperature: str | None = None
    domain_success: bool = False
    absorbing_failure: bool = False
    failure_code: str | None = None
    details: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        step_id = int(self.step_id)
        cup_id = str(self.cup_id)
        service_slot = str(self.service_slot)
        if step_id < 0:
            raise ValueError("step_id must be non-negative")
        if cup_id not in CUP_IDS:
            raise ValueError(f"cup_id must be one of {CUP_IDS}")
        if service_slot not in SERVICE_SLOTS:
            raise ValueError(f"service_slot must be one of {SERVICE_SLOTS}")
        filled = _temperature(
            self.filled_temperature,
            field_name="filled_temperature",
        )
        pressed = _temperature(
            self.pressed_temperature,
            field_name="pressed_temperature",
        )
        staged = bool(self.staged_before_answer)
        success = bool(self.domain_success)
        failed = bool(self.absorbing_failure)
        failure_code = (
            None if self.failure_code is None else str(self.failure_code)
        )
        if staged and filled is None:
            raise ValueError("staged_before_answer requires a verified fill")
        if success and (
            filled is None
            or not staged
            or pressed != filled
            or failed
        ):
            raise ValueError(
                "domain_success requires fill, staging, matching answer, "
                "and no absorbing failure"
            )
        if failed and failure_code is None:
            raise ValueError("absorbing_failure requires failure_code")

        object.__setattr__(self, "step_id", step_id)
        object.__setattr__(self, "cup_id", cup_id)
        object.__setattr__(self, "service_slot", service_slot)
        object.__setattr__(self, "filled_temperature", filled)
        object.__setattr__(self, "staged_before_answer", staged)
        object.__setattr__(self, "pressed_temperature", pressed)
        object.__setattr__(self, "domain_success", success)
        object.__setattr__(self, "absorbing_failure", failed)
        object.__setattr__(self, "failure_code", failure_code)
        object.__setattr__(self, "details", dict(self.details or {}))

    def raw_stage_values(self) -> dict[str, bool]:
        return {
            "cup_filled": self.filled_temperature is not None,
            "cup_staged": bool(self.staged_before_answer),
            "success": bool(self.domain_success),
        }


@dataclass(frozen=True, slots=True)
class StageEvalResult:
    """Observed and currently qualified state for one WA-05 stage."""

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
class WA05EvalResult:
    """Immutable verdict returned by the WA-05 evaluator."""

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
            "task_id": WA05_REPORT_PROFILE.task_id,
            "profile_id": REPORT_STAGE_PROFILE_ID,
            "order": list(REPORT_STAGE_ORDER),
            "stages": stages,
        }


class WA05Evaluator:
    """Idempotent evaluator for one seed-resolved WA-05 episode."""

    def __init__(self, *, cup_id: str, service_slot: str) -> None:
        cup = str(cup_id)
        slot = str(service_slot)
        if cup not in CUP_IDS:
            raise ValueError(f"cup_id must be one of {CUP_IDS}")
        if slot not in SERVICE_SLOTS:
            raise ValueError(f"service_slot must be one of {SERVICE_SLOTS}")
        self.cup_id = cup
        self.service_slot = slot
        self._first_observed_steps: dict[str, int | None] = {
            name: None for name in STAGE_ORDER
        }
        self._first_qualified_steps: dict[str, int | None] = {
            name: None for name in STAGE_ORDER
        }
        self._ever_qualified = {name: False for name in STAGE_ORDER}
        self._last_step_id: int | None = None
        self._cached_result: WA05EvalResult | None = None

    def reset(self, snapshot: WA05EvaluationSnapshot) -> WA05EvalResult:
        self._validate_configuration(snapshot)
        self._first_observed_steps = {name: None for name in STAGE_ORDER}
        self._first_qualified_steps = {name: None for name in STAGE_ORDER}
        self._ever_qualified = {name: False for name in STAGE_ORDER}
        self._last_step_id = snapshot.step_id
        self._cached_result = self._evaluate(snapshot)
        return self._cached_result

    def observe(self, snapshot: WA05EvaluationSnapshot) -> WA05EvalResult:
        self._validate_configuration(snapshot)
        if self._last_step_id is None:
            return self.reset(snapshot)
        if snapshot.step_id == self._last_step_id:
            if self._cached_result is None:
                raise RuntimeError("WA-05 evaluator has no cached result")
            return self._cached_result
        if snapshot.step_id < self._last_step_id:
            raise ValueError(
                "WA-05 evaluation step moved backwards: "
                f"{snapshot.step_id} < {self._last_step_id}"
            )
        self._last_step_id = snapshot.step_id
        self._cached_result = self._evaluate(snapshot)
        return self._cached_result

    def result(self) -> WA05EvalResult:
        if self._cached_result is None:
            raise RuntimeError("WA-05 evaluator has not been reset")
        return self._cached_result

    def _validate_configuration(self, snapshot: WA05EvaluationSnapshot) -> None:
        if (
            snapshot.cup_id != self.cup_id
            or snapshot.service_slot != self.service_slot
        ):
            raise ValueError(
                "WA-05 evaluation snapshot configuration changed within "
                "an episode"
            )

    def _evaluate(self, snapshot: WA05EvaluationSnapshot) -> WA05EvalResult:
        raw = snapshot.raw_stage_values()
        currently_qualified: dict[str, bool] = {}

        for name in STAGE_ORDER:
            if raw[name] and self._first_observed_steps[name] is None:
                self._first_observed_steps[name] = snapshot.step_id
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
                    self._first_qualified_steps[name] = snapshot.step_id

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
                    "ordered_semantic_events_with_episode_level_success"
                ),
                "raw_stage_checks": raw,
            }
        )
        details.update(
            {
                "evaluation": evaluation,
                "cup_id": snapshot.cup_id,
                "service_slot": snapshot.service_slot,
                "filled_temperature": snapshot.filled_temperature,
                "staged_before_answer": bool(snapshot.staged_before_answer),
                "pressed_temperature": snapshot.pressed_temperature,
                "domain_success": bool(snapshot.domain_success),
                "absorbing_failure": bool(snapshot.absorbing_failure),
            }
        )
        success = bool(stages["success"].qualified)
        return WA05EvalResult(
            success=success,
            terminal=bool(success or snapshot.absorbing_failure),
            failure_code=(None if success else snapshot.failure_code),
            stages=stages,
            details=details,
        )

    def _stage_result(
        self,
        name: str,
        *,
        snapshot: WA05EvaluationSnapshot,
        raw: Mapping[str, bool],
        currently_qualified: Mapping[str, bool],
    ) -> StageEvalResult:
        qualified = bool(currently_qualified[name])
        prerequisite = STAGE_PREREQUISITES[name]
        invalidated = bool(self._ever_qualified[name] and not qualified)
        if qualified:
            reason = None
        elif name == "success" and snapshot.absorbing_failure:
            reason = snapshot.failure_code
            invalidated = True
        elif prerequisite is not None and not currently_qualified[prerequisite]:
            reason = f"waiting_for_prerequisite:{prerequisite}"
        elif not raw[name]:
            reason = "not_observed"
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


def evaluate_initialized_runtime(runtime: Any) -> WA05EvalResult:
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
        evaluator = getattr(env, "_wa05_evaluator", None)
    if not isinstance(evaluator, WA05Evaluator):
        raise RuntimeError("WA-05 runtime has no initialized task evaluator")
    snapshot_builder = getattr(env, "evaluation_snapshot", None)
    if not callable(snapshot_builder):
        raise AttributeError("WA-05 environment has no evaluation_snapshot()")
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
    "WA05EvalResult",
    "WA05EvaluationSnapshot",
    "WA05Evaluator",
    "diagnostics_for_runtime",
    "evaluate_initialized_runtime",
    "report_stages_for_runtime",
    "success_for_runtime",
]
