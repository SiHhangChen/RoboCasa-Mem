"""Authoritative WX-01 benchmark evaluation state machine.

The environment supplies the historical wipe fact and current physical state.
This module qualifies those signals into the stable three-stage public funnel.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from bench.evaluation.report_profiles import WX01_REPORT_PROFILE


STAGE_ORDER = WX01_REPORT_PROFILE.stage_ids
STAGE_PREREQUISITES: dict[str, str | None] = {
    "wipe_completed": None,
    "cleaning_tool_released": "wipe_completed",
    "success": "cleaning_tool_released",
}
REPORT_STAGE_ORDER = STAGE_ORDER
REPORT_STAGE_PROFILE_ID = WX01_REPORT_PROFILE.profile_id
TASK_EVALUATOR_METADATA_KEY = "task_evaluator"


@dataclass(frozen=True, slots=True)
class WX01EvaluationSnapshot:
    """One read-only semantic and physical observation."""

    step_id: int
    remembered_area_wiped: bool = False
    cleaning_tool_released: bool = False
    blocking_object_at_origin: bool = False
    blocking_object_on_dining_table: bool = False
    blocking_object_released: bool = False
    domain_success: bool = False
    details: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        step_id = int(self.step_id)
        if step_id < 0:
            raise ValueError("WX-01 evaluation step_id must be non-negative")
        wiped = bool(self.remembered_area_wiped)
        tool_released = bool(self.cleaning_tool_released)
        at_origin = bool(self.blocking_object_at_origin)
        on_table = bool(self.blocking_object_on_dining_table)
        object_released = bool(self.blocking_object_released)
        success = bool(self.domain_success)
        if success and not (wiped and tool_released and at_origin and on_table):
            raise ValueError(
                "WX-01 domain_success requires completed cleaning and the "
                "blocking object at its original dining-table position"
            )
        object.__setattr__(self, "step_id", step_id)
        object.__setattr__(self, "remembered_area_wiped", wiped)
        object.__setattr__(self, "cleaning_tool_released", tool_released)
        object.__setattr__(self, "blocking_object_at_origin", at_origin)
        object.__setattr__(self, "blocking_object_on_dining_table", on_table)
        object.__setattr__(self, "blocking_object_released", object_released)
        object.__setattr__(self, "domain_success", success)
        object.__setattr__(self, "details", dict(self.details or {}))

    @property
    def cleaning_tool_released_after_wipe(self) -> bool:
        return bool(self.remembered_area_wiped and self.cleaning_tool_released)

    def raw_stage_values(self) -> dict[str, bool]:
        return {
            "wipe_completed": self.remembered_area_wiped,
            "cleaning_tool_released": self.cleaning_tool_released_after_wipe,
            "success": self.domain_success,
        }


@dataclass(frozen=True, slots=True)
class StageEvalResult:
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
class WX01EvalResult:
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
            "task_id": WX01_REPORT_PROFILE.task_id,
            "profile_id": REPORT_STAGE_PROFILE_ID,
            "order": list(REPORT_STAGE_ORDER),
            "stages": stages,
        }


class WX01Evaluator:
    """History-aware evaluator for wiping, cleanup, and object return."""

    def __init__(self) -> None:
        self._latched = {name: False for name in STAGE_ORDER[:-1]}
        self._first_observed_steps = {name: None for name in STAGE_ORDER}
        self._first_qualified_steps = {name: None for name in STAGE_ORDER}
        self._rejected: dict[str, list[dict[str, Any]]] = {
            name: [] for name in STAGE_ORDER
        }
        self._previous_raw = {name: False for name in STAGE_ORDER}
        self._ever_success = False
        self._last_step_id: int | None = None
        self._cached_result: WX01EvalResult | None = None

    def reset(self, snapshot: WX01EvaluationSnapshot) -> WX01EvalResult:
        self._latched = {name: False for name in STAGE_ORDER[:-1]}
        self._first_observed_steps = {name: None for name in STAGE_ORDER}
        self._first_qualified_steps = {name: None for name in STAGE_ORDER}
        self._rejected = {name: [] for name in STAGE_ORDER}
        self._ever_success = False
        raw = snapshot.raw_stage_values()
        for name, value in raw.items():
            if value:
                self._first_observed_steps[name] = snapshot.step_id
                self._reject(name, snapshot.step_id, "present_at_reset")
        self._previous_raw = raw
        self._last_step_id = snapshot.step_id
        self._cached_result = self._build_result(
            snapshot,
            success_current=False,
        )
        return self._cached_result

    def observe(self, snapshot: WX01EvaluationSnapshot) -> WX01EvalResult:
        if self._last_step_id is None:
            return self.reset(snapshot)
        if snapshot.step_id == self._last_step_id:
            if self._cached_result is None:
                raise RuntimeError("WX-01 evaluator has no cached result")
            return self._cached_result
        if snapshot.step_id < self._last_step_id:
            raise ValueError(
                "WX-01 evaluation step moved backwards: "
                f"{snapshot.step_id} < {self._last_step_id}"
            )

        raw = snapshot.raw_stage_values()
        for name in ("wipe_completed", "cleaning_tool_released"):
            if raw[name] and self._first_observed_steps[name] is None:
                self._first_observed_steps[name] = snapshot.step_id
            if self._latched[name] or not raw[name] or self._previous_raw[name]:
                continue
            prerequisite = STAGE_PREREQUISITES[name]
            if prerequisite is not None and not self._latched[prerequisite]:
                self._reject(
                    name,
                    snapshot.step_id,
                    f"prerequisite_not_qualified:{prerequisite}",
                )
            else:
                self._qualify(name, snapshot.step_id)

        if raw["success"] and self._first_observed_steps["success"] is None:
            self._first_observed_steps["success"] = snapshot.step_id
        success_current = bool(
            raw["success"] and self._latched["cleaning_tool_released"]
        )
        if raw["success"] and not success_current:
            self._reject(
                "success",
                snapshot.step_id,
                "prerequisite_not_qualified:cleaning_tool_released",
            )
        if success_current:
            self._ever_success = True
            if self._first_qualified_steps["success"] is None:
                self._first_qualified_steps["success"] = snapshot.step_id

        self._previous_raw = raw
        self._last_step_id = snapshot.step_id
        self._cached_result = self._build_result(
            snapshot,
            success_current=success_current,
        )
        return self._cached_result

    def result(self) -> WX01EvalResult:
        if self._cached_result is None:
            raise RuntimeError("WX-01 evaluator has not been reset")
        return self._cached_result

    def _qualify(self, name: str, step_id: int) -> None:
        self._latched[name] = True
        if self._first_qualified_steps[name] is None:
            self._first_qualified_steps[name] = int(step_id)

    def _reject(self, name: str, step_id: int, reason: str) -> None:
        entry = {"step": int(step_id), "reason": str(reason)}
        if entry not in self._rejected[name]:
            self._rejected[name].append(entry)

    def _build_result(
        self,
        snapshot: WX01EvaluationSnapshot,
        *,
        success_current: bool,
    ) -> WX01EvalResult:
        stages = {
            name: self._stage_result(
                name,
                snapshot=snapshot,
                success_current=success_current,
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
                    "historical_wipe_then_tool_release_with_final_state_success"
                ),
                "raw_stage_checks": snapshot.raw_stage_values(),
                "rejected_stage_observations": {
                    name: list(values)
                    for name, values in self._rejected.items()
                    if values
                },
            }
        )
        details["evaluation"] = evaluation
        details.update(
            {
                "remembered_area_wiped": snapshot.remembered_area_wiped,
                "cleaning_tool_released": snapshot.cleaning_tool_released,
                "blocking_object_at_origin": snapshot.blocking_object_at_origin,
                "blocking_object_on_dining_table": (
                    snapshot.blocking_object_on_dining_table
                ),
                "blocking_object_released": snapshot.blocking_object_released,
                "domain_success": snapshot.domain_success,
            }
        )
        return WX01EvalResult(
            success=bool(success_current),
            terminal=bool(success_current),
            failure_code=None,
            stages=stages,
            details=details,
        )

    def _stage_result(
        self,
        name: str,
        *,
        snapshot: WX01EvaluationSnapshot,
        success_current: bool,
    ) -> StageEvalResult:
        if name == "success":
            qualified = bool(success_current)
            invalidated = bool(self._ever_success and not qualified)
        else:
            qualified = bool(self._latched[name])
            invalidated = False
        rejected = self._rejected[name]
        prerequisite = STAGE_PREREQUISITES[name]
        if qualified:
            reason = None
        elif name == "success" and invalidated:
            reason = "final_state_regressed"
        elif rejected:
            invalidated = True
            reason = str(rejected[-1]["reason"])
        elif prerequisite is not None and not self._latched[prerequisite]:
            reason = f"waiting_for_prerequisite:{prerequisite}"
        elif not snapshot.raw_stage_values()[name]:
            reason = "not_observed"
        else:
            reason = "waiting_for_qualified_event"
        return StageEvalResult(
            applicable=True,
            observed=self._first_observed_steps[name] is not None,
            qualified=qualified,
            first_observed_step=self._first_observed_steps[name],
            qualified_step=self._first_qualified_steps[name],
            invalidated=invalidated,
            reason=reason,
        )


def evaluate_initialized_runtime(runtime: Any) -> WX01EvalResult:
    initialized = getattr(runtime, "initialized", None) or runtime
    metadata = getattr(initialized, "metadata", {}) or {}
    evaluator = metadata.get(TASK_EVALUATOR_METADATA_KEY)
    env = (
        getattr(initialized, "env", None)
        or getattr(runtime, "env", None)
        or initialized
    )
    if evaluator is None:
        evaluator = getattr(env, "_wx01_evaluator", None)
    if not isinstance(evaluator, WX01Evaluator):
        raise RuntimeError("WX-01 runtime has no initialized task evaluator")
    snapshot_builder = getattr(env, "evaluation_snapshot", None)
    if not callable(snapshot_builder):
        raise AttributeError("WX-01 environment has no evaluation_snapshot()")
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
    "WX01EvalResult",
    "WX01EvaluationSnapshot",
    "WX01Evaluator",
    "diagnostics_for_runtime",
    "evaluate_initialized_runtime",
    "report_stages_for_runtime",
    "success_for_runtime",
]
