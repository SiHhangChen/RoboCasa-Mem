"""History-aware benchmark evaluator for WX-02."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from bench.evaluation.report_profiles import WX02_REPORT_PROFILE


STAGE_ORDER = WX02_REPORT_PROFILE.stage_ids
STAGE_PREREQUISITES: dict[str, str | None] = {
    "tool_acquired": None,
    "primary_cup_moved": "tool_acquired",
    "primary_stain_cleaned": "primary_cup_moved",
    "primary_cup_returned": "primary_stain_cleaned",
    "secondary_cup_moved": "primary_cup_returned",
    "secondary_stain_cleaned": "secondary_cup_moved",
    "secondary_cup_returned": "secondary_stain_cleaned",
    "tool_returned": "secondary_cup_returned",
    "success": "tool_returned",
}
REPORT_STAGE_ORDER = STAGE_ORDER
REPORT_STAGE_PROFILE_ID = WX02_REPORT_PROFILE.profile_id
TASK_EVALUATOR_METADATA_KEY = "task_evaluator"


@dataclass(frozen=True, slots=True)
class WX02EvaluationSnapshot:
    step_id: int
    cleaning_tool_acquired: bool = False
    primary_cup_moved: bool = False
    primary_stain_cleaned: bool = False
    primary_cup_returned: bool = False
    secondary_cup_moved: bool = False
    secondary_stain_cleaned: bool = False
    secondary_cup_returned: bool = False
    cleaning_tool_returned: bool = False
    cleaning_tool_released: bool = False
    primary_cup_at_origin: bool = False
    primary_cup_on_dining_table: bool = False
    secondary_cup_at_origin: bool = False
    secondary_cup_on_dining_table: bool = False
    cleaning_tool_in_return_region: bool = False
    cleaning_tool_on_dining_table: bool = False
    primary_cup_released: bool = False
    secondary_cup_released: bool = False
    domain_success: bool = False
    details: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        step_id = int(self.step_id)
        if step_id < 0:
            raise ValueError("WX-02 evaluation step_id must be non-negative")
        object.__setattr__(self, "step_id", step_id)
        for name in self.__dataclass_fields__:
            if name not in {"step_id", "details"}:
                object.__setattr__(self, name, bool(getattr(self, name)))
        object.__setattr__(self, "details", dict(self.details or {}))
        if self.domain_success and not all(
            (
                self.cleaning_tool_acquired,
                self.primary_cup_moved,
                self.primary_stain_cleaned,
                self.primary_cup_returned,
                self.secondary_cup_moved,
                self.secondary_stain_cleaned,
                self.secondary_cup_returned,
                self.cleaning_tool_returned,
                self.cleaning_tool_released,
                self.primary_cup_at_origin,
                self.primary_cup_on_dining_table,
                self.secondary_cup_at_origin,
                self.secondary_cup_on_dining_table,
                self.cleaning_tool_in_return_region,
                self.cleaning_tool_on_dining_table,
            )
        ):
            raise ValueError(
                "WX-02 domain_success requires both ordered cup-cleaning loops "
                "and a final released sponge return"
            )

    def raw_stage_values(self) -> dict[str, bool]:
        return {
            "tool_acquired": self.cleaning_tool_acquired,
            "primary_cup_moved": self.primary_cup_moved,
            "primary_stain_cleaned": self.primary_stain_cleaned,
            "primary_cup_returned": self.primary_cup_returned,
            "secondary_cup_moved": self.secondary_cup_moved,
            "secondary_stain_cleaned": self.secondary_stain_cleaned,
            "secondary_cup_returned": self.secondary_cup_returned,
            "tool_returned": self.cleaning_tool_returned,
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
class WX02EvalResult:
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
                    name: values["qualified_step"] for name, values in stages.items()
                },
            }
        )
        return payload

    def to_report_stages(self) -> dict[str, Any]:
        return {
            "schema_version": "membench.report_stages/v1",
            "task_id": WX02_REPORT_PROFILE.task_id,
            "profile_id": REPORT_STAGE_PROFILE_ID,
            "order": list(REPORT_STAGE_ORDER),
            "stages": {
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
            },
        }


class WX02Evaluator:
    """Qualify the strict two-cup retrieve/clean/return sequence."""

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
        self._cached_result: WX02EvalResult | None = None

    def reset(self, snapshot: WX02EvaluationSnapshot) -> WX02EvalResult:
        self.__init__()
        raw = snapshot.raw_stage_values()
        for name, value in raw.items():
            if value:
                self._first_observed_steps[name] = snapshot.step_id
                self._reject(name, snapshot.step_id, "present_at_reset")
        self._previous_raw = raw
        self._last_step_id = snapshot.step_id
        self._cached_result = self._build_result(snapshot, success_current=False)
        return self._cached_result

    def observe(self, snapshot: WX02EvaluationSnapshot) -> WX02EvalResult:
        if self._last_step_id is None:
            return self.reset(snapshot)
        if snapshot.step_id == self._last_step_id:
            if self._cached_result is None:
                raise RuntimeError("WX-02 evaluator has no cached result")
            return self._cached_result
        if snapshot.step_id < self._last_step_id:
            raise ValueError(
                "WX-02 evaluation step moved backwards: "
                f"{snapshot.step_id} < {self._last_step_id}"
            )

        raw = snapshot.raw_stage_values()
        for name in STAGE_ORDER[:-1]:
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
        success_prerequisite = STAGE_PREREQUISITES["success"]
        success_current = bool(
            raw["success"]
            and success_prerequisite is not None
            and self._latched[success_prerequisite]
        )
        if raw["success"] and not success_current:
            self._reject(
                "success",
                snapshot.step_id,
                f"prerequisite_not_qualified:{success_prerequisite}",
            )
        if success_current:
            self._ever_success = True
            if self._first_qualified_steps["success"] is None:
                self._first_qualified_steps["success"] = snapshot.step_id

        self._previous_raw = raw
        self._last_step_id = snapshot.step_id
        self._cached_result = self._build_result(
            snapshot, success_current=success_current
        )
        return self._cached_result

    def result(self) -> WX02EvalResult:
        if self._cached_result is None:
            raise RuntimeError("WX-02 evaluator has not been reset")
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
        snapshot: WX02EvaluationSnapshot,
        *,
        success_current: bool,
    ) -> WX02EvalResult:
        stages = {
            name: self._stage_result(
                name, snapshot=snapshot, success_current=success_current
            )
            for name in STAGE_ORDER
        }
        details = dict(snapshot.details)
        details["evaluation"] = {
            **dict(details.get("evaluation") or {}),
            "schema_version": 1,
            "step_id": int(snapshot.step_id),
            "stage_order": list(STAGE_ORDER),
            "stage_prerequisites": dict(STAGE_PREREQUISITES),
            "qualification_policy": "ordered_two_cup_local_memory_loops",
            "raw_stage_checks": snapshot.raw_stage_values(),
            "rejected_stage_observations": {
                name: list(values)
                for name, values in self._rejected.items()
                if values
            },
        }
        details.update(
            {
                name: getattr(snapshot, name)
                for name in snapshot.__dataclass_fields__
                if name not in {"step_id", "details"}
            }
        )
        return WX02EvalResult(
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
        snapshot: WX02EvaluationSnapshot,
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


def evaluate_initialized_runtime(runtime: Any) -> WX02EvalResult:
    initialized = getattr(runtime, "initialized", None) or runtime
    metadata = getattr(initialized, "metadata", {}) or {}
    evaluator = metadata.get(TASK_EVALUATOR_METADATA_KEY)
    env = (
        getattr(initialized, "env", None)
        or getattr(runtime, "env", None)
        or initialized
    )
    if evaluator is None:
        evaluator = getattr(env, "_wx02_evaluator", None)
    if not isinstance(evaluator, WX02Evaluator):
        raise RuntimeError("WX-02 runtime has no initialized task evaluator")
    snapshot_builder = getattr(env, "evaluation_snapshot", None)
    if not callable(snapshot_builder):
        raise AttributeError("WX-02 environment has no evaluation_snapshot()")
    return evaluator.observe(snapshot_builder())


def success_for_runtime(
    runtime: Any, info: Mapping[str, Any] | None = None
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
    "WX02EvalResult",
    "WX02EvaluationSnapshot",
    "WX02Evaluator",
    "diagnostics_for_runtime",
    "evaluate_initialized_runtime",
    "report_stages_for_runtime",
    "success_for_runtime",
]
