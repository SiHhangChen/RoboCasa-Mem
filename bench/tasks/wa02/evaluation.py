"""Authoritative ordered evaluator for the WA-02 seven-stage funnel."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from bench.evaluation.report_profiles import WA02_REPORT_PROFILE
from bench.tasks.wa02.domain.contracts import CUP_IDS, SERVICE_SLOTS, TEMPERATURES


STAGE_ORDER = (
    "first_cup_pick",
    "first_cup_fill",
    "first_cup_ready",
    "second_cup_pick",
    "second_cup_fill",
    "second_cup_ready",
    "selection_success",
)
STAGE_LABELS = {
    "first_cup_pick": "第一杯取杯",
    "first_cup_fill": "第一杯加水",
    "first_cup_ready": "第一杯就位",
    "second_cup_pick": "第二杯取杯",
    "second_cup_fill": "第二杯加水",
    "second_cup_ready": "第二杯就位",
    "selection_success": "选择成功",
}
STAGE_PREREQUISITES: dict[str, str | None] = {
    name: (None if index == 0 else STAGE_ORDER[index - 1])
    for index, name in enumerate(STAGE_ORDER)
}
TASK_EVALUATOR_METADATA_KEY = "task_evaluator"
REPORT_STAGE_ORDER = WA02_REPORT_PROFILE.stage_ids
REPORT_STAGE_PROFILE_ID = WA02_REPORT_PROFILE.profile_id

_REPORT_STAGE_SOURCES = {
    "first_cup_filled": "first_cup_fill",
    "first_cup_ready": "first_cup_ready",
    "second_cup_filled": "second_cup_fill",
    "both_cups_ready": "second_cup_ready",
    "success": "selection_success",
}


def _cup(value: Any) -> str:
    result = str(value)
    if result not in CUP_IDS:
        raise ValueError(f"invalid WA-02 cup id: {value!r}")
    return result


def _temperature(value: Any) -> str:
    result = str(value).strip().lower()
    if result not in TEMPERATURES:
        raise ValueError(f"invalid WA-02 temperature: {value!r}")
    return result


def _order(values: Sequence[Any]) -> tuple[str, str]:
    if isinstance(values, (str, bytes)) or len(values) != 2:
        raise ValueError("WA-02 fill_cup_order must contain two cups")
    result = tuple(_cup(value) for value in values)
    if set(result) != set(CUP_IDS):
        raise ValueError("WA-02 fill_cup_order must contain each cup exactly once")
    return result  # type: ignore[return-value]


def _temperature_order(values: Sequence[Any]) -> tuple[str, str]:
    if isinstance(values, (str, bytes)) or len(values) != 2:
        raise ValueError("WA-02 fill_temperature_order must contain two temperatures")
    result = tuple(_temperature(value) for value in values)
    if set(result) != set(TEMPERATURES):
        raise ValueError(
            "WA-02 fill_temperature_order must contain hot and cold exactly once"
        )
    return result  # type: ignore[return-value]


def _slot_by_cup(values: Mapping[str, Any]) -> dict[str, str]:
    result = {_cup(cup): str(slot) for cup, slot in values.items()}
    if set(result) != set(CUP_IDS) or set(result.values()) != set(SERVICE_SLOTS):
        raise ValueError("WA-02 service_slot_by_cup must map both cups to both slots")
    return result


@dataclass(frozen=True, slots=True)
class WA02EvaluationSnapshot:
    """Physical and semantic facts sampled at one environment step."""

    step_id: int
    fill_cup_order: tuple[str, str]
    fill_temperature_order: tuple[str, str]
    service_slot_by_cup: Mapping[str, str]
    requested_cup_id: str
    pickup_events_by_cup: Mapping[str, bool] = field(default_factory=dict)
    filled_temperature_by_cup: Mapping[str, str] = field(default_factory=dict)
    staged_slot_by_cup: Mapping[str, str] = field(default_factory=dict)
    final_taken_cup_id: str | None = None
    details: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "step_id", int(self.step_id))
        if self.step_id < 0:
            raise ValueError("WA-02 evaluation step_id must be non-negative")
        object.__setattr__(self, "fill_cup_order", _order(self.fill_cup_order))
        object.__setattr__(
            self,
            "fill_temperature_order",
            _temperature_order(self.fill_temperature_order),
        )
        object.__setattr__(
            self, "service_slot_by_cup", _slot_by_cup(self.service_slot_by_cup)
        )
        object.__setattr__(self, "requested_cup_id", _cup(self.requested_cup_id))
        pickup = {
            _cup(cup): bool(value) for cup, value in self.pickup_events_by_cup.items()
        }
        if set(pickup) - set(CUP_IDS):
            raise ValueError("unknown cup in pickup_events_by_cup")
        object.__setattr__(
            self,
            "pickup_events_by_cup",
            {cup: bool(pickup.get(cup, False)) for cup in CUP_IDS},
        )
        fills = {
            _cup(cup): _temperature(temp)
            for cup, temp in self.filled_temperature_by_cup.items()
        }
        object.__setattr__(self, "filled_temperature_by_cup", fills)
        staged = {
            _cup(cup): str(slot) for cup, slot in self.staged_slot_by_cup.items()
        }
        object.__setattr__(self, "staged_slot_by_cup", staged)
        final = (
            None if self.final_taken_cup_id is None else _cup(self.final_taken_cup_id)
        )
        object.__setattr__(self, "final_taken_cup_id", final)
        object.__setattr__(self, "details", dict(self.details or {}))

    def cup_for_ordinal(self, ordinal: int) -> str:
        return self.fill_cup_order[int(ordinal) - 1]

    def temperature_for_ordinal(self, ordinal: int) -> str:
        return self.fill_temperature_order[int(ordinal) - 1]

    def raw_stage_values(self) -> dict[str, bool]:
        first, second = self.fill_cup_order
        first_temp, second_temp = self.fill_temperature_order
        first_pick = bool(self.pickup_events_by_cup[first])
        first_fill = first in self.filled_temperature_by_cup
        first_ready = first in self.staged_slot_by_cup
        second_pick = bool(self.pickup_events_by_cup[second])
        second_fill = second in self.filled_temperature_by_cup
        second_ready = second in self.staged_slot_by_cup
        del first_temp, second_temp
        return {
            "first_cup_pick": first_pick,
            "first_cup_fill": first_fill,
            "first_cup_ready": first_ready,
            "second_cup_pick": second_pick,
            "second_cup_fill": second_fill,
            "second_cup_ready": second_ready,
            "selection_success": self.final_taken_cup_id is not None,
        }

    def correct_stage_values(self) -> dict[str, bool]:
        first, second = self.fill_cup_order
        first_temp, second_temp = self.fill_temperature_order
        return {
            "first_cup_pick": bool(self.pickup_events_by_cup[first]),
            "first_cup_fill": self.filled_temperature_by_cup.get(first) == first_temp,
            "first_cup_ready": self.staged_slot_by_cup.get(first)
            == self.service_slot_by_cup[first],
            "second_cup_pick": bool(self.pickup_events_by_cup[second]),
            "second_cup_fill": self.filled_temperature_by_cup.get(second)
            == second_temp,
            "second_cup_ready": self.staged_slot_by_cup.get(second)
            == self.service_slot_by_cup[second],
            "selection_success": self.final_taken_cup_id == self.requested_cup_id,
        }


@dataclass(frozen=True, slots=True)
class StageEvalResult:
    observed: bool
    qualified: bool
    first_observed_step: int | None
    qualified_step: int | None
    invalidated: bool
    reason: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "observed": bool(self.observed),
            "qualified": bool(self.qualified),
            "first_observed_step": self.first_observed_step,
            "qualified_step": self.qualified_step,
            "invalidated": bool(self.invalidated),
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True)
class WA02EvalResult:
    success: bool
    terminal: bool
    failure_code: str | None
    stages: Mapping[str, StageEvalResult]
    details: Mapping[str, Any] = field(default_factory=dict)

    def to_diagnostics(self) -> dict[str, Any]:
        payload = dict(self.details or {})
        stages = {name: value.to_dict() for name, value in self.stages.items()}
        qualified = {name: bool(value["qualified"]) for name, value in stages.items()}
        observed = {name: bool(value["observed"]) for name, value in stages.items()}
        payload.update(
            {
                "success": bool(self.success),
                "terminal": bool(self.terminal),
                "failure_code": self.failure_code,
                "stages": stages,
                "stage_checks": qualified,
                "stage_observed": observed,
                "stage_labels": dict(STAGE_LABELS),
                "stage_first_observed_steps": {
                    name: value["first_observed_step"] for name, value in stages.items()
                },
                "stage_qualified_steps": {
                    name: value["qualified_step"] for name, value in stages.items()
                },
                "success_checks": dict(qualified),
            }
        )
        return payload

    def to_report_stages(self) -> dict[str, Any]:
        """Project the internal protocol onto order-neutral public reporting."""

        stages = {}
        for name in REPORT_STAGE_ORDER:
            internal = self.stages[_REPORT_STAGE_SOURCES[name]]
            passed = bool(self.success) if name == "success" else bool(
                internal.qualified
            )
            stages[name] = {
                "applicable": True,
                "passed": passed,
                "first_pass_step": internal.qualified_step if passed else None,
            }
        return {
            "schema_version": "membench.report_stages/v1",
            "task_id": WA02_REPORT_PROFILE.task_id,
            "profile_id": REPORT_STAGE_PROFILE_ID,
            "order": list(REPORT_STAGE_ORDER),
            "stages": stages,
        }


@dataclass
class _MutableStage:
    observed: bool = False
    qualified: bool = False
    first_observed_step: int | None = None
    qualified_step: int | None = None


class WA02Evaluator:
    """Strict, idempotent, history-aware seven-stage evaluator."""

    def __init__(
        self,
        *,
        fill_cup_order: Sequence[str],
        fill_temperature_order: Sequence[str],
        service_slot_by_cup: Mapping[str, str],
        requested_cup_id: str,
    ) -> None:
        self.fill_cup_order = _order(fill_cup_order)
        self.fill_temperature_order = _temperature_order(fill_temperature_order)
        self.service_slot_by_cup = _slot_by_cup(service_slot_by_cup)
        self.requested_cup_id = _cup(requested_cup_id)
        self._stages = {name: _MutableStage() for name in STAGE_ORDER}
        self._rejected: dict[str, list[dict[str, Any]]] = {name: [] for name in STAGE_ORDER}
        self._previous_raw = {name: False for name in STAGE_ORDER}
        self._last_step_id: int | None = None
        self._cached_result: WA02EvalResult | None = None

    def reset(self, snapshot: WA02EvaluationSnapshot) -> WA02EvalResult:
        self._validate_configuration(snapshot)
        self._stages = {name: _MutableStage() for name in STAGE_ORDER}
        self._rejected = {name: [] for name in STAGE_ORDER}
        raw = snapshot.raw_stage_values()
        for name, value in raw.items():
            if value:
                self._mark_observed(name, snapshot.step_id)
                self._reject(name, snapshot.step_id, "present_at_reset")
        self._previous_raw = raw
        self._last_step_id = snapshot.step_id
        self._cached_result = self._build_result(snapshot)
        return self._cached_result

    def observe(self, snapshot: WA02EvaluationSnapshot) -> WA02EvalResult:
        self._validate_configuration(snapshot)
        if self._last_step_id is None:
            return self.reset(snapshot)
        if snapshot.step_id == self._last_step_id:
            if self._cached_result is None:
                raise RuntimeError("WA-02 evaluator has no cached result")
            return self._cached_result
        if snapshot.step_id < self._last_step_id:
            raise ValueError(
                f"WA-02 evaluation step moved backwards: {snapshot.step_id} < {self._last_step_id}"
            )
        raw = snapshot.raw_stage_values()
        correct = snapshot.correct_stage_values()
        qualified_before = {name: self._stages[name].qualified for name in STAGE_ORDER}
        newly_qualified: set[str] = set()
        for name, value in raw.items():
            if value:
                self._mark_observed(name, snapshot.step_id)
        for name in STAGE_ORDER:
            if not raw[name] or self._previous_raw[name] or self._stages[name].qualified:
                continue
            prerequisite = STAGE_PREREQUISITES[name]
            if prerequisite is not None and not (
                qualified_before[prerequisite] or prerequisite in newly_qualified
            ):
                self._reject(name, snapshot.step_id, f"prerequisite_not_qualified:{prerequisite}")
                continue
            if not correct[name]:
                self._reject(name, snapshot.step_id, self._incorrect_reason(name))
                continue
            self._qualify(name, snapshot.step_id)
            newly_qualified.add(name)
        self._previous_raw = raw
        self._last_step_id = snapshot.step_id
        self._cached_result = self._build_result(snapshot)
        return self._cached_result

    def result(self) -> WA02EvalResult:
        if self._cached_result is None:
            raise RuntimeError("WA-02 evaluator has not been reset")
        return self._cached_result

    def _validate_configuration(self, snapshot: WA02EvaluationSnapshot) -> None:
        if (
            snapshot.fill_cup_order != self.fill_cup_order
            or snapshot.fill_temperature_order != self.fill_temperature_order
            or dict(snapshot.service_slot_by_cup) != self.service_slot_by_cup
            or snapshot.requested_cup_id != self.requested_cup_id
        ):
            raise ValueError("WA-02 evaluation snapshot configuration changed within an episode")

    def _incorrect_reason(self, name: str) -> str:
        if name in {"first_cup_fill", "second_cup_fill"}:
            return "wrong_fill_temperature"
        if name in {"first_cup_ready", "second_cup_ready"}:
            return "wrong_service_slot"
        if name == "selection_success":
            return "wrong_cup_selected"
        return "physical_condition_not_verified"

    def _mark_observed(self, name: str, step_id: int) -> None:
        stage = self._stages[name]
        if not stage.observed:
            stage.observed = True
            stage.first_observed_step = int(step_id)

    def _qualify(self, name: str, step_id: int) -> None:
        stage = self._stages[name]
        if not stage.qualified:
            stage.qualified = True
            stage.qualified_step = int(step_id)

    def _reject(self, name: str, step_id: int, reason: str) -> None:
        entry = {"step": int(step_id), "reason": str(reason)}
        if entry not in self._rejected[name]:
            self._rejected[name].append(entry)

    def _stage_result(self, name: str) -> StageEvalResult:
        stage = self._stages[name]
        rejected = self._rejected[name]
        if stage.qualified:
            reason, invalidated = None, False
        elif rejected:
            reason, invalidated = str(rejected[-1]["reason"]), True
        else:
            prerequisite = STAGE_PREREQUISITES[name]
            reason = (
                f"waiting_for_prerequisite:{prerequisite}"
                if prerequisite is not None and not self._stages[prerequisite].qualified
                else "not_observed"
            )
            invalidated = False
        return StageEvalResult(
            observed=stage.observed,
            qualified=stage.qualified,
            first_observed_step=stage.first_observed_step,
            qualified_step=stage.qualified_step,
            invalidated=invalidated,
            reason=reason,
        )

    def _build_result(self, snapshot: WA02EvaluationSnapshot) -> WA02EvalResult:
        stages = {name: self._stage_result(name) for name in STAGE_ORDER}
        failure_code = None
        final = stages["selection_success"]
        if final.observed and not final.qualified:
            failure_code = final.reason
        else:
            for name in STAGE_ORDER:
                if stages[name].observed and stages[name].invalidated:
                    failure_code = stages[name].reason
                    break
        details = dict(snapshot.details)
        evaluation = dict(details.get("evaluation") or {})
        evaluation.update(
            {
                "profile_id": "wa02-seven-stage-v1",
                "schema_version": 1,
                "step_id": int(snapshot.step_id),
                "stage_order": list(STAGE_ORDER),
                "stage_prerequisites": dict(STAGE_PREREQUISITES),
                "qualification_policy": "rising_edge_after_prerequisite_and_correctness",
                "stage_bindings": {
                    "first_cup_id": self.fill_cup_order[0],
                    "first_temperature": self.fill_temperature_order[0],
                    "second_cup_id": self.fill_cup_order[1],
                    "second_temperature": self.fill_temperature_order[1],
                },
                "rejected_stage_observations": {
                    name: list(values) for name, values in self._rejected.items() if values
                },
            }
        )
        details["evaluation"] = evaluation
        return WA02EvalResult(
            success=stages["selection_success"].qualified,
            terminal=stages["selection_success"].qualified,
            failure_code=failure_code,
            stages=stages,
            details=details,
        )


def evaluate_initialized_runtime(runtime: Any) -> WA02EvalResult:
    initialized = getattr(runtime, "initialized", None) or runtime
    metadata = getattr(initialized, "metadata", {}) or {}
    evaluator = metadata.get(TASK_EVALUATOR_METADATA_KEY)
    env = getattr(initialized, "env", None) or getattr(runtime, "env", None) or initialized
    if evaluator is None:
        evaluator = getattr(env, "_wa02_evaluator", None)
    if not isinstance(evaluator, WA02Evaluator):
        raise RuntimeError("WA-02 runtime has no initialized task evaluator")
    snapshot_builder = getattr(env, "evaluation_snapshot", None)
    if not callable(snapshot_builder):
        raise AttributeError("WA-02 environment has no evaluation_snapshot()")
    return evaluator.observe(snapshot_builder())


def success_for_runtime(runtime: Any, info: Mapping[str, Any] | None = None) -> bool:
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
    "STAGE_LABELS",
    "STAGE_PREREQUISITES",
    "TASK_EVALUATOR_METADATA_KEY",
    "StageEvalResult",
    "WA02EvalResult",
    "WA02EvaluationSnapshot",
    "WA02Evaluator",
    "diagnostics_for_runtime",
    "evaluate_initialized_runtime",
    "report_stages_for_runtime",
    "success_for_runtime",
]
