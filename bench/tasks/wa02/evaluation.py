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


def _temperature_order(values: Sequence[Any]) -> tuple[str, str]:
    if isinstance(values, (str, bytes)) or len(values) != 2:
        raise ValueError("WA-02 fill_temperature_order must contain two temperatures")
    result = tuple(_temperature(value) for value in values)
    if set(result) != set(TEMPERATURES):
        raise ValueError(
            "WA-02 fill_temperature_order must contain hot and cold exactly once"
        )
    return result  # type: ignore[return-value]

@dataclass(frozen=True, slots=True)
class WA02EvaluationSnapshot:
    """Physical and semantic facts sampled at one environment step."""

    step_id: int
    requested_temperature: str
    # Optional recording metadata. Evaluation binds temperature from physical
    # fill events and does not require this demonstration order.
    fill_temperature_order: tuple[str, str] | None = None
    pickup_step_by_cup: Mapping[str, int] = field(default_factory=dict)
    filled_temperature_by_cup: Mapping[str, str] = field(default_factory=dict)
    staged_slot_by_cup: Mapping[str, str] = field(default_factory=dict)
    final_taken_cup_id: str | None = None
    details: Mapping[str, Any] = field(default_factory=dict)
    # Policy evaluation intentionally mirrors WA-01: readiness is derived
    # from the physical counter-contact and upright predicates only.  The
    # service-slot map above remains recording metadata and is not required
    # when these physical observations are supplied.
    cup_on_counter: Mapping[str, bool] = field(default_factory=dict)
    cup_upright: Mapping[str, bool] = field(default_factory=dict)
    physically_ready_by_cup: Mapping[str, bool] = field(default_factory=dict)

    _physical_readiness_provided: bool = field(
        init=False, repr=False, compare=False
    )
    _physical_ready_map_provided: bool = field(
        init=False, repr=False, compare=False
    )
    _counter_upright_provided: bool = field(
        init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "step_id", int(self.step_id))
        if self.step_id < 0:
            raise ValueError("WA-02 evaluation step_id must be non-negative")
        object.__setattr__(
            self, "requested_temperature", _temperature(self.requested_temperature)
        )
        if self.fill_temperature_order is not None:
            object.__setattr__(
                self,
                "fill_temperature_order",
                _temperature_order(self.fill_temperature_order),
            )
        pickup = {
            _cup(cup): int(value) for cup, value in self.pickup_step_by_cup.items()
        }
        if any(step < 0 for step in pickup.values()):
            raise ValueError("WA-02 pickup steps must be non-negative")
        object.__setattr__(
            self,
            "pickup_step_by_cup",
            pickup,
        )
        fills = {
            _cup(cup): _temperature(temp)
            for cup, temp in self.filled_temperature_by_cup.items()
        }
        object.__setattr__(self, "filled_temperature_by_cup", fills)
        physical_ready_map_provided = bool(self.physically_ready_by_cup)
        counter_upright_provided = bool(self.cup_on_counter or self.cup_upright)
        readiness_provided = bool(
            counter_upright_provided or physical_ready_map_provided
        )
        staged = {
            _cup(cup): str(slot) for cup, slot in self.staged_slot_by_cup.items()
        }
        # ``staged_slot_by_cup`` is recording metadata.  Do not reject a policy
        # snapshot because an old recorder emitted a duplicate or unknown slot;
        # readiness is sourced from the physical predicates below.  Preserve
        # validation for slot-only legacy snapshots, where this map is the only
        # available readiness source.  The expert recording handler performs
        # strict slot validation before it commits live metadata.
        if not readiness_provided and any(
            slot not in SERVICE_SLOTS for slot in staged.values()
        ):
            raise ValueError("invalid WA-02 service slot in staged_slot_by_cup")
        object.__setattr__(self, "staged_slot_by_cup", staged)
        for name in ("cup_on_counter", "cup_upright", "physically_ready_by_cup"):
            values = getattr(self, name)
            unknown = set(values) - set(CUP_IDS)
            if unknown:
                raise ValueError(
                    f"unknown WA-02 cup ids in {name}: {sorted(unknown)}"
                )
            object.__setattr__(
                self,
                name,
                {cup: bool(values.get(cup, False)) for cup in CUP_IDS},
            )
        object.__setattr__(self, "_physical_readiness_provided", readiness_provided)
        object.__setattr__(
            self, "_physical_ready_map_provided", physical_ready_map_provided
        )
        object.__setattr__(
            self, "_counter_upright_provided", counter_upright_provided
        )
        final = (
            None if self.final_taken_cup_id is None else _cup(self.final_taken_cup_id)
        )
        object.__setattr__(self, "final_taken_cup_id", final)
        object.__setattr__(self, "details", dict(self.details or {}))

    @property
    def has_physical_readiness(self) -> bool:
        """Whether this snapshot carries the slot-independent ready signal."""

        return bool(self._physical_readiness_provided)

    @property
    def ready_by_cup(self) -> dict[str, bool]:
        """Return one physical readiness bit per cup.

        New runtime snapshots use ``counter contact + upright``.  A composed
        ``physically_ready_by_cup`` map is accepted as an equivalent physical
        observation.  Slot-only snapshots are retained solely for compatibility
        with old serialized artifacts; the live environment always supplies
        physical readiness and therefore never uses this fallback.
        """

        if self._counter_upright_provided:
            return {
                cup: bool(self.cup_on_counter[cup] and self.cup_upright[cup])
                for cup in CUP_IDS
            }
        if self._physical_ready_map_provided:
            return dict(self.physically_ready_by_cup)
        return {cup: cup in self.staged_slot_by_cup for cup in CUP_IDS}

    def cup_for_temperature(self, temperature: str) -> str | None:
        temp = _temperature(temperature)
        matches = [
            cup
            for cup, recorded in self.filled_temperature_by_cup.items()
            if recorded == temp
        ]
        return matches[0] if len(matches) == 1 else None


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
        service = dict(payload.get("service") or {})
        service.setdefault("placement_rule", "counter_contact_and_upright")
        service["evaluation_readiness_policy"] = "counter_contact_and_upright"
        service["service_slot_assignment_required_for_evaluation"] = False
        service["evaluation_ready"] = bool(qualified.get("second_cup_ready", False))
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
                "service": service,
            }
        )
        return payload

    def to_report_stages(self) -> dict[str, Any]:
        """Project the physical first/second funnel onto the public report."""

        stages = {}
        for name in REPORT_STAGE_ORDER:
            internal = self.stages[_REPORT_STAGE_SOURCES[name]]
            passed = (
                bool(self.success)
                if name == "success"
                else bool(internal.qualified)
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
        requested_temperature: str,
        fill_temperature_order: Sequence[str] | None = None,
    ) -> None:
        self.fill_temperature_order = (
            None
            if fill_temperature_order is None
            else _temperature_order(fill_temperature_order)
        )
        self.requested_temperature = _temperature(requested_temperature)
        self.first_cup_id: str | None = None
        self.second_cup_id: str | None = None
        self._stages = {name: _MutableStage() for name in STAGE_ORDER}
        self._rejected: dict[str, list[dict[str, Any]]] = {name: [] for name in STAGE_ORDER}
        self._previous_raw = {name: False for name in STAGE_ORDER}
        self._last_step_id: int | None = None
        self._cached_result: WA02EvalResult | None = None

    def reset(self, snapshot: WA02EvaluationSnapshot) -> WA02EvalResult:
        self._validate_configuration(snapshot)
        self.first_cup_id = None
        self.second_cup_id = None
        self._stages = {name: _MutableStage() for name in STAGE_ORDER}
        self._rejected = {name: [] for name in STAGE_ORDER}
        self._bind_cup_order(snapshot)
        raw, _ = self._stage_values(snapshot)
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
        self._bind_cup_order(snapshot)
        raw, correct = self._stage_values(snapshot)
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
                self._reject(
                    name,
                    snapshot.step_id,
                    self._incorrect_reason(
                        name, legacy_staging=not snapshot.has_physical_readiness
                    ),
                )
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
        if snapshot.requested_temperature != self.requested_temperature:
            raise ValueError("WA-02 evaluation snapshot configuration changed within an episode")

    def _bind_cup_order(self, snapshot: WA02EvaluationSnapshot) -> None:
        if self.first_cup_id is not None:
            return
        pickup_steps = dict(snapshot.pickup_step_by_cup)
        if not pickup_steps:
            return
        earliest = min(pickup_steps.values())
        first_candidates = [
            cup for cup, step in pickup_steps.items() if int(step) == earliest
        ]
        if len(first_candidates) != 1:
            self._mark_observed("first_cup_pick", snapshot.step_id)
            self._reject("first_cup_pick", snapshot.step_id, "ambiguous_pickup_order")
            return
        first = first_candidates[0]
        self.first_cup_id = first
        self.second_cup_id = next(cup for cup in CUP_IDS if cup != first)

    def _stage_values(
        self, snapshot: WA02EvaluationSnapshot
    ) -> tuple[dict[str, bool], dict[str, bool]]:
        first = self.first_cup_id
        second = self.second_cup_id
        first_temp = snapshot.filled_temperature_by_cup.get(first or "")
        second_temp = snapshot.filled_temperature_by_cup.get(second or "")
        staged = dict(snapshot.staged_slot_by_cup)
        ready_by_cup = snapshot.ready_by_cup
        first_ready = bool(first is not None and ready_by_cup.get(first, False))
        both_cups_ready = bool(
            first is not None
            and second is not None
            and ready_by_cup.get(first, False)
            and ready_by_cup.get(second, False)
        )
        second_ready_observed = both_cups_ready
        # Old snapshot fixtures only carried staged_slot_by_cup.  Keep their
        # historical validation semantics without making live evaluation
        # depend on slot identity.
        if not snapshot.has_physical_readiness:
            both_slots_used = bool(
                first is not None
                and second is not None
                and first in staged
                and second in staged
                and set(staged.values()) == set(SERVICE_SLOTS)
            )
            first_ready = bool(first is not None and first in staged)
            second_ready_observed = bool(second is not None and second in staged)
            both_cups_ready = both_slots_used
        requested_cup = snapshot.cup_for_temperature(self.requested_temperature)
        raw = {
            "first_cup_pick": first is not None and first in snapshot.pickup_step_by_cup,
            "first_cup_fill": first is not None and first in snapshot.filled_temperature_by_cup,
            "first_cup_ready": first_ready,
            "second_cup_pick": second is not None and second in snapshot.pickup_step_by_cup,
            "second_cup_fill": second is not None and second in snapshot.filled_temperature_by_cup,
            "second_cup_ready": second_ready_observed,
            "selection_success": snapshot.final_taken_cup_id is not None,
        }
        correct = {
            "first_cup_pick": raw["first_cup_pick"],
            "first_cup_fill": raw["first_cup_fill"],
            "first_cup_ready": first_ready,
            "second_cup_pick": raw["second_cup_pick"],
            "second_cup_fill": bool(
                raw["second_cup_fill"]
                and first_temp is not None
                and second_temp is not None
                and first_temp != second_temp
            ),
            "second_cup_ready": both_cups_ready,
            "selection_success": bool(
                requested_cup is not None
                and snapshot.final_taken_cup_id == requested_cup
            ),
        }
        return raw, correct

    def _incorrect_reason(self, name: str, *, legacy_staging: bool = False) -> str:
        if name == "second_cup_fill":
            return "duplicate_fill_temperature"
        if name == "first_cup_fill":
            return "invalid_fill_temperature"
        if name in {"first_cup_ready", "second_cup_ready"}:
            return (
                "invalid_service_staging"
                if legacy_staging
                else "counter_contact_and_upright_not_verified"
            )
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
                "profile_id": "wa02-seven-stage-v3",
                "schema_version": 3,
                "step_id": int(snapshot.step_id),
                "stage_order": list(STAGE_ORDER),
                "stage_prerequisites": dict(STAGE_PREREQUISITES),
                "qualification_policy": (
                    "rising_edge_after_prerequisite_and_dynamic_temperature_binding"
                ),
                "readiness_policy": "counter_contact_and_upright",
                "service_slot_assignment_required_for_evaluation": False,
                "physical_readiness_by_cup": snapshot.ready_by_cup,
                "stage_bindings": {
                    "binding_source": "physical_events",
                    "first_cup_id": self.first_cup_id,
                    "first_temperature": snapshot.filled_temperature_by_cup.get(
                        self.first_cup_id or ""
                    ),
                    "second_cup_id": self.second_cup_id,
                    "second_temperature": snapshot.filled_temperature_by_cup.get(
                        self.second_cup_id or ""
                    ),
                    "requested_temperature": self.requested_temperature,
                    "requested_cup_id": snapshot.cup_for_temperature(
                        self.requested_temperature
                    ),
                    # Retained as recording metadata only.  It is deliberately
                    # not consulted by ``_stage_values``.
                    "service_slot_by_cup": dict(snapshot.staged_slot_by_cup),
                },
                "demonstration_temperature_order": (
                    None
                    if self.fill_temperature_order is None
                    else list(self.fill_temperature_order)
                ),
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
