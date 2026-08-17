"""Authoritative WA-03 evaluation state machine.

The simulator owns physical observation and event detection.  This module owns
the evaluation contract: stage order, historical qualification, diagnostics,
and the final verdict.  In particular, a stage that is merely observable is
not necessarily a stage that is qualified for reporting.

WA-03 extends WA-01 to three identical coffee cups: the robot sugars any one
cup, carries all three cups into parallel dining regions (left/center/right,
one cup each), and must finally select the cup it sugared.  The ``ready``
stage therefore requires not only the cups to be on the dining counter and
upright but also that each of the three dining regions is occupied by exactly
one distinct cup.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from bench.tasks.wa03.domain.contracts import COFFEE_IDS


STAGE_ORDER = ("sugar", "one_ready", "two_ready", "ready", "selection", "correct", "success")
STAGE_PREREQUISITES: dict[str, str | None] = {
    "sugar": None,
    "one_ready": "sugar",
    "two_ready": "one_ready",
    "ready": "two_ready",
    "selection": "ready",
    "correct": "selection",
    "success": "correct",
}
_PHYSICAL_STAGES = ("sugar", "one_ready", "two_ready", "ready", "selection")
_DERIVED_STAGES = ("correct", "success")

TASK_EVALUATOR_METADATA_KEY = "task_evaluator"


def _coffee_id(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = str(value)
    if normalized not in COFFEE_IDS:
        raise ValueError(f"invalid WA-03 coffee id: {value!r}")
    return normalized


def _configuration(mode: str, target: str | None) -> tuple[str, str | None]:
    normalized_mode = str(mode).strip().lower()
    if normalized_mode not in {"unmarked", "marked"}:
        raise ValueError(f"unsupported WA-03 sugar target mode: {mode!r}")
    normalized_target = _coffee_id(target)
    if normalized_mode == "marked" and normalized_target is None:
        raise ValueError("marked WA-03 evaluation requires a target coffee")
    if normalized_mode == "unmarked" and normalized_target is not None:
        raise ValueError("unmarked WA-03 evaluation cannot define a target coffee")
    return normalized_mode, normalized_target


@dataclass(frozen=True, slots=True)
class WA03EvaluationSnapshot:
    """One simulator observation presented to the task evaluator.

    ``sugar_delivery_coffee_id`` and ``selected_coffee_id`` are event outputs
    from the shared physical detector.  They are intentionally not the
    evaluator's state; the evaluator records them only when their event first
    appears after the required predecessor stage.

    ``coffee_in_dining_region`` maps each coffee to the one dining slot whose
    region currently contains that cup's center.  A cup in no region maps to
    None; the ``ready`` stage requires every coffee to occupy its own slot.
    """

    step_id: int
    sugar_target_mode: str
    target_sugared_coffee_id: str | None
    sugar_delivery_coffee_id: str | None
    coffee_on_dining_counter: Mapping[str, bool] = field(default_factory=dict)
    coffee_upright: Mapping[str, bool] = field(default_factory=dict)
    coffee_in_dining_region: Mapping[str, str | None] = field(default_factory=dict)
    selected_coffee_id: str | None = None
    details: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        mode, target = _configuration(
            self.sugar_target_mode,
            self.target_sugared_coffee_id,
        )
        object.__setattr__(self, "step_id", int(self.step_id))
        object.__setattr__(self, "sugar_target_mode", mode)
        object.__setattr__(self, "target_sugared_coffee_id", target)
        object.__setattr__(
            self,
            "sugar_delivery_coffee_id",
            _coffee_id(self.sugar_delivery_coffee_id),
        )
        object.__setattr__(self, "selected_coffee_id", _coffee_id(self.selected_coffee_id))
        for name in ("coffee_on_dining_counter", "coffee_upright"):
            values = getattr(self, name)
            unknown = set(values) - set(COFFEE_IDS)
            if unknown:
                raise ValueError(f"unknown WA-03 coffee ids in {name}: {sorted(unknown)}")
            object.__setattr__(
                self,
                name,
                {coffee_id: bool(values.get(coffee_id, False)) for coffee_id in COFFEE_IDS},
            )
        regions = dict(self.coffee_in_dining_region)
        unknown_regions = set(regions) - set(COFFEE_IDS)
        if unknown_regions:
            raise ValueError(f"unknown WA-03 coffee ids in coffee_in_dining_region: {sorted(unknown_regions)}")
        normalized_regions: dict[str, str | None] = {}
        for coffee_id in COFFEE_IDS:
            slot = regions.get(coffee_id)
            normalized_regions[coffee_id] = None if slot is None else str(slot)
        object.__setattr__(self, "coffee_in_dining_region", normalized_regions)
        object.__setattr__(self, "details", dict(self.details or {}))

    @property
    def one_ready(self) -> bool:
        """At least one cup is physically placed and upright."""

        return any(
            self.coffee_on_dining_counter[coffee_id] and self.coffee_upright[coffee_id]
            for coffee_id in COFFEE_IDS
        )

    @property
    def two_ready(self) -> bool:
        """At least two cups are physically placed and upright."""

        return sum(
            1
            for coffee_id in COFFEE_IDS
            if self.coffee_on_dining_counter[coffee_id] and self.coffee_upright[coffee_id]
        ) >= 2

    @property
    def ready(self) -> bool:
        """All three cups are placed, upright, and each in its own dining region."""

        if not all(
            self.coffee_on_dining_counter[coffee_id] and self.coffee_upright[coffee_id]
            for coffee_id in COFFEE_IDS
        ):
            return False
        slots = [self.coffee_in_dining_region[coffee_id] for coffee_id in COFFEE_IDS]
        occupied = [slot for slot in slots if slot is not None]
        return len(occupied) == len(COFFEE_IDS) and len(set(occupied)) == len(occupied)

    @property
    def sugar_assignment_correct(self) -> bool | None:
        if self.sugar_target_mode == "unmarked":
            return None
        return self.sugar_delivery_coffee_id == self.target_sugared_coffee_id

    @property
    def correct_selection(self) -> bool:
        return bool(
            self.selected_coffee_id is not None
            and self.selected_coffee_id == self.sugar_delivery_coffee_id
        )

    def raw_stage_values(self) -> dict[str, bool]:
        """Return current detector values before prerequisite qualification."""

        return {
            "sugar": self.sugar_delivery_coffee_id is not None,
            "one_ready": self.one_ready,
            "two_ready": self.two_ready,
            "ready": self.ready,
            "selection": self.selected_coffee_id is not None,
            "correct": self.correct_selection,
            "success": bool(
                self.correct_selection and self.sugar_assignment_correct is not False
            ),
        }


@dataclass(frozen=True, slots=True)
class StageEvalResult:
    """Observed versus prerequisite-qualified state for one stage."""

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
class WA03EvalResult:
    """Immutable verdict returned by the WA-03 evaluator."""

    success: bool
    terminal: bool
    failure_code: str | None
    stages: Mapping[str, StageEvalResult]
    details: Mapping[str, Any] = field(default_factory=dict)

    def to_diagnostics(self) -> dict[str, Any]:
        """Build the stable result payload consumed by rollout artifacts."""

        payload = dict(self.details or {})
        stages = {name: result.to_dict() for name, result in self.stages.items()}
        qualified = {name: bool(values["qualified"]) for name, values in stages.items()}
        observed = {name: bool(values["observed"]) for name, values in stages.items()}

        sugar = dict(payload.get("sugar") or {})
        sugar["consumed"] = qualified["sugar"]
        selection = dict(payload.get("selection") or {})
        selection["observed_correct"] = observed["correct"]
        selection["correct"] = qualified["correct"]
        coffee_set = dict(payload.get("coffee_set") or {})
        coffee_set["detected_together_before_selection"] = coffee_set.get(
            "together_before_selection",
            False,
        )
        coffee_set["together_before_selection"] = qualified["ready"]

        payload.update(
            {
                "success": bool(self.success),
                "terminal": bool(self.terminal),
                "failure_code": self.failure_code,
                "success_checks": {
                    "sugar_consumed": qualified["sugar"],
                    "cups_together_before_selection": qualified["ready"],
                    "correct_sugared_selection": qualified["correct"],
                },
                "sugar": sugar,
                "selection": selection,
                "coffee_set": coffee_set,
                "stages": stages,
                "stage_checks": qualified,
                "stage_observed": observed,
                "stage_first_observed_steps": {
                    name: values["first_observed_step"] for name, values in stages.items()
                },
                "stage_qualified_steps": {
                    name: values["qualified_step"] for name, values in stages.items()
                },
            }
        )
        assignment = sugar.get("assignment_correct")
        if assignment is not None:
            payload["success_checks"]["sugar_assignment_correct"] = bool(assignment)
        return payload


@dataclass
class _MutableStage:
    observed: bool = False
    qualified: bool = False
    first_observed_step: int | None = None
    qualified_step: int | None = None


class WA03Evaluator:
    """Strict, idempotent, history-aware WA-03 stage evaluator."""

    def __init__(self, *, sugar_target_mode: str, target_sugared_coffee_id: str | None) -> None:
        self.sugar_target_mode, self.target_sugared_coffee_id = _configuration(
            sugar_target_mode,
            target_sugared_coffee_id,
        )
        self._stages = {name: _MutableStage() for name in STAGE_ORDER}
        self._rejected: dict[str, list[dict[str, Any]]] = {name: [] for name in STAGE_ORDER}
        self._previous_raw: dict[str, bool] = {name: False for name in STAGE_ORDER}
        self._last_step_id: int | None = None
        self._cached_result: WA03EvalResult | None = None

    def reset(self, snapshot: WA03EvaluationSnapshot) -> WA03EvalResult:
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

    def observe(self, snapshot: WA03EvaluationSnapshot) -> WA03EvalResult:
        self._validate_configuration(snapshot)
        if self._last_step_id is None:
            return self.reset(snapshot)
        if snapshot.step_id == self._last_step_id:
            if self._cached_result is None:
                raise RuntimeError("WA-03 evaluator has no cached result")
            return self._cached_result
        if snapshot.step_id < self._last_step_id:
            raise ValueError(
                f"WA-03 evaluation step moved backwards: {snapshot.step_id} < {self._last_step_id}"
            )

        raw = snapshot.raw_stage_values()
        qualified_before = {name: self._stages[name].qualified for name in STAGE_ORDER}
        newly_qualified: set[str] = set()

        for name, value in raw.items():
            if value:
                self._mark_observed(name, snapshot.step_id)

        for name in _PHYSICAL_STAGES:
            if not raw[name] or self._previous_raw[name] or self._stages[name].qualified:
                continue
            prerequisite = STAGE_PREREQUISITES[name]
            if prerequisite is None or qualified_before[prerequisite]:
                self._qualify(name, snapshot.step_id)
                newly_qualified.add(name)
            else:
                self._reject(
                    name,
                    snapshot.step_id,
                    f"prerequisite_not_qualified:{prerequisite}",
                )

        # Correctness and final success are derived from a valid selection and
        # may therefore be committed in the same control step as that event.
        if raw["correct"] and not self._stages["correct"].qualified:
            if qualified_before["selection"] or "selection" in newly_qualified:
                self._qualify("correct", snapshot.step_id)
                newly_qualified.add("correct")
            elif not self._previous_raw["correct"]:
                self._reject("correct", snapshot.step_id, "prerequisite_not_qualified:selection")

        if raw["success"] and not self._stages["success"].qualified:
            if qualified_before["correct"] or "correct" in newly_qualified:
                self._qualify("success", snapshot.step_id)
                newly_qualified.add("success")
            elif not self._previous_raw["success"]:
                self._reject("success", snapshot.step_id, "prerequisite_not_qualified:correct")

        self._previous_raw = raw
        self._last_step_id = snapshot.step_id
        self._cached_result = self._build_result(snapshot)
        return self._cached_result

    def result(self) -> WA03EvalResult:
        if self._cached_result is None:
            raise RuntimeError("WA-03 evaluator has not been reset")
        return self._cached_result

    def _validate_configuration(self, snapshot: WA03EvaluationSnapshot) -> None:
        if (
            snapshot.sugar_target_mode != self.sugar_target_mode
            or snapshot.target_sugared_coffee_id != self.target_sugared_coffee_id
        ):
            raise ValueError("WA-03 evaluation snapshot configuration changed within an episode")

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
        entries = self._rejected[name]
        entry = {"step": int(step_id), "reason": str(reason)}
        if entry not in entries:
            entries.append(entry)

    def _stage_result(self, name: str) -> StageEvalResult:
        stage = self._stages[name]
        rejected = self._rejected[name]
        if stage.qualified:
            reason = None
            invalidated = False
        elif rejected:
            reason = str(rejected[-1]["reason"])
            invalidated = True
        else:
            prerequisite = STAGE_PREREQUISITES[name]
            if prerequisite is not None and not self._stages[prerequisite].qualified:
                reason = f"waiting_for_prerequisite:{prerequisite}"
                invalidated = False
            else:
                reason = "not_observed"
                invalidated = False
        return StageEvalResult(
            observed=stage.observed,
            qualified=stage.qualified,
            first_observed_step=stage.first_observed_step,
            qualified_step=stage.qualified_step,
            invalidated=invalidated,
            reason=reason,
        )

    def _build_result(self, snapshot: WA03EvaluationSnapshot) -> WA03EvalResult:
        stages = {name: self._stage_result(name) for name in STAGE_ORDER}
        assignment = snapshot.sugar_assignment_correct
        failure_code: str | None = None
        if stages["sugar"].qualified and assignment is False:
            failure_code = "wrong_sugar_assignment"
        elif (
            stages["selection"].qualified
            and snapshot.selected_coffee_id != snapshot.sugar_delivery_coffee_id
        ):
            failure_code = "wrong_coffee_selected"
        elif stages["selection"].observed and not stages["selection"].qualified:
            failure_code = "selection_before_ready"

        details = dict(snapshot.details)
        evaluation = dict(details.get("evaluation") or {})
        evaluation.update(
            {
                "schema_version": 1,
                "step_id": int(snapshot.step_id),
                "stage_order": list(STAGE_ORDER),
                "stage_prerequisites": dict(STAGE_PREREQUISITES),
                "qualification_policy": "rising_edge_after_prerequisite",
                "same_step_derived_stages": list(_DERIVED_STAGES),
                "raw_stage_checks": snapshot.raw_stage_values(),
                "rejected_stage_observations": {
                    name: list(values) for name, values in self._rejected.items() if values
                },
            }
        )
        details["evaluation"] = evaluation
        if "sugar" in details:
            sugar = dict(details["sugar"] or {})
            sugar["assignment_correct"] = assignment
            details["sugar"] = sugar
        return WA03EvalResult(
            success=stages["success"].qualified,
            terminal=stages["success"].qualified,
            failure_code=failure_code,
            stages=stages,
            details=details,
        )


def evaluate_initialized_runtime(runtime: Any) -> WA03EvalResult:
    """Evaluate one initialized task runtime through its task-local evaluator."""

    initialized = getattr(runtime, "initialized", None) or runtime
    metadata = getattr(initialized, "metadata", {}) or {}
    evaluator = metadata.get(TASK_EVALUATOR_METADATA_KEY)
    env = getattr(initialized, "env", None) or getattr(runtime, "env", None) or initialized
    if evaluator is None:
        evaluator = getattr(env, "_wa03_evaluator", None)
    if not isinstance(evaluator, WA03Evaluator):
        raise RuntimeError("WA-03 runtime has no initialized task evaluator")
    snapshot_builder = getattr(env, "evaluation_snapshot", None)
    if not callable(snapshot_builder):
        raise AttributeError("WA-03 environment has no evaluation_snapshot()")
    return evaluator.observe(snapshot_builder())


def success_for_runtime(runtime: Any, info: Mapping[str, Any] | None = None) -> bool:
    del info
    return bool(evaluate_initialized_runtime(runtime).success)


def diagnostics_for_runtime(runtime: Any) -> Mapping[str, Any]:
    return evaluate_initialized_runtime(runtime).to_diagnostics()


__all__ = [
    "STAGE_ORDER",
    "STAGE_PREREQUISITES",
    "TASK_EVALUATOR_METADATA_KEY",
    "StageEvalResult",
    "WA03EvalResult",
    "WA03EvaluationSnapshot",
    "WA03Evaluator",
    "diagnostics_for_runtime",
    "evaluate_initialized_runtime",
    "success_for_runtime",
]
