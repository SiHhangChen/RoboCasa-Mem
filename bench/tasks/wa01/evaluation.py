"""Authoritative WA-01 evaluation state machine.

The simulator owns physical observation and event detection.  This module owns
the evaluation contract: stage order, historical qualification, diagnostics,
and the final verdict.  In particular, a stage that is merely observable is
not necessarily a stage that is qualified for reporting.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from bench.evaluation.report_profiles import WA01_REPORT_PROFILE
from bench.tasks.wa01.domain.contracts import COFFEE_IDS


STAGE_ORDER = ("sugar", "one_ready", "ready", "selection")
STAGE_PREREQUISITES: dict[str, str | None] = {
    "sugar": None,
    "one_ready": "sugar",
    "ready": "one_ready",
    "selection": "ready",
}

TASK_EVALUATOR_METADATA_KEY = "task_evaluator"
REPORT_STAGE_ORDER = WA01_REPORT_PROFILE.stage_ids
REPORT_STAGE_PROFILE_ID = WA01_REPORT_PROFILE.profile_id

_REPORT_STAGE_SOURCES = {
    "sugar_added": "sugar",
    "one_cup_ready": "one_ready",
    "two_cups_ready": "ready",
    "success": "selection",
}


def _coffee_id(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = str(value)
    if normalized not in COFFEE_IDS:
        raise ValueError(f"invalid WA-01 coffee id: {value!r}")
    return normalized


@dataclass(frozen=True, slots=True)
class WA01EvaluationSnapshot:
    """One simulator observation presented to the task evaluator.

    ``sugar_delivery_coffee_id`` and ``selected_coffee_id`` are event outputs
    from the shared physical detector.  They are intentionally not the
    evaluator's state; the evaluator records them only when their event first
    appears after the required predecessor stage.
    """

    step_id: int
    sugar_delivery_coffee_id: str | None
    coffee_on_dining_counter: Mapping[str, bool] = field(default_factory=dict)
    coffee_upright: Mapping[str, bool] = field(default_factory=dict)
    selected_coffee_id: str | None = None
    details: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "step_id", int(self.step_id))
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
                raise ValueError(f"unknown WA-01 coffee ids in {name}: {sorted(unknown)}")
            object.__setattr__(
                self,
                name,
                {coffee_id: bool(values.get(coffee_id, False)) for coffee_id in COFFEE_IDS},
            )
        object.__setattr__(self, "details", dict(self.details or {}))

    @property
    def one_ready(self) -> bool:
        """At least one cup is physically placed and upright."""

        return any(
            self.coffee_on_dining_counter[coffee_id] and self.coffee_upright[coffee_id]
            for coffee_id in COFFEE_IDS
        )

    @property
    def ready(self) -> bool:
        """Both cups are physically placed and upright."""

        return all(
            self.coffee_on_dining_counter[coffee_id] and self.coffee_upright[coffee_id]
            for coffee_id in COFFEE_IDS
        )

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
            "ready": self.ready,
            "selection": self.selected_coffee_id is not None,
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
class WA01EvalResult:
    """Immutable verdict returned by the WA-01 evaluator."""

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
        selection["observed_correct"] = bool(selection.get("correct", False))
        selection["correct"] = qualified["selection"]
        coffee_pair = dict(payload.get("coffee_pair") or {})
        coffee_pair["detected_together_before_selection"] = coffee_pair.get(
            "together_before_selection",
            False,
        )
        coffee_pair["together_before_selection"] = qualified["ready"]

        payload.update(
            {
                "success": bool(self.success),
                "terminal": bool(self.terminal),
                "failure_code": self.failure_code,
                "success_checks": {
                    "sugar_consumed": qualified["sugar"],
                    "cups_together_before_selection": qualified["ready"],
                    "correct_sugared_selection": qualified["selection"],
                },
                "sugar": sugar,
                "selection": selection,
                "coffee_pair": coffee_pair,
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
        return payload

    def to_report_stages(self) -> dict[str, Any]:
        """Project the detailed evaluator onto the public WA-01 funnel."""

        valid_sugar = bool(self.stages["sugar"].qualified)
        passed = {
            "sugar_added": valid_sugar,
            "one_cup_ready": bool(
                valid_sugar and self.stages["one_ready"].qualified
            ),
            "two_cups_ready": bool(
                valid_sugar and self.stages["ready"].qualified
            ),
            "success": bool(self.success),
        }
        stages = {}
        for name in REPORT_STAGE_ORDER:
            internal = self.stages[_REPORT_STAGE_SOURCES[name]]
            stages[name] = {
                "applicable": True,
                "passed": passed[name],
                "first_pass_step": internal.qualified_step if passed[name] else None,
            }
        return {
            "schema_version": "membench.report_stages/v1",
            "task_id": WA01_REPORT_PROFILE.task_id,
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


class WA01Evaluator:
    """Strict, idempotent, history-aware WA-01 stage evaluator."""

    def __init__(self) -> None:
        self._stages = {name: _MutableStage() for name in STAGE_ORDER}
        self._rejected: dict[str, list[dict[str, Any]]] = {name: [] for name in STAGE_ORDER}
        self._previous_raw: dict[str, bool] = {name: False for name in STAGE_ORDER}
        self._last_step_id: int | None = None
        self._cached_result: WA01EvalResult | None = None

    def reset(self, snapshot: WA01EvaluationSnapshot) -> WA01EvalResult:
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

    def observe(self, snapshot: WA01EvaluationSnapshot) -> WA01EvalResult:
        if self._last_step_id is None:
            return self.reset(snapshot)
        if snapshot.step_id == self._last_step_id:
            if self._cached_result is None:
                raise RuntimeError("WA-01 evaluator has no cached result")
            return self._cached_result
        if snapshot.step_id < self._last_step_id:
            raise ValueError(
                f"WA-01 evaluation step moved backwards: {snapshot.step_id} < {self._last_step_id}"
            )

        raw = snapshot.raw_stage_values()
        qualified_before = {name: self._stages[name].qualified for name in STAGE_ORDER}
        for name, value in raw.items():
            if value:
                self._mark_observed(name, snapshot.step_id)

        for name in STAGE_ORDER:
            if not raw[name] or self._previous_raw[name] or self._stages[name].qualified:
                continue
            prerequisite = STAGE_PREREQUISITES[name]
            if prerequisite is not None and not qualified_before[prerequisite]:
                self._reject(
                    name,
                    snapshot.step_id,
                    f"prerequisite_not_qualified:{prerequisite}",
                )
            elif name == "selection" and not snapshot.correct_selection:
                self._reject(name, snapshot.step_id, "wrong_coffee_selected")
            else:
                self._qualify(name, snapshot.step_id)

        self._previous_raw = raw
        self._last_step_id = snapshot.step_id
        self._cached_result = self._build_result(snapshot)
        return self._cached_result

    def result(self) -> WA01EvalResult:
        if self._cached_result is None:
            raise RuntimeError("WA-01 evaluator has not been reset")
        return self._cached_result

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

    def _build_result(self, snapshot: WA01EvaluationSnapshot) -> WA01EvalResult:
        stages = {name: self._stage_result(name) for name in STAGE_ORDER}
        failure_code: str | None = None
        selection = stages["selection"]
        if selection.observed and not selection.qualified:
            failure_code = (
                "wrong_coffee_selected"
                if selection.reason == "wrong_coffee_selected"
                else "selection_before_ready"
            )

        details = dict(snapshot.details)
        evaluation = dict(details.get("evaluation") or {})
        evaluation.update(
            {
                "schema_version": 1,
                "step_id": int(snapshot.step_id),
                "stage_order": list(STAGE_ORDER),
                "stage_prerequisites": dict(STAGE_PREREQUISITES),
                "qualification_policy": "rising_edge_after_prerequisite",
                "same_step_derived_stages": [],
                "raw_stage_checks": snapshot.raw_stage_values(),
                "rejected_stage_observations": {
                    name: list(values) for name, values in self._rejected.items() if values
                },
            }
        )
        details["evaluation"] = evaluation
        return WA01EvalResult(
            success=selection.qualified,
            terminal=selection.qualified,
            failure_code=failure_code,
            stages=stages,
            details=details,
        )


def evaluate_initialized_runtime(runtime: Any) -> WA01EvalResult:
    """Evaluate one initialized task runtime through its task-local evaluator."""

    initialized = getattr(runtime, "initialized", None) or runtime
    metadata = getattr(initialized, "metadata", {}) or {}
    evaluator = metadata.get(TASK_EVALUATOR_METADATA_KEY)
    env = getattr(initialized, "env", None) or getattr(runtime, "env", None) or initialized
    if evaluator is None:
        evaluator = getattr(env, "_wa01_evaluator", None)
    if not isinstance(evaluator, WA01Evaluator):
        raise RuntimeError("WA-01 runtime has no initialized task evaluator")
    snapshot_builder = getattr(env, "evaluation_snapshot", None)
    if not callable(snapshot_builder):
        raise AttributeError("WA-01 environment has no evaluation_snapshot()")
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
    "STAGE_PREREQUISITES",
    "TASK_EVALUATOR_METADATA_KEY",
    "StageEvalResult",
    "WA01EvalResult",
    "WA01EvaluationSnapshot",
    "WA01Evaluator",
    "diagnostics_for_runtime",
    "evaluate_initialized_runtime",
    "report_stages_for_runtime",
    "success_for_runtime",
]
