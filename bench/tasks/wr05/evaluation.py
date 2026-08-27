"""Authoritative WR-05 evaluation state machine.

The environment supplies historical plate-staging facts and current physical
return facts. This module owns temporal qualification and the stable public
benchmark report stages.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from bench.evaluation.report_profiles import WR05_REPORT_PROFILE
from bench.tasks.wr05.domain.contracts import CONDIMENT_IDS


STAGE_ORDER = WR05_REPORT_PROFILE.stage_ids
STAGE_PREREQUISITES: dict[str, str | None] = {
    "one_item_staged": None,
    "two_items_staged": "one_item_staged",
    "one_item_correctly_returned": "two_items_staged",
    "success": "one_item_correctly_returned",
}
REPORT_STAGE_ORDER = STAGE_ORDER
REPORT_STAGE_PROFILE_ID = WR05_REPORT_PROFILE.profile_id
TASK_EVALUATOR_METADATA_KEY = "task_evaluator"


def _flags(values: Mapping[str, Any], *, field_name: str) -> dict[str, bool]:
    raw = dict(values or {})
    unknown = sorted(set(raw) - set(CONDIMENT_IDS))
    if unknown:
        raise ValueError(f"unknown WR-05 condiment ids in {field_name}: {unknown}")
    return {
        condiment_id: bool(raw.get(condiment_id, False))
        for condiment_id in CONDIMENT_IDS
    }


@dataclass(frozen=True, slots=True)
class WR05EvaluationSnapshot:
    """One read-only semantic and physical observation."""

    step_id: int
    staged_at_plate: Mapping[str, bool] = field(default_factory=dict)
    home_correct_by_condiment: Mapping[str, bool] = field(default_factory=dict)
    released_by_condiment: Mapping[str, bool] = field(default_factory=dict)
    shelf_level_relation_correct: bool = False
    domain_success: bool = False
    details: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        step_id = int(self.step_id)
        if step_id < 0:
            raise ValueError("WR-05 evaluation step_id must be non-negative")
        staged = _flags(self.staged_at_plate, field_name="staged_at_plate")
        home = _flags(
            self.home_correct_by_condiment,
            field_name="home_correct_by_condiment",
        )
        released = _flags(
            self.released_by_condiment,
            field_name="released_by_condiment",
        )
        relation_correct = bool(self.shelf_level_relation_correct)
        success = bool(self.domain_success)
        if success and not (
            all(staged.values())
            and all(home.values())
            and all(released.values())
            and relation_correct
        ):
            raise ValueError(
                "WR-05 domain_success requires both items staged, correctly "
                "returned, released, and in the correct shelf relation"
            )
        object.__setattr__(self, "step_id", step_id)
        object.__setattr__(self, "staged_at_plate", staged)
        object.__setattr__(self, "home_correct_by_condiment", home)
        object.__setattr__(self, "released_by_condiment", released)
        object.__setattr__(self, "shelf_level_relation_correct", relation_correct)
        object.__setattr__(self, "domain_success", success)
        object.__setattr__(self, "details", dict(self.details or {}))

    @property
    def staged_count(self) -> int:
        return sum(bool(value) for value in self.staged_at_plate.values())

    @property
    def return_candidates(self) -> dict[str, bool]:
        return {
            condiment_id: bool(
                self.home_correct_by_condiment[condiment_id]
                and self.released_by_condiment[condiment_id]
            )
            for condiment_id in CONDIMENT_IDS
        }

    def raw_stage_values(self) -> dict[str, bool]:
        return {
            "one_item_staged": self.staged_count >= 1,
            "two_items_staged": self.staged_count == len(CONDIMENT_IDS),
            "one_item_correctly_returned": any(self.return_candidates.values()),
            "success": bool(self.domain_success),
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
class WR05EvalResult:
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
            "task_id": WR05_REPORT_PROFILE.task_id,
            "profile_id": REPORT_STAGE_PROFILE_ID,
            "order": list(REPORT_STAGE_ORDER),
            "stages": stages,
        }


class WR05Evaluator:
    """History-aware evaluator for the two-item stage-and-return protocol."""

    def __init__(self) -> None:
        self._latched = {name: False for name in STAGE_ORDER[:-1]}
        self._first_observed_steps = {name: None for name in STAGE_ORDER}
        self._first_qualified_steps = {name: None for name in STAGE_ORDER}
        self._rejected: dict[str, list[dict[str, Any]]] = {
            name: [] for name in STAGE_ORDER
        }
        self._previous_raw = {name: False for name in STAGE_ORDER}
        self._previous_return_candidates = {
            condiment_id: False for condiment_id in CONDIMENT_IDS
        }
        self._ever_success = False
        self._last_step_id: int | None = None
        self._cached_result: WR05EvalResult | None = None

    def reset(self, snapshot: WR05EvaluationSnapshot) -> WR05EvalResult:
        self._latched = {name: False for name in STAGE_ORDER[:-1]}
        self._first_observed_steps = {name: None for name in STAGE_ORDER}
        self._first_qualified_steps = {name: None for name in STAGE_ORDER}
        self._rejected = {name: [] for name in STAGE_ORDER}
        self._ever_success = False
        raw = snapshot.raw_stage_values()
        for name, value in raw.items():
            if value and name != "one_item_correctly_returned":
                self._first_observed_steps[name] = snapshot.step_id
                self._reject(name, snapshot.step_id, "present_at_reset")
        self._previous_raw = raw
        self._previous_return_candidates = snapshot.return_candidates
        self._last_step_id = snapshot.step_id
        self._cached_result = self._build_result(snapshot, success_current=False)
        return self._cached_result

    def observe(self, snapshot: WR05EvaluationSnapshot) -> WR05EvalResult:
        if self._last_step_id is None:
            return self.reset(snapshot)
        if snapshot.step_id == self._last_step_id:
            if self._cached_result is None:
                raise RuntimeError("WR-05 evaluator has no cached result")
            return self._cached_result
        if snapshot.step_id < self._last_step_id:
            raise ValueError(
                "WR-05 evaluation step moved backwards: "
                f"{snapshot.step_id} < {self._last_step_id}"
            )

        raw = snapshot.raw_stage_values()
        two_items_staged_before = self._latched["two_items_staged"]
        for name, value in raw.items():
            if (
                name != "one_item_correctly_returned"
                and value
                and self._first_observed_steps[name] is None
            ):
                self._first_observed_steps[name] = snapshot.step_id

        for name in ("one_item_staged", "two_items_staged"):
            if self._latched[name] or not raw[name] or self._previous_raw[name]:
                continue
            prerequisite = STAGE_PREREQUISITES[name]
            if prerequisite is not None and not self._latched[prerequisite]:
                self._reject(
                    name,
                    snapshot.step_id,
                    f"prerequisite_not_qualified:{prerequisite}",
                )
                continue
            self._qualify(name, snapshot.step_id)

        return_edges = [
            condiment_id
            for condiment_id, returned in snapshot.return_candidates.items()
            if returned and not self._previous_return_candidates[condiment_id]
        ]
        if (
            return_edges
            and self._first_observed_steps["one_item_correctly_returned"] is None
        ):
            self._first_observed_steps["one_item_correctly_returned"] = (
                snapshot.step_id
            )
        if return_edges and not self._latched["one_item_correctly_returned"]:
            if not two_items_staged_before:
                self._reject(
                    "one_item_correctly_returned",
                    snapshot.step_id,
                    "return_before_two_items_staged:" + ",".join(return_edges),
                )
            else:
                self._qualify("one_item_correctly_returned", snapshot.step_id)

        success_current = bool(
            raw["success"] and self._latched["one_item_correctly_returned"]
        )
        if raw["success"] and not success_current:
            self._reject(
                "success",
                snapshot.step_id,
                "prerequisite_not_qualified:one_item_correctly_returned",
            )
        if success_current:
            self._ever_success = True
            if self._first_qualified_steps["success"] is None:
                self._first_qualified_steps["success"] = snapshot.step_id

        self._previous_raw = raw
        self._previous_return_candidates = snapshot.return_candidates
        self._last_step_id = snapshot.step_id
        self._cached_result = self._build_result(
            snapshot,
            success_current=success_current,
        )
        return self._cached_result

    def result(self) -> WR05EvalResult:
        if self._cached_result is None:
            raise RuntimeError("WR-05 evaluator has not been reset")
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
        snapshot: WR05EvaluationSnapshot,
        *,
        success_current: bool,
    ) -> WR05EvalResult:
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
                    "historical_staging_then_per_item_return_edge"
                ),
                "raw_stage_checks": snapshot.raw_stage_values(),
                "return_candidates": snapshot.return_candidates,
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
                "staged_at_plate": dict(snapshot.staged_at_plate),
                "current_condiment_home_correct": dict(
                    snapshot.home_correct_by_condiment
                ),
                "released_by_condiment": dict(snapshot.released_by_condiment),
                "condiment_shelf_level_relation_correct": bool(
                    snapshot.shelf_level_relation_correct
                ),
                "domain_success": bool(snapshot.domain_success),
            }
        )
        return WR05EvalResult(
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
        snapshot: WR05EvaluationSnapshot,
        success_current: bool,
    ) -> StageEvalResult:
        if name == "success":
            qualified = bool(success_current)
            invalidated = bool(self._ever_success and not qualified)
        else:
            qualified = bool(self._latched[name])
            invalidated = False
        prerequisite = STAGE_PREREQUISITES[name]
        rejected = self._rejected[name]
        if qualified:
            reason = None
        elif name == "success" and invalidated:
            reason = "final_state_regressed"
        elif rejected:
            reason = str(rejected[-1]["reason"])
            invalidated = True
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


def evaluate_initialized_runtime(runtime: Any) -> WR05EvalResult:
    initialized = getattr(runtime, "initialized", None) or runtime
    metadata = getattr(initialized, "metadata", {}) or {}
    evaluator = metadata.get(TASK_EVALUATOR_METADATA_KEY)
    env = (
        getattr(initialized, "env", None)
        or getattr(runtime, "env", None)
        or initialized
    )
    if evaluator is None:
        evaluator = getattr(env, "_wr05_evaluator", None)
    if not isinstance(evaluator, WR05Evaluator):
        raise RuntimeError("WR-05 runtime has no initialized task evaluator")
    snapshot_builder = getattr(env, "evaluation_snapshot", None)
    if not callable(snapshot_builder):
        raise AttributeError("WR-05 environment has no evaluation_snapshot()")
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
    "WR05EvalResult",
    "WR05EvaluationSnapshot",
    "WR05Evaluator",
    "diagnostics_for_runtime",
    "evaluate_initialized_runtime",
    "report_stages_for_runtime",
    "success_for_runtime",
]
