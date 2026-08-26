"""Authoritative WR-06 benchmark evaluation state machine.

The physical observer owns the fine-grained bottle protocol. This module
projects its history and the current final-state predicate onto the stable
public report funnel without exposing the internal memory barrier as a stage.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from bench.evaluation.report_profiles import WR06_REPORT_PROFILE
from bench.tasks.wr06.domain.contracts import ITEM_ID
from bench.tasks.wr06.domain.episode import validate_level


STAGE_ORDER = WR06_REPORT_PROFILE.stage_ids
STAGE_PREREQUISITES: dict[str, str | None] = {
    "item_removed": None,
    "item_staged": "item_removed",
    "correct_return_level": "item_staged",
    "success": "correct_return_level",
}
REPORT_STAGE_ORDER = STAGE_ORDER
REPORT_STAGE_PROFILE_ID = WR06_REPORT_PROFILE.profile_id
TASK_EVALUATOR_METADATA_KEY = "task_evaluator"


@dataclass(frozen=True, slots=True)
class WR06EvaluationSnapshot:
    """One read-only observation of WR-06 semantic history and final state."""

    step_id: int
    initial_level: str | None = None
    initial_level_physically_verified: bool = False
    item_removed_from_home: bool = False
    item_staged_at_plate: bool = False
    memory_barrier_latched: bool = False
    first_return_level: str | None = None
    wrong_return_level_attempted: bool = False
    returned_item_id: str | None = None
    protocol_violation: bool = False
    stage_regressions: Sequence[str] = ()
    domain_success: bool = False
    failure_code: str | None = None
    details: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        step_id = int(self.step_id)
        if step_id < 0:
            raise ValueError("WR-06 evaluation step_id must be non-negative")
        initial_level = (
            None
            if self.initial_level is None
            else validate_level(self.initial_level)
        )
        first_return_level = (
            None
            if self.first_return_level is None
            else validate_level(self.first_return_level)
        )
        returned_item_id = (
            None
            if self.returned_item_id is None
            else str(self.returned_item_id)
        )
        if returned_item_id is not None and returned_item_id != ITEM_ID:
            raise ValueError(
                f"invalid WR-06 returned item id: {returned_item_id!r}"
            )

        initial_verified = bool(self.initial_level_physically_verified)
        item_removed = bool(self.item_removed_from_home)
        item_staged = bool(self.item_staged_at_plate)
        barrier = bool(self.memory_barrier_latched)
        wrong_return = bool(self.wrong_return_level_attempted)
        protocol_violation = bool(self.protocol_violation)
        domain_success = bool(self.domain_success)
        if initial_verified and initial_level is None:
            raise ValueError(
                "WR-06 verified initial level requires an initial_level"
            )
        if barrier and not (initial_verified and item_removed and item_staged):
            raise ValueError(
                "WR-06 memory barrier requires verified removal and staging"
            )
        if first_return_level is not None and not barrier:
            raise ValueError(
                "WR-06 first return choice requires the memory barrier"
            )
        correct_return = bool(
            initial_level is not None
            and first_return_level == initial_level
            and not wrong_return
        )
        if domain_success and not (
            initial_verified
            and item_removed
            and item_staged
            and barrier
            and correct_return
            and returned_item_id == ITEM_ID
            and not protocol_violation
        ):
            raise ValueError(
                "WR-06 domain_success requires the complete legal return protocol"
            )

        object.__setattr__(self, "step_id", step_id)
        object.__setattr__(self, "initial_level", initial_level)
        object.__setattr__(
            self,
            "initial_level_physically_verified",
            initial_verified,
        )
        object.__setattr__(self, "item_removed_from_home", item_removed)
        object.__setattr__(self, "item_staged_at_plate", item_staged)
        object.__setattr__(self, "memory_barrier_latched", barrier)
        object.__setattr__(self, "first_return_level", first_return_level)
        object.__setattr__(self, "wrong_return_level_attempted", wrong_return)
        object.__setattr__(self, "returned_item_id", returned_item_id)
        object.__setattr__(self, "protocol_violation", protocol_violation)
        object.__setattr__(
            self,
            "stage_regressions",
            tuple(str(value) for value in (self.stage_regressions or ())),
        )
        object.__setattr__(self, "domain_success", domain_success)
        object.__setattr__(
            self,
            "failure_code",
            None if self.failure_code is None else str(self.failure_code),
        )
        object.__setattr__(self, "details", dict(self.details or {}))

    @property
    def return_choice_observed(self) -> bool:
        return self.first_return_level is not None

    @property
    def first_return_level_correct(self) -> bool:
        return bool(
            self.initial_level is not None
            and self.first_return_level == self.initial_level
            and not self.wrong_return_level_attempted
        )

    def raw_stage_values(self) -> dict[str, bool]:
        return {
            "item_removed": self.item_removed_from_home,
            "item_staged": self.item_staged_at_plate,
            "correct_return_level": self.first_return_level_correct,
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
class WR06EvalResult:
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
            "task_id": WR06_REPORT_PROFILE.task_id,
            "profile_id": REPORT_STAGE_PROFILE_ID,
            "order": list(REPORT_STAGE_ORDER),
            "stages": stages,
        }


class WR06Evaluator:
    """History-aware evaluator for the single-bottle return protocol."""

    def __init__(self) -> None:
        self._latched = {name: False for name in STAGE_ORDER[:-1]}
        self._first_observed_steps = {name: None for name in STAGE_ORDER}
        self._first_qualified_steps = {name: None for name in STAGE_ORDER}
        self._rejected: dict[str, list[dict[str, Any]]] = {
            name: [] for name in STAGE_ORDER
        }
        self._previous_raw = {name: False for name in STAGE_ORDER}
        self._ever_success = False
        self._wrong_first_return = False
        self._last_step_id: int | None = None
        self._cached_result: WR06EvalResult | None = None

    def reset(self, snapshot: WR06EvaluationSnapshot) -> WR06EvalResult:
        self._latched = {name: False for name in STAGE_ORDER[:-1]}
        self._first_observed_steps = {name: None for name in STAGE_ORDER}
        self._first_qualified_steps = {name: None for name in STAGE_ORDER}
        self._rejected = {name: [] for name in STAGE_ORDER}
        self._ever_success = False
        self._wrong_first_return = False
        raw = snapshot.raw_stage_values()
        for name, value in raw.items():
            if value:
                self._first_observed_steps[name] = snapshot.step_id
                self._reject(name, snapshot.step_id, "present_at_reset")
        if snapshot.return_choice_observed and not raw["correct_return_level"]:
            self._first_observed_steps["correct_return_level"] = snapshot.step_id
            self._wrong_first_return = True
            self._reject(
                "correct_return_level",
                snapshot.step_id,
                f"wrong_first_return_level:{snapshot.first_return_level}",
            )
        self._previous_raw = raw
        self._last_step_id = snapshot.step_id
        self._cached_result = self._build_result(snapshot, success_current=False)
        return self._cached_result

    def observe(self, snapshot: WR06EvaluationSnapshot) -> WR06EvalResult:
        if self._last_step_id is None:
            return self.reset(snapshot)
        if snapshot.step_id == self._last_step_id:
            if self._cached_result is None:
                raise RuntimeError("WR-06 evaluator has no cached result")
            return self._cached_result
        if snapshot.step_id < self._last_step_id:
            raise ValueError(
                "WR-06 evaluation step moved backwards: "
                f"{snapshot.step_id} < {self._last_step_id}"
            )

        raw = snapshot.raw_stage_values()
        for name in ("item_removed", "item_staged"):
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

        if (
            snapshot.return_choice_observed
            and self._first_observed_steps["correct_return_level"] is None
        ):
            self._first_observed_steps["correct_return_level"] = snapshot.step_id
        if snapshot.return_choice_observed and not self._latched[
            "correct_return_level"
        ]:
            if not raw["correct_return_level"]:
                if not self._wrong_first_return:
                    self._wrong_first_return = True
                    self._reject(
                        "correct_return_level",
                        snapshot.step_id,
                        f"wrong_first_return_level:{snapshot.first_return_level}",
                    )
            elif self._wrong_first_return:
                self._reject(
                    "correct_return_level",
                    snapshot.step_id,
                    "first_return_choice_already_failed",
                )
            elif not self._latched["item_staged"]:
                self._reject(
                    "correct_return_level",
                    snapshot.step_id,
                    "prerequisite_not_qualified:item_staged",
                )
            elif not snapshot.memory_barrier_latched:
                self._reject(
                    "correct_return_level",
                    snapshot.step_id,
                    "memory_barrier_not_latched",
                )
            else:
                self._qualify("correct_return_level", snapshot.step_id)

        if raw["success"] and self._first_observed_steps["success"] is None:
            self._first_observed_steps["success"] = snapshot.step_id
        success_current = bool(
            raw["success"]
            and self._latched["correct_return_level"]
            and not self._wrong_first_return
            and not snapshot.protocol_violation
        )
        if raw["success"] and not success_current:
            self._reject(
                "success",
                snapshot.step_id,
                "prerequisite_not_qualified:correct_return_level",
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

    def result(self) -> WR06EvalResult:
        if self._cached_result is None:
            raise RuntimeError("WR-06 evaluator has not been reset")
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
        snapshot: WR06EvaluationSnapshot,
        *,
        success_current: bool,
    ) -> WR06EvalResult:
        stages = {
            name: self._stage_result(
                name,
                snapshot=snapshot,
                success_current=success_current,
            )
            for name in STAGE_ORDER
        }
        failure_code = self._failure_code(snapshot, success_current)
        details = dict(snapshot.details)
        evaluation = dict(details.get("evaluation") or {})
        evaluation.update(
            {
                "schema_version": 1,
                "step_id": int(snapshot.step_id),
                "stage_order": list(STAGE_ORDER),
                "stage_prerequisites": dict(STAGE_PREREQUISITES),
                "qualification_policy": "ordered_history_with_first_return_choice",
                "raw_stage_checks": snapshot.raw_stage_values(),
                "return_choice_observed": snapshot.return_choice_observed,
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
                "initial_level": snapshot.initial_level,
                "memory_barrier_latched": snapshot.memory_barrier_latched,
                "first_return_level": snapshot.first_return_level,
                "wrong_return_level_attempted": (
                    snapshot.wrong_return_level_attempted
                ),
                "returned_item_id": snapshot.returned_item_id,
                "protocol_violation": snapshot.protocol_violation,
                "stage_regressions": list(snapshot.stage_regressions),
                "domain_success": snapshot.domain_success,
            }
        )
        return WR06EvalResult(
            success=bool(success_current),
            terminal=bool(success_current or failure_code is not None),
            failure_code=failure_code,
            stages=stages,
            details=details,
        )

    def _stage_result(
        self,
        name: str,
        *,
        snapshot: WR06EvaluationSnapshot,
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

    def _failure_code(
        self,
        snapshot: WR06EvaluationSnapshot,
        success_current: bool,
    ) -> str | None:
        if success_current:
            return None
        if snapshot.failure_code:
            return snapshot.failure_code
        if self._wrong_first_return or snapshot.wrong_return_level_attempted:
            return "wrong_return_level"
        if snapshot.stage_regressions:
            return snapshot.stage_regressions[-1]
        if snapshot.protocol_violation:
            return "protocol_violation"
        return None


def evaluate_initialized_runtime(runtime: Any) -> WR06EvalResult:
    initialized = getattr(runtime, "initialized", None) or runtime
    metadata = getattr(initialized, "metadata", {}) or {}
    evaluator = metadata.get(TASK_EVALUATOR_METADATA_KEY)
    env = (
        getattr(initialized, "env", None)
        or getattr(runtime, "env", None)
        or initialized
    )
    if evaluator is None:
        evaluator = getattr(env, "_wr06_evaluator", None)
    if not isinstance(evaluator, WR06Evaluator):
        raise RuntimeError("WR-06 runtime has no initialized task evaluator")
    snapshot_builder = getattr(env, "evaluation_snapshot", None)
    if not callable(snapshot_builder):
        raise AttributeError("WR-06 environment has no evaluation_snapshot()")
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
    "WR06EvalResult",
    "WR06EvaluationSnapshot",
    "WR06Evaluator",
    "diagnostics_for_runtime",
    "evaluate_initialized_runtime",
    "report_stages_for_runtime",
    "success_for_runtime",
]
