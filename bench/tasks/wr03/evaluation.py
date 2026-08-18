"""Authoritative ordered WR-03 evaluation state machine.

The physical observer reconstructs stable containment, door, and retrieval
facts.  This module owns the public evaluation contract: ordered stage
qualification, skipped-stage diagnostics, terminal protocol failures, and the
final full-horizon verdict.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from bench.tasks.wr03.domain.contracts import CAN_ID, CABINET_LEVELS


STAGE_ORDER = (
    "can_pick",
    "can_store",
    "close_both_cabinets",
    "open_remembered_cabinet",
    "retrieve_can",
)
STAGE_LABELS = {
    "can_pick": "罐头取",
    "can_store": "罐头存",
    "close_both_cabinets": "双柜门关闭并锁定记忆",
    "open_remembered_cabinet": "打开记忆柜",
    "retrieve_can": "取回罐头",
}
STAGE_PREREQUISITES: dict[str, str | None] = {
    name: None if index == 0 else STAGE_ORDER[index - 1]
    for index, name in enumerate(STAGE_ORDER)
}
TASK_EVALUATOR_METADATA_KEY = "task_evaluator"


def _level(value: Any) -> str:
    level = str(value)
    if level not in CABINET_LEVELS:
        raise ValueError(f"invalid WR-03 cabinet level: {value!r}")
    return level


@dataclass(frozen=True, slots=True)
class WR03EvaluationSnapshot:
    """One read-only physical/semantic observation for the evaluator."""

    step_id: int
    can_grasped: bool = False
    stored_level: str | None = None
    storage_hidden: bool = False
    lower_door_closed: bool = False
    upper_door_closed: bool = False
    opened_level: str | None = None
    wrong_retrieval_level_opened: bool = False
    retrieved_item_id: str | None = None
    can_outside_both: bool = False
    other_door_closed: bool = False
    protocol_violation: bool = False
    details: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        step = int(self.step_id)
        if step < 0:
            raise ValueError("WR-03 evaluation step_id must be non-negative")
        stored = None if self.stored_level is None else _level(self.stored_level)
        opened = None if self.opened_level is None else _level(self.opened_level)
        retrieved = (
            None
            if self.retrieved_item_id is None
            else str(self.retrieved_item_id)
        )
        if retrieved is not None and retrieved != CAN_ID:
            raise ValueError(f"invalid WR-03 retrieved item id: {retrieved!r}")
        object.__setattr__(self, "step_id", step)
        object.__setattr__(self, "stored_level", stored)
        object.__setattr__(self, "opened_level", opened)
        object.__setattr__(self, "retrieved_item_id", retrieved)
        for name in (
            "can_grasped",
            "storage_hidden",
            "lower_door_closed",
            "upper_door_closed",
            "wrong_retrieval_level_opened",
            "can_outside_both",
            "other_door_closed",
            "protocol_violation",
        ):
            object.__setattr__(self, name, bool(getattr(self, name)))
        object.__setattr__(self, "details", dict(self.details or {}))

    @property
    def both_doors_closed(self) -> bool:
        return bool(self.lower_door_closed and self.upper_door_closed)

    @property
    def correct_cabinet_opened(self) -> bool:
        return bool(
            self.storage_hidden
            and self.stored_level is not None
            and self.opened_level == self.stored_level
            and not self.wrong_retrieval_level_opened
            and not self.protocol_violation
        )

    @property
    def final_retrieval_valid(self) -> bool:
        return bool(
            self.retrieved_item_id == CAN_ID
            and self.can_outside_both
            and self.correct_cabinet_opened
            and self.other_door_closed
            and not self.protocol_violation
        )

    def raw_stage_values(self) -> dict[str, bool]:
        return {
            "can_pick": self.can_grasped,
            "can_store": self.stored_level is not None,
            "close_both_cabinets": bool(
                self.stored_level is not None
                and self.storage_hidden
                and self.both_doors_closed
            ),
            "open_remembered_cabinet": self.opened_level is not None,
            "retrieve_can": self.retrieved_item_id is not None,
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
class WR03EvalResult:
    success: bool
    terminal: bool
    failure_code: str | None
    stages: Mapping[str, StageEvalResult]
    details: Mapping[str, Any] = field(default_factory=dict)

    def to_diagnostics(self) -> dict[str, Any]:
        payload = dict(self.details or {})
        stages = {name: self.stages[name].to_dict() for name in STAGE_ORDER}
        qualified = {
            name: bool(values["qualified"])
            for name, values in stages.items()
        }
        observed = {
            name: bool(values["observed"])
            for name, values in stages.items()
        }
        legacy_checks = dict(payload.get("success_checks") or {})
        evaluation = dict(payload.get("evaluation") or {})
        evaluation.update(
            {
                "schema_version": 1,
                "step_id": int(evaluation.get("step_id", payload.get("step_id", 0))),
                "stage_order": list(STAGE_ORDER),
                "stage_labels": dict(STAGE_LABELS),
                "stage_prerequisites": dict(STAGE_PREREQUISITES),
                "qualification_policy": "rising_edge_after_prerequisite",
                "raw_stage_checks": payload.get("raw_stage_checks", {}),
                "rejected_stage_observations": payload.get(
                    "rejected_stage_observations", {}
                ),
            }
        )
        payload["evaluation"] = evaluation
        payload.update(
            {
                "success": bool(self.success),
                "terminal": bool(self.terminal),
                "failure_code": self.failure_code,
                # Keep the pre-existing physical SuccessReport contract.
                "success_checks": legacy_checks,
                "stages": stages,
                "stage_checks": qualified,
                "stage_observed": observed,
                "stage_labels": dict(STAGE_LABELS),
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


@dataclass
class _MutableStage:
    observed: bool = False
    qualified: bool = False
    first_observed_step: int | None = None
    qualified_step: int | None = None


class WR03Evaluator:
    """Strict ordered evaluator backed by read-only WR-03 observations."""

    def __init__(self) -> None:
        self._stages = {name: _MutableStage() for name in STAGE_ORDER}
        self._rejected: dict[str, list[dict[str, Any]]] = {
            name: [] for name in STAGE_ORDER
        }
        self._previous_raw = {name: False for name in STAGE_ORDER}
        self._last_step_id: int | None = None
        self._terminal_failure_code: str | None = None
        self._cached_result: WR03EvalResult | None = None

    def reset(self, snapshot: WR03EvaluationSnapshot) -> WR03EvalResult:
        self._stages = {name: _MutableStage() for name in STAGE_ORDER}
        self._rejected = {name: [] for name in STAGE_ORDER}
        self._terminal_failure_code = None
        raw = snapshot.raw_stage_values()
        for name, value in raw.items():
            if value:
                self._mark_observed(name, snapshot.step_id)
                self._reject(name, snapshot.step_id, "present_at_reset")
        self._previous_raw = raw
        self._last_step_id = snapshot.step_id
        self._cached_result = self._build_result(snapshot)
        return self._cached_result

    def observe(self, snapshot: WR03EvaluationSnapshot) -> WR03EvalResult:
        if self._last_step_id is None:
            return self.reset(snapshot)
        if snapshot.step_id == self._last_step_id:
            if self._cached_result is None:
                raise RuntimeError("WR-03 evaluator has no cached result")
            return self._cached_result
        if snapshot.step_id < self._last_step_id:
            raise ValueError(
                "WR-03 evaluation step moved backwards: "
                f"{snapshot.step_id} < {self._last_step_id}"
            )
        # Failure states are absorbing. A successful result is deliberately
        # re-evaluated on later steps because WR-03 uses full-horizon final
        # validation: opening the other door or putting the can back must revoke
        # the final success, even though the historical stages remain qualified.
        if self._terminal_failure_code is not None:
            return self._cached_result or self._build_result(snapshot)

        raw = snapshot.raw_stage_values()
        qualified_before = {
            name: self._stages[name].qualified for name in STAGE_ORDER
        }
        for name, value in raw.items():
            if value:
                self._mark_observed(name, snapshot.step_id)

        if snapshot.protocol_violation:
            self._terminal_failure_code = "protocol_violation"
        for name in STAGE_ORDER:
            if (
                not raw[name]
                or self._previous_raw[name]
                or self._stages[name].qualified
            ):
                continue
            prerequisite = STAGE_PREREQUISITES[name]
            if prerequisite is not None and not qualified_before[prerequisite]:
                self._reject(
                    name,
                    snapshot.step_id,
                    f"prerequisite_not_qualified:{prerequisite}",
                )
                continue
            reason = self._stage_validation_failure(name, snapshot)
            if reason is not None:
                self._reject(name, snapshot.step_id, reason)
                if name in {"open_remembered_cabinet", "retrieve_can"}:
                    self._terminal_failure_code = reason
                continue
            self._qualify(name, snapshot.step_id)

        self._previous_raw = raw
        self._last_step_id = snapshot.step_id
        self._cached_result = self._build_result(snapshot)
        return self._cached_result

    def result(self) -> WR03EvalResult:
        if self._cached_result is None:
            raise RuntimeError("WR-03 evaluator has not been reset")
        return self._cached_result

    @staticmethod
    def _stage_validation_failure(
        name: str,
        snapshot: WR03EvaluationSnapshot,
    ) -> str | None:
        if name == "open_remembered_cabinet":
            if snapshot.wrong_retrieval_level_opened:
                return "wrong_retrieval_cabinet"
            if not snapshot.correct_cabinet_opened:
                return "wrong_retrieval_cabinet"
        elif name == "retrieve_can":
            if snapshot.retrieved_item_id != CAN_ID:
                return "wrong_item_retrieved"
            if not snapshot.can_outside_both:
                return "can_not_extracted"
            if not snapshot.other_door_closed:
                return "other_door_not_closed"
            if not snapshot.correct_cabinet_opened:
                return "wrong_retrieval_cabinet"
        return None

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
        entry = {"step": int(step_id), "reason": str(reason)}
        if entry not in self._rejected[name]:
            self._rejected[name].append(entry)

    def _stage_result(self, name: str) -> StageEvalResult:
        stage = self._stages[name]
        rejected = self._rejected[name]
        if stage.qualified:
            invalidated = False
            reason = None
        elif rejected:
            invalidated = True
            reason = str(rejected[-1]["reason"])
        else:
            prerequisite = STAGE_PREREQUISITES[name]
            reason = (
                f"waiting_for_prerequisite:{prerequisite}"
                if prerequisite is not None
                and not self._stages[prerequisite].qualified
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

    def _build_result(self, snapshot: WR03EvaluationSnapshot) -> WR03EvalResult:
        stages = {name: self._stage_result(name) for name in STAGE_ORDER}
        final_valid = snapshot.final_retrieval_valid
        success = bool(stages[STAGE_ORDER[-1]].qualified and final_valid)
        details = dict(snapshot.details)
        evaluation = dict(details.get("evaluation") or {})
        evaluation.update(
            {
                "step_id": int(snapshot.step_id),
                "raw_stage_checks": snapshot.raw_stage_values(),
                "rejected_stage_observations": {
                    name: list(values)
                    for name, values in self._rejected.items()
                    if values
                },
            }
        )
        details["evaluation"] = evaluation
        return WR03EvalResult(
            success=success,
            terminal=bool(success or self._terminal_failure_code),
            failure_code=self._terminal_failure_code,
            stages=stages,
            details=details,
        )


def evaluate_initialized_runtime(runtime: Any) -> WR03EvalResult:
    initialized = getattr(runtime, "initialized", None) or runtime
    metadata = getattr(initialized, "metadata", {}) or {}
    evaluator = metadata.get(TASK_EVALUATOR_METADATA_KEY)
    env = getattr(initialized, "env", None) or getattr(runtime, "env", None) or initialized
    if evaluator is None:
        evaluator = getattr(env, "_wr03_evaluator", None)
    if not isinstance(evaluator, WR03Evaluator):
        raise RuntimeError("WR-03 runtime has no initialized task evaluator")
    snapshot_builder = getattr(env, "evaluation_snapshot", None)
    if not callable(snapshot_builder):
        raise AttributeError("WR-03 environment has no evaluation_snapshot()")
    return evaluator.observe(snapshot_builder())


def success_for_runtime(runtime: Any, info: Mapping[str, Any] | None = None) -> bool:
    del info
    return bool(evaluate_initialized_runtime(runtime).success)


def diagnostics_for_runtime(runtime: Any) -> Mapping[str, Any]:
    return evaluate_initialized_runtime(runtime).to_diagnostics()


__all__ = [
    "STAGE_LABELS",
    "STAGE_ORDER",
    "STAGE_PREREQUISITES",
    "TASK_EVALUATOR_METADATA_KEY",
    "StageEvalResult",
    "WR03EvalResult",
    "WR03EvaluationSnapshot",
    "WR03Evaluator",
    "diagnostics_for_runtime",
    "evaluate_initialized_runtime",
    "success_for_runtime",
]
