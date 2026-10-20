"""Authoritative WR-01 task progress with unordered cabinet operations.

The simulator supplies physical grasp, cabinet occupancy, door, and retrieval
facts. This module owns task prerequisites, diagnostics, and the final verdict
for policy evaluation and recorded-action replay. Either cabinet may be closed
first; after storage is hidden, the target cabinet must be opened first.
The non-target can must remain in its initial cabinet from reset until success;
any observed departure is a terminal failure, even if the can is later returned.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from bench.evaluation.report_profiles import WR01_REPORT_PROFILE
from bench.tasks.wr01.domain.contracts import CABINET_LEVELS, ITEM_IDS


STAGE_ORDER = (
    "close_first_door",
    "close_second_door",
    "open_cabinet",
    "retrieve_item",
    "place_on_table",
)
STAGE_LABELS = {
    "close_first_door": "关闭任意一扇柜门",
    "close_second_door": "关闭两扇柜门",
    "open_cabinet": "开记忆柜",
    "retrieve_item": "取目标罐头",
    "place_on_table": "放回桌面",
}
STAGE_PREREQUISITES: dict[str, str | None] = {
    name: None if index == 0 else STAGE_ORDER[index - 1]
    for index, name in enumerate(STAGE_ORDER)
}
REPORT_STAGE_ORDER = WR01_REPORT_PROFILE.stage_ids
REPORT_STAGE_PROFILE_ID = WR01_REPORT_PROFILE.profile_id
TASK_EVALUATOR_METADATA_KEY = "task_evaluator"

_REPORT_STAGE_SOURCES = {
    "one_can_stored": "close_first_door",
    "two_cans_stored": "close_second_door",
    "correct_cabinet_opened": "open_cabinet",
    "success": "place_on_table",
}


def _item(value: Any) -> str:
    item_id = str(value)
    if item_id not in ITEM_IDS:
        raise ValueError(f"invalid WR-01 item id: {value!r}")
    return item_id


def _level(value: Any) -> str:
    level = str(value)
    if level not in CABINET_LEVELS:
        raise ValueError(f"invalid WR-01 cabinet level: {value!r}")
    return level


def _storage_order(values: Sequence[Any]) -> tuple[str, str]:
    if isinstance(values, (str, bytes)) or len(values) != len(ITEM_IDS):
        raise ValueError("WR-01 storage_order must contain two item ids")
    result = tuple(_item(value) for value in values)
    if set(result) != set(ITEM_IDS):
        raise ValueError("WR-01 storage_order must contain each item exactly once")
    return result  # type: ignore[return-value]


@dataclass(frozen=True, slots=True)
class WR01EvaluationSnapshot:
    """One physical observation presented to the ordered evaluator."""

    step_id: int
    storage_order: tuple[str, str]
    target_item_id: str
    target_level: str
    item_grasped: Mapping[str, bool] = field(default_factory=dict)
    stored_levels_by_item: Mapping[str, str] = field(default_factory=dict)
    storage_hidden: bool = False
    upper_door_closed: bool = False
    lower_door_closed: bool = False
    opened_level: str | None = None
    wrong_retrieval_level_opened: bool = False
    retrieved_item_id: str | None = None
    distractor_still_stored: bool = False
    final_placement_verified: bool = False
    details: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        step = int(self.step_id)
        if step < 0:
            raise ValueError("WR-01 evaluation step_id must be non-negative")
        order = _storage_order(self.storage_order)
        target = _item(self.target_item_id)
        target_level = _level(self.target_level)
        grasped = {
            item_id: bool(self.item_grasped.get(item_id, False))
            for item_id in ITEM_IDS
        }
        stored: dict[str, str] = {}
        for item_id, level in self.stored_levels_by_item.items():
            stored[_item(item_id)] = _level(level)
        opened = None if self.opened_level is None else _level(self.opened_level)
        retrieved = (
            None if self.retrieved_item_id is None else _item(self.retrieved_item_id)
        )
        object.__setattr__(self, "step_id", step)
        object.__setattr__(self, "storage_order", order)
        object.__setattr__(self, "target_item_id", target)
        object.__setattr__(self, "target_level", target_level)
        object.__setattr__(self, "item_grasped", grasped)
        object.__setattr__(self, "stored_levels_by_item", stored)
        object.__setattr__(self, "storage_hidden", bool(self.storage_hidden))
        object.__setattr__(self, "upper_door_closed", bool(self.upper_door_closed))
        object.__setattr__(self, "lower_door_closed", bool(self.lower_door_closed))
        object.__setattr__(self, "opened_level", opened)
        object.__setattr__(
            self,
            "wrong_retrieval_level_opened",
            bool(self.wrong_retrieval_level_opened),
        )
        object.__setattr__(self, "retrieved_item_id", retrieved)
        object.__setattr__(
            self,
            "distractor_still_stored",
            bool(self.distractor_still_stored),
        )
        object.__setattr__(self, "final_placement_verified", bool(self.final_placement_verified))
        object.__setattr__(self, "details", dict(self.details or {}))

    @property
    def first_item_id(self) -> str:
        return self.storage_order[0]

    @property
    def second_item_id(self) -> str:
        return self.storage_order[1]

    @property
    def storage_assignment_complete(self) -> bool:
        return bool(
            set(self.stored_levels_by_item) == set(ITEM_IDS)
            and set(self.stored_levels_by_item.values()) == set(CABINET_LEVELS)
        )

    @property
    def correct_cabinet_opened(self) -> bool:
        return bool(
            self.storage_hidden
            and self.opened_level == self.target_level
        )

    @property
    def correct_item_retrieved(self) -> bool:
        return bool(
            self.retrieved_item_id == self.target_item_id
            and self.distractor_still_stored
        )

    def raw_stage_values(self) -> dict[str, bool]:
        # The legacy first/second ids count closed doors, not cabinet identities.
        # Include storage evidence so a late observer confirmation does not
        # force the policy to reopen and reclose already closed doors.
        stored = self.storage_assignment_complete
        return {
            "close_first_door": stored and (self.upper_door_closed or self.lower_door_closed),
            "close_second_door": stored and self.upper_door_closed and self.lower_door_closed,
            "open_cabinet": self.correct_cabinet_opened,
            "retrieve_item": self.retrieved_item_id is not None,
            "place_on_table": self.final_placement_verified,
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
class WR01EvalResult:
    success: bool
    terminal: bool
    failure_code: str | None
    stages: Mapping[str, StageEvalResult]
    details: Mapping[str, Any] = field(default_factory=dict)

    def to_diagnostics(self) -> dict[str, Any]:
        payload = dict(self.details or {})
        stages = {name: self.stages[name].to_dict() for name in STAGE_ORDER}
        qualified = {
            name: bool(values["qualified"]) for name, values in stages.items()
        }
        observed = {
            name: bool(values["observed"]) for name, values in stages.items()
        }
        payload.update(
            {
                "success": bool(self.success),
                "terminal": bool(self.terminal),
                "failure_code": self.failure_code,
                "success_checks": dict(qualified),
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

    def to_report_stages(self) -> dict[str, Any]:
        """Project the detailed evaluator state onto the public WR-01 funnel."""

        stages: dict[str, dict[str, Any]] = {}
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
            "task_id": WR01_REPORT_PROFILE.task_id,
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


class WR01Evaluator:
    """Evaluate the five task stages; neutral reset belongs to the expert plan."""

    def __init__(
        self,
        *,
        storage_order: Sequence[str],
        target_item_id: str,
        target_level: str,
    ) -> None:
        self.storage_order = _storage_order(storage_order)
        self.target_item_id = _item(target_item_id)
        self.target_level = _level(target_level)
        self._stages = {name: _MutableStage() for name in STAGE_ORDER}
        self._rejected: dict[str, list[dict[str, Any]]] = {
            name: [] for name in STAGE_ORDER
        }
        self._previous_raw = {name: False for name in STAGE_ORDER}
        self._last_step_id: int | None = None
        self._terminal_failure_code: str | None = None
        self._cached_result: WR01EvalResult | None = None

    def reset(self, snapshot: WR01EvaluationSnapshot) -> WR01EvalResult:
        self._validate_configuration(snapshot)
        self._stages = {name: _MutableStage() for name in STAGE_ORDER}
        self._rejected = {name: [] for name in STAGE_ORDER}
        self._terminal_failure_code = self._episode_validation_failure(snapshot)
        raw = snapshot.raw_stage_values()
        for name, value in raw.items():
            if value:
                self._mark_observed(name, snapshot.step_id)
                self._reject(name, snapshot.step_id, "present_at_reset")
        self._previous_raw = raw
        self._last_step_id = snapshot.step_id
        self._cached_result = self._build_result(snapshot)
        return self._cached_result

    def observe(self, snapshot: WR01EvaluationSnapshot) -> WR01EvalResult:
        self._validate_configuration(snapshot)
        if self._last_step_id is None:
            return self.reset(snapshot)
        if snapshot.step_id == self._last_step_id:
            if self._cached_result is None:
                raise RuntimeError("WR-01 evaluator has no cached result")
            return self._cached_result
        if snapshot.step_id < self._last_step_id:
            raise ValueError(
                "WR-01 evaluation step moved backwards: "
                f"{snapshot.step_id} < {self._last_step_id}"
            )
        if self._cached_result is not None and self._cached_result.terminal:
            return self._cached_result

        # Check episode-wide constraints on every new control step, before any
        # stage can qualify. A departure cannot be undone by replacing the can,
        # and it wins over target placement on the same step.
        failure = self._episode_validation_failure(snapshot)
        if failure is not None:
            self._terminal_failure_code = failure
            self._last_step_id = snapshot.step_id
            self._cached_result = self._build_result(snapshot)
            return self._cached_result

        raw = snapshot.raw_stage_values()
        qualified_before = {
            name: self._stages[name].qualified for name in STAGE_ORDER
        }
        qualified_now = dict(qualified_before)
        for name, value in raw.items():
            if value:
                self._mark_observed(name, snapshot.step_id)

        for name in STAGE_ORDER:
            if (
                not raw[name]
                or self._previous_raw[name]
                or self._stages[name].qualified
            ):
                continue
            prerequisite = STAGE_PREREQUISITES[name]
            if prerequisite is not None and not qualified_now[prerequisite]:
                self._reject(
                    name,
                    snapshot.step_id,
                    f"prerequisite_not_qualified:{prerequisite}",
                )
                continue
            reason = self._stage_validation_failure(name, snapshot)
            if reason is not None:
                self._reject(name, snapshot.step_id, reason)
                if name == "retrieve_item":
                    self._terminal_failure_code = reason
                continue
            self._qualify(name, snapshot.step_id)
            qualified_now[name] = True

        self._previous_raw = raw
        self._last_step_id = snapshot.step_id
        self._cached_result = self._build_result(snapshot)
        return self._cached_result

    def result(self) -> WR01EvalResult:
        if self._cached_result is None:
            raise RuntimeError("WR-01 evaluator has not been reset")
        return self._cached_result

    def _validate_configuration(self, snapshot: WR01EvaluationSnapshot) -> None:
        if (
            snapshot.storage_order != self.storage_order
            or snapshot.target_item_id != self.target_item_id
            or snapshot.target_level != self.target_level
        ):
            raise ValueError(
                "WR-01 evaluation snapshot configuration changed within an episode"
            )

    @staticmethod
    def _episode_validation_failure(snapshot: WR01EvaluationSnapshot) -> str | None:
        if snapshot.wrong_retrieval_level_opened:
            return "wrong_retrieval_cabinet"
        if not snapshot.distractor_still_stored:
            return "distractor_not_stored"
        return None

    @staticmethod
    def _stage_validation_failure(
        name: str,
        snapshot: WR01EvaluationSnapshot,
    ) -> str | None:
        if name == "retrieve_item":
            if snapshot.retrieved_item_id != snapshot.target_item_id:
                return "wrong_item_retrieved"
        elif name == "place_on_table":
            if not snapshot.final_placement_verified:
                return "final_placement_not_verified"
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
            invalidated = False
            prerequisite = STAGE_PREREQUISITES[name]
            reason = (
                f"waiting_for_prerequisite:{prerequisite}"
                if prerequisite is not None
                and not self._stages[prerequisite].qualified
                else "not_observed"
            )
        return StageEvalResult(
            observed=stage.observed,
            qualified=stage.qualified,
            first_observed_step=stage.first_observed_step,
            qualified_step=stage.qualified_step,
            invalidated=invalidated,
            reason=reason,
        )

    def _build_result(self, snapshot: WR01EvaluationSnapshot) -> WR01EvalResult:
        stages = {name: self._stage_result(name) for name in STAGE_ORDER}
        success = bool(
            stages[STAGE_ORDER[-1]].qualified and self._terminal_failure_code is None
        )
        details = dict(snapshot.details)
        evaluation = dict(details.get("evaluation") or {})
        evaluation.update(
            {
                "schema_version": 1,
                "step_id": int(snapshot.step_id),
                "stage_order": list(STAGE_ORDER),
                "stage_labels": dict(STAGE_LABELS),
                "stage_prerequisites": dict(STAGE_PREREQUISITES),
                "qualification_policy": "rising_edge_after_prerequisite",
                "storage_order": list(self.storage_order),
                "door_order_policy": "unordered",
                "distractor_policy": "remain_in_initial_cabinet_until_success",
                "stage_bindings": {
                    "first_can": self.storage_order[0],
                    "first_store_level": "lower",
                    "second_can": self.storage_order[1],
                    "second_store_level": "upper",
                    "retrieval_target": self.target_item_id,
                    "retrieval_level": self.target_level,
                },
                "raw_stage_checks": snapshot.raw_stage_values(),
                "rejected_stage_observations": {
                    name: list(values)
                    for name, values in self._rejected.items()
                    if values
                },
            }
        )
        details["evaluation"] = evaluation
        return WR01EvalResult(
            success=success,
            terminal=bool(success or self._terminal_failure_code),
            failure_code=self._terminal_failure_code,
            stages=stages,
            details=details,
        )


def evaluate_initialized_runtime(runtime: Any) -> WR01EvalResult:
    initialized = getattr(runtime, "initialized", None) or runtime
    metadata = getattr(initialized, "metadata", {}) or {}
    evaluator = metadata.get(TASK_EVALUATOR_METADATA_KEY)
    env = getattr(initialized, "env", None) or getattr(runtime, "env", None) or initialized
    if evaluator is None:
        evaluator = getattr(env, "_wr01_evaluator", None)
    if not isinstance(evaluator, WR01Evaluator):
        raise RuntimeError("WR-01 runtime has no initialized task evaluator")
    snapshot_builder = getattr(env, "evaluation_snapshot", None)
    if not callable(snapshot_builder):
        raise AttributeError("WR-01 environment has no evaluation_snapshot()")
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
    "STAGE_LABELS",
    "STAGE_ORDER",
    "STAGE_PREREQUISITES",
    "TASK_EVALUATOR_METADATA_KEY",
    "StageEvalResult",
    "WR01EvalResult",
    "WR01EvaluationSnapshot",
    "WR01Evaluator",
    "diagnostics_for_runtime",
    "evaluate_initialized_runtime",
    "report_stages_for_runtime",
    "success_for_runtime",
]
