"""Authoritative ordered TS03 evaluation state machine.

TS03 owns this evaluator independently from TS01.  The simulator supplies a
debounced saucepan sugar count and physical completion-button state; this
module enforces the profile-frozen sugar_1 -> sugar_2 -> button protocol,
terminal over-add, and the two-stage reporting view shared by all four TS03
K/N profiles.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from bench.evaluation.report_profiles import TS03_REPORT_PROFILE
from bench.tasks.ts03.domain.contracts import (
    SugarCountProfile,
    profile_for_counts,
    profile_for_id,
    profile_metadata,
)


SUGAR_STAGE_NAMES = ("sugar_1", "sugar_2")
STAGE_ORDER = (*SUGAR_STAGE_NAMES, "button", "success")
REPORT_STAGE_ORDER = TS03_REPORT_PROFILE.stage_ids
REPORT_STAGE_PROFILE_ID = TS03_REPORT_PROFILE.profile_id
TASK_EVALUATOR_METADATA_KEY = "task_evaluator"


def _profile_configuration(
    profile_id: Any,
    target_sugar_count: Any,
    total_sugar_count: Any,
) -> SugarCountProfile:
    selected = profile_for_id(profile_id)
    by_counts = profile_for_counts(target_sugar_count, total_sugar_count)
    if selected != by_counts:
        raise ValueError(
            f"TS03 profile_id={selected.profile_id!r} conflicts with "
            f"K={target_sugar_count}, N={total_sugar_count}"
        )
    return selected


@dataclass(frozen=True, slots=True)
class TS03EvaluationSnapshot:
    """One debounced physical observation for a frozen TS03 profile."""

    step_id: int
    profile_id: str
    target_sugar_count: int
    total_sugar_count: int
    confirmed_sugar_count: int
    objects_released: bool
    completion_button_pressed: bool
    details: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        step = int(self.step_id)
        count = int(self.confirmed_sugar_count)
        if step < 0:
            raise ValueError("step_id must be non-negative")
        if count < 0:
            raise ValueError("confirmed_sugar_count must be non-negative")
        profile = _profile_configuration(
            self.profile_id,
            self.target_sugar_count,
            self.total_sugar_count,
        )
        if count > profile.total_sugar_count:
            raise ValueError(
                "confirmed_sugar_count cannot exceed the profile's total "
                f"sugar count N={profile.total_sugar_count}, got {count}"
            )
        object.__setattr__(self, "step_id", step)
        object.__setattr__(self, "profile_id", profile.profile_id)
        object.__setattr__(
            self,
            "target_sugar_count",
            profile.target_sugar_count,
        )
        object.__setattr__(self, "total_sugar_count", profile.total_sugar_count)
        object.__setattr__(self, "confirmed_sugar_count", count)
        object.__setattr__(self, "objects_released", bool(self.objects_released))
        object.__setattr__(
            self,
            "completion_button_pressed",
            bool(self.completion_button_pressed),
        )
        object.__setattr__(self, "details", dict(self.details or {}))

    @property
    def sugar_goal_ready(self) -> bool:
        return bool(
            self.confirmed_sugar_count == self.target_sugar_count
            and self.objects_released
        )

    @property
    def over_added(self) -> bool:
        return self.confirmed_sugar_count > self.target_sugar_count

    def raw_stage_values(self) -> dict[str, bool]:
        values = {
            name: bool(
                index <= self.target_sugar_count
                and self.confirmed_sugar_count >= index
            )
            for index, name in enumerate(SUGAR_STAGE_NAMES, start=1)
        }
        values["button"] = bool(self.completion_button_pressed)
        values["success"] = bool(
            self.completion_button_pressed and self.sugar_goal_ready
        )
        return values


@dataclass(frozen=True, slots=True)
class StageEvalResult:
    """Observed versus prerequisite-qualified state for one TS03 stage."""

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
class TS03EvalResult:
    """Immutable TS03 profile and ordered-stage verdict."""

    profile_id: str
    target_sugar_count: int
    total_sugar_count: int
    success: bool
    terminal: bool
    failure_code: str | None
    stages: Mapping[str, StageEvalResult]
    details: Mapping[str, Any] = field(default_factory=dict)

    def to_diagnostics(self) -> dict[str, Any]:
        payload = dict(self.details or {})
        profile = profile_for_id(self.profile_id)
        stages: dict[str, dict[str, Any]] = {}
        required_stage = f"sugar_{self.target_sugar_count}"
        required_qualified = bool(self.stages[required_stage].qualified)
        over_added = self.failure_code == "over_add"

        for name in STAGE_ORDER:
            stage = self.stages[name]
            values = stage.to_dict()
            if name in SUGAR_STAGE_NAMES and not stage.applicable:
                if over_added:
                    effective_success = False
                    completion_mode = "blocked_by_over_add"
                elif required_qualified:
                    effective_success = True
                    completion_mode = "not_required"
                else:
                    effective_success = False
                    completion_mode = "waiting_for_required_sugar_goal"
                derived_from: str | None = required_stage
            else:
                effective_success = bool(stage.qualified)
                completion_mode = "physical" if stage.qualified else None
                derived_from = None
            values.update(
                {
                    "effective_success": effective_success,
                    "completion_mode": completion_mode,
                    "derived_from": derived_from,
                }
            )
            stages[name] = values

        payload.update(profile_metadata(profile))
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
                "effective_stage_checks": {
                    name: bool(values["effective_success"])
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
        """Project internal TS03 stages onto the public benchmark funnel."""

        stages: dict[str, dict[str, Any]] = {}
        for index, name in enumerate(SUGAR_STAGE_NAMES, start=1):
            applicable = index <= self.target_sugar_count
            internal = self.stages[name]
            stages[name] = {
                "applicable": applicable,
                "passed": bool(internal.qualified) if applicable else None,
                "first_pass_step": (
                    internal.qualified_step
                    if applicable and internal.qualified
                    else None
                ),
            }

        success = self.stages["success"]
        stages["success"] = {
            "applicable": True,
            "passed": bool(self.success),
            "first_pass_step": success.qualified_step if self.success else None,
        }
        return {
            "schema_version": "membench.report_stages/v1",
            "task_id": TS03_REPORT_PROFILE.task_id,
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


class TS03Evaluator:
    """Strict evaluator for one seed-resolved TS03 K/N profile."""

    def __init__(
        self,
        *,
        profile_id: str,
        target_sugar_count: int,
        total_sugar_count: int,
    ) -> None:
        profile = _profile_configuration(
            profile_id,
            target_sugar_count,
            total_sugar_count,
        )
        self.profile_id = profile.profile_id
        self.target_sugar_count = profile.target_sugar_count
        self.total_sugar_count = profile.total_sugar_count
        self._stages = {name: _MutableStage() for name in STAGE_ORDER}
        self._rejected: dict[str, list[dict[str, Any]]] = {
            name: [] for name in STAGE_ORDER
        }
        self._previous_raw = {name: False for name in STAGE_ORDER}
        self._last_step_id: int | None = None
        self._terminal_failure_code: str | None = None
        self._cached_result: TS03EvalResult | None = None

    @property
    def stage_prerequisites(self) -> dict[str, str | None]:
        return {
            "sugar_1": None,
            "sugar_2": "sugar_1",
            "button": f"sugar_{self.target_sugar_count}",
            "success": "button",
        }

    def reset(self, snapshot: TS03EvaluationSnapshot) -> TS03EvalResult:
        self._validate_configuration(snapshot)
        self._stages = {name: _MutableStage() for name in STAGE_ORDER}
        self._rejected = {name: [] for name in STAGE_ORDER}
        self._terminal_failure_code = "over_add" if snapshot.over_added else None
        raw = snapshot.raw_stage_values()
        if self._terminal_failure_code is None:
            for name, value in raw.items():
                if value:
                    self._mark_observed(name, snapshot.step_id)
                    self._reject(name, snapshot.step_id, "present_at_reset")
        self._previous_raw = raw
        self._last_step_id = snapshot.step_id
        self._cached_result = self._build_result(snapshot)
        return self._cached_result

    def observe(self, snapshot: TS03EvaluationSnapshot) -> TS03EvalResult:
        self._validate_configuration(snapshot)
        if self._last_step_id is None:
            return self.reset(snapshot)
        if snapshot.step_id == self._last_step_id:
            if self._cached_result is None:
                raise RuntimeError("TS03 evaluator has no cached result")
            return self._cached_result
        if snapshot.step_id < self._last_step_id:
            raise ValueError(
                "TS03 evaluation step moved backwards: "
                f"{snapshot.step_id} < {self._last_step_id}"
            )
        if self._cached_result is not None and self._cached_result.terminal:
            return self._cached_result

        raw = snapshot.raw_stage_values()
        if snapshot.over_added:
            self._terminal_failure_code = "over_add"
            self._previous_raw = raw
            self._last_step_id = snapshot.step_id
            self._cached_result = self._build_result(snapshot)
            return self._cached_result

        qualified_before = {
            name: self._stages[name].qualified for name in STAGE_ORDER
        }
        for name, value in raw.items():
            if value:
                self._mark_observed(name, snapshot.step_id)

        for index, name in enumerate(SUGAR_STAGE_NAMES, start=1):
            if index > self.target_sugar_count:
                continue
            if not raw[name] or self._previous_raw[name] or self._stages[name].qualified:
                continue
            prerequisite = self.stage_prerequisites[name]
            if prerequisite is None or qualified_before[prerequisite]:
                self._qualify(name, snapshot.step_id)
            else:
                self._reject(
                    name,
                    snapshot.step_id,
                    f"prerequisite_not_qualified:{prerequisite}",
                )

        if (
            raw["button"]
            and not self._previous_raw["button"]
            and not self._stages["button"].qualified
        ):
            prerequisite = self.stage_prerequisites["button"]
            if prerequisite is None or not qualified_before[prerequisite]:
                self._reject(
                    "button",
                    snapshot.step_id,
                    f"prerequisite_not_qualified:{prerequisite}",
                )
            elif not snapshot.sugar_goal_ready:
                self._reject("button", snapshot.step_id, "sugar_goal_not_ready")
            else:
                self._qualify("button", snapshot.step_id)
                self._mark_observed("success", snapshot.step_id)
                self._qualify("success", snapshot.step_id)

        self._previous_raw = raw
        self._last_step_id = snapshot.step_id
        self._cached_result = self._build_result(snapshot)
        return self._cached_result

    def result(self) -> TS03EvalResult:
        if self._cached_result is None:
            raise RuntimeError("TS03 evaluator has not been reset")
        return self._cached_result

    def _validate_configuration(self, snapshot: TS03EvaluationSnapshot) -> None:
        actual = (
            snapshot.profile_id,
            snapshot.target_sugar_count,
            snapshot.total_sugar_count,
        )
        expected = (
            self.profile_id,
            self.target_sugar_count,
            self.total_sugar_count,
        )
        if actual != expected:
            raise ValueError(
                "TS03 evaluation snapshot profile configuration changed within "
                f"an episode: {actual!r} != {expected!r}"
            )

    def _stage_applicable(self, name: str) -> bool:
        if name not in SUGAR_STAGE_NAMES:
            return True
        return SUGAR_STAGE_NAMES.index(name) < self.target_sugar_count

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
        applicable = self._stage_applicable(name)
        stage = self._stages[name]
        rejected = self._rejected[name]
        if not applicable:
            invalidated = False
            reason = "not_applicable_for_target_count"
        elif stage.qualified:
            invalidated = False
            reason = None
        elif rejected:
            invalidated = True
            reason = str(rejected[-1]["reason"])
        else:
            prerequisite = self.stage_prerequisites[name]
            if prerequisite is not None and not self._stages[prerequisite].qualified:
                invalidated = False
                reason = f"waiting_for_prerequisite:{prerequisite}"
            else:
                invalidated = False
                reason = "not_observed"
        return StageEvalResult(
            applicable=applicable,
            observed=stage.observed,
            qualified=stage.qualified,
            first_observed_step=stage.first_observed_step,
            qualified_step=stage.qualified_step,
            invalidated=invalidated,
            reason=reason,
        )

    def _build_result(self, snapshot: TS03EvaluationSnapshot) -> TS03EvalResult:
        stages = {name: self._stage_result(name) for name in STAGE_ORDER}
        details = dict(snapshot.details)
        evaluation = dict(details.get("evaluation") or {})
        evaluation.update(
            {
                "schema_version": 1,
                "step_id": int(snapshot.step_id),
                "stage_order": [
                    *SUGAR_STAGE_NAMES[: self.target_sugar_count],
                    "button",
                    "success",
                ],
                "reporting_stage_order": list(STAGE_ORDER),
                "stage_prerequisites": self.stage_prerequisites,
                "qualification_policy": "rising_edge_after_prerequisite",
                "raw_stage_checks": snapshot.raw_stage_values(),
                "rejected_stage_observations": {
                    name: list(values)
                    for name, values in self._rejected.items()
                    if values
                },
            }
        )
        details.update(profile_metadata(profile_for_id(self.profile_id)))
        details.update(
            {
                "evaluation": evaluation,
                "confirmed_sugar_count": int(snapshot.confirmed_sugar_count),
                "objects_released": bool(snapshot.objects_released),
                "completion_button_pressed": bool(
                    snapshot.completion_button_pressed
                ),
                "sugar_goal_ready": bool(snapshot.sugar_goal_ready),
            }
        )
        if self._terminal_failure_code == "over_add":
            details["over_add"] = {
                "profile_id": self.profile_id,
                "target_sugar_count": int(self.target_sugar_count),
                "confirmed_sugar_count": int(snapshot.confirmed_sugar_count),
                "step": int(snapshot.step_id),
            }
        success = bool(stages["success"].qualified)
        return TS03EvalResult(
            profile_id=self.profile_id,
            target_sugar_count=self.target_sugar_count,
            total_sugar_count=self.total_sugar_count,
            success=success,
            terminal=bool(success or self._terminal_failure_code is not None),
            failure_code=self._terminal_failure_code,
            stages=stages,
            details=details,
        )


def evaluate_initialized_runtime(runtime: Any) -> TS03EvalResult:
    """Evaluate one initialized task runtime through its task-local evaluator."""

    initialized = getattr(runtime, "initialized", None) or runtime
    metadata = getattr(initialized, "metadata", {}) or {}
    evaluator = metadata.get(TASK_EVALUATOR_METADATA_KEY)
    env = (
        getattr(initialized, "env", None)
        or getattr(runtime, "env", None)
        or initialized
    )
    if evaluator is None:
        evaluator = getattr(env, "_ts03_evaluator", None)
    if not isinstance(evaluator, TS03Evaluator):
        raise RuntimeError("TS03 runtime has no initialized task evaluator")
    snapshot_builder = getattr(env, "evaluation_snapshot", None)
    if not callable(snapshot_builder):
        raise AttributeError("TS03 environment has no evaluation_snapshot()")
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
    "SUGAR_STAGE_NAMES",
    "STAGE_ORDER",
    "TASK_EVALUATOR_METADATA_KEY",
    "StageEvalResult",
    "TS03EvalResult",
    "TS03EvaluationSnapshot",
    "TS03Evaluator",
    "diagnostics_for_runtime",
    "evaluate_initialized_runtime",
    "report_stages_for_runtime",
    "success_for_runtime",
]
