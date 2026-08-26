"""Authoritative TM02 benchmark evaluation state machine.

The physical observer owns faucet, fill, retrieval, cabinet, and button event
detection. This module qualifies those facts into the stable public funnel.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from bench.evaluation.report_profiles import TM02_REPORT_PROFILE
from bench.tasks.tm02.domain.contracts import StageEvent


STAGE_ORDER = TM02_REPORT_PROFILE.stage_ids
STAGE_PREREQUISITES: dict[str, str | None] = {
    "faucet_on": None,
    "pot_filled_and_can_retrieved": "faucet_on",
    "both_threads_completed": "pot_filled_and_can_retrieved",
    "success": "both_threads_completed",
}
REPORT_STAGE_ORDER = STAGE_ORDER
REPORT_STAGE_PROFILE_ID = TM02_REPORT_PROFILE.profile_id
TASK_EVALUATOR_METADATA_KEY = "task_evaluator"


@dataclass(frozen=True, slots=True)
class TM02EvaluationSnapshot:
    """One read-only view of TM02 semantic history and final-state checks."""

    step_id: int
    faucet_on_completed: bool = False
    pot_filled_and_can_retrieved: bool = False
    fill_pot_thread_completed: bool = False
    retrieve_can_thread_completed: bool = False
    completion_button_pressed: bool = False
    absorbing_failure: bool = False
    failure_code: str | None = None
    domain_success: bool = False
    event_steps: Mapping[str, int] = field(default_factory=dict)
    details: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        step_id = int(self.step_id)
        if step_id < 0:
            raise ValueError("TM02 evaluation step_id must be non-negative")
        event_steps: dict[str, int] = {}
        for event, raw_step in dict(self.event_steps or {}).items():
            event_step = int(raw_step)
            if event_step < 0:
                raise ValueError("TM02 event steps must be non-negative")
            event_steps[str(event)] = event_step

        faucet_on = bool(self.faucet_on_completed)
        combined_work = bool(self.pot_filled_and_can_retrieved)
        fill_done = bool(self.fill_pot_thread_completed)
        retrieve_done = bool(self.retrieve_can_thread_completed)
        button = bool(self.completion_button_pressed)
        failed = bool(self.absorbing_failure)
        failure_code = None if self.failure_code is None else str(self.failure_code)
        success = bool(self.domain_success)
        if failed and failure_code is None:
            raise ValueError("TM02 absorbing_failure requires failure_code")
        if success and not (
            faucet_on
            and combined_work
            and fill_done
            and retrieve_done
            and button
            and not failed
        ):
            raise ValueError(
                "TM02 domain_success requires both completed threads, a valid "
                "completion-button press, and no absorbing failure"
            )

        object.__setattr__(self, "step_id", step_id)
        object.__setattr__(self, "faucet_on_completed", faucet_on)
        object.__setattr__(self, "pot_filled_and_can_retrieved", combined_work)
        object.__setattr__(self, "fill_pot_thread_completed", fill_done)
        object.__setattr__(self, "retrieve_can_thread_completed", retrieve_done)
        object.__setattr__(self, "completion_button_pressed", button)
        object.__setattr__(self, "absorbing_failure", failed)
        object.__setattr__(self, "failure_code", failure_code)
        object.__setattr__(self, "domain_success", success)
        object.__setattr__(self, "event_steps", event_steps)
        object.__setattr__(self, "details", dict(self.details or {}))

    @property
    def both_threads_completed(self) -> bool:
        return bool(
            self.fill_pot_thread_completed
            and self.retrieve_can_thread_completed
        )

    def raw_stage_values(self) -> dict[str, bool]:
        return {
            "faucet_on": self.faucet_on_completed,
            "pot_filled_and_can_retrieved": self.pot_filled_and_can_retrieved,
            "both_threads_completed": self.both_threads_completed,
            "success": self.domain_success,
        }

    def evidence_step(self, stage_name: str) -> int:
        event_steps = self.event_steps
        if stage_name == "faucet_on":
            value = event_steps.get(StageEvent.A1.value)
        elif stage_name == "pot_filled_and_can_retrieved":
            value = event_steps.get(StageEvent.B1.value)
        elif stage_name == "both_threads_completed":
            a2 = event_steps.get(StageEvent.A2.value)
            b2 = event_steps.get(StageEvent.B2.value)
            value = max(a2, b2) if a2 is not None and b2 is not None else None
        elif stage_name == "success":
            value = event_steps.get(StageEvent.COMPLETION_BUTTON.value)
        else:
            raise KeyError(f"unknown TM02 evaluation stage {stage_name!r}")
        return self.step_id if value is None else int(value)


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
class TM02EvalResult:
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
            "task_id": TM02_REPORT_PROFILE.task_id,
            "profile_id": REPORT_STAGE_PROFILE_ID,
            "order": list(REPORT_STAGE_ORDER),
            "stages": stages,
        }


class TM02Evaluator:
    """History-aware evaluator for TM02's two interleavable threads."""

    def __init__(self) -> None:
        self._latched = {name: False for name in STAGE_ORDER[:-1]}
        self._first_observed_steps = {name: None for name in STAGE_ORDER}
        self._first_qualified_steps = {name: None for name in STAGE_ORDER}
        self._rejected: dict[str, list[dict[str, Any]]] = {
            name: [] for name in STAGE_ORDER
        }
        self._ever_success = False
        self._last_step_id: int | None = None
        self._cached_result: TM02EvalResult | None = None

    def reset(self, snapshot: TM02EvaluationSnapshot) -> TM02EvalResult:
        self._latched = {name: False for name in STAGE_ORDER[:-1]}
        self._first_observed_steps = {name: None for name in STAGE_ORDER}
        self._first_qualified_steps = {name: None for name in STAGE_ORDER}
        self._rejected = {name: [] for name in STAGE_ORDER}
        self._ever_success = False
        raw = snapshot.raw_stage_values()
        for name, value in raw.items():
            if value:
                observed_step = snapshot.evidence_step(name)
                self._first_observed_steps[name] = observed_step
                self._reject(name, observed_step, "present_at_reset")
        self._last_step_id = snapshot.step_id
        self._cached_result = self._build_result(snapshot, success_current=False)
        return self._cached_result

    def observe(self, snapshot: TM02EvaluationSnapshot) -> TM02EvalResult:
        if self._last_step_id is None:
            return self.reset(snapshot)
        if snapshot.step_id == self._last_step_id:
            if self._cached_result is None:
                raise RuntimeError("TM02 evaluator has no cached result")
            return self._cached_result
        if snapshot.step_id < self._last_step_id:
            raise ValueError(
                "TM02 evaluation step moved backwards: "
                f"{snapshot.step_id} < {self._last_step_id}"
            )

        raw = snapshot.raw_stage_values()
        for name in STAGE_ORDER[:-1]:
            if not raw[name] or self._latched[name]:
                continue
            evidence_step = snapshot.evidence_step(name)
            if self._first_observed_steps[name] is None:
                self._first_observed_steps[name] = evidence_step
            prerequisite = STAGE_PREREQUISITES[name]
            if prerequisite is not None and not self._latched[prerequisite]:
                self._reject(
                    name,
                    evidence_step,
                    f"prerequisite_not_qualified:{prerequisite}",
                )
                continue
            self._latched[name] = True
            self._first_qualified_steps[name] = evidence_step

        if raw["success"]:
            success_step = snapshot.evidence_step("success")
            if self._first_observed_steps["success"] is None:
                self._first_observed_steps["success"] = success_step
        success_current = bool(
            raw["success"]
            and self._latched["both_threads_completed"]
            and not snapshot.absorbing_failure
        )
        if raw["success"] and not success_current:
            self._reject(
                "success",
                snapshot.evidence_step("success"),
                "prerequisite_not_qualified:both_threads_completed",
            )
        if success_current:
            self._ever_success = True
            if self._first_qualified_steps["success"] is None:
                self._first_qualified_steps["success"] = snapshot.evidence_step(
                    "success"
                )

        self._last_step_id = snapshot.step_id
        self._cached_result = self._build_result(
            snapshot,
            success_current=success_current,
        )
        return self._cached_result

    def result(self) -> TM02EvalResult:
        if self._cached_result is None:
            raise RuntimeError("TM02 evaluator has not been reset")
        return self._cached_result

    def _reject(self, name: str, step_id: int, reason: str) -> None:
        entry = {"step": int(step_id), "reason": str(reason)}
        if entry not in self._rejected[name]:
            self._rejected[name].append(entry)

    def _build_result(
        self,
        snapshot: TM02EvaluationSnapshot,
        *,
        success_current: bool,
    ) -> TM02EvalResult:
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
                    "ordered_semantic_events_with_interleavable_thread_completion_"
                    "and_final_state_success"
                ),
                "raw_stage_checks": snapshot.raw_stage_values(),
                "semantic_event_steps": dict(snapshot.event_steps),
                "rejected_stage_observations": {
                    name: list(values)
                    for name, values in self._rejected.items()
                    if values
                },
            }
        )
        details.update(
            {
                "evaluation": evaluation,
                "faucet_on_completed": snapshot.faucet_on_completed,
                "pot_filled_and_can_retrieved": (
                    snapshot.pot_filled_and_can_retrieved
                ),
                "fill_pot_thread_completed": snapshot.fill_pot_thread_completed,
                "retrieve_can_thread_completed": (
                    snapshot.retrieve_can_thread_completed
                ),
                "completion_button_pressed": snapshot.completion_button_pressed,
                "absorbing_failure": snapshot.absorbing_failure,
                "domain_success": snapshot.domain_success,
            }
        )
        success = bool(success_current)
        return TM02EvalResult(
            success=success,
            terminal=bool(success or snapshot.absorbing_failure),
            failure_code=None if success else snapshot.failure_code,
            stages=stages,
            details=details,
        )

    def _stage_result(
        self,
        name: str,
        *,
        snapshot: TM02EvaluationSnapshot,
        success_current: bool,
    ) -> StageEvalResult:
        if name == "success":
            qualified = bool(success_current)
            invalidated = bool(
                (self._ever_success and not qualified)
                or snapshot.absorbing_failure
            )
        else:
            qualified = bool(self._latched[name])
            invalidated = False
        rejected = self._rejected[name]
        prerequisite = STAGE_PREREQUISITES[name]
        if qualified:
            reason = None
        elif name == "success" and snapshot.absorbing_failure:
            reason = snapshot.failure_code
        elif name == "success" and self._ever_success:
            reason = "final_state_regressed"
        elif rejected:
            invalidated = True
            reason = str(rejected[-1]["reason"])
        elif prerequisite is not None and not self._latched.get(
            prerequisite, False
        ):
            reason = f"waiting_for_prerequisite:{prerequisite}"
        else:
            reason = "not_observed"
        return StageEvalResult(
            applicable=True,
            observed=self._first_observed_steps[name] is not None,
            qualified=qualified,
            first_observed_step=self._first_observed_steps[name],
            qualified_step=self._first_qualified_steps[name],
            invalidated=invalidated,
            reason=reason,
        )


def evaluate_initialized_runtime(runtime: Any) -> TM02EvalResult:
    initialized = getattr(runtime, "initialized", None) or runtime
    metadata = getattr(initialized, "metadata", {}) or {}
    evaluator = metadata.get(TASK_EVALUATOR_METADATA_KEY)
    env = (
        getattr(initialized, "env", None)
        or getattr(runtime, "env", None)
        or initialized
    )
    if evaluator is None:
        evaluator = getattr(env, "_tm02_evaluator", None)
    if not isinstance(evaluator, TM02Evaluator):
        raise RuntimeError("TM02 runtime has no initialized task evaluator")
    snapshot_builder = getattr(env, "evaluation_snapshot", None)
    if not callable(snapshot_builder):
        raise AttributeError("TM02 environment has no evaluation_snapshot()")
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
    "TM02EvalResult",
    "TM02EvaluationSnapshot",
    "TM02Evaluator",
    "diagnostics_for_runtime",
    "evaluate_initialized_runtime",
    "report_stages_for_runtime",
    "success_for_runtime",
]
