"""WR-07 report/evaluation adapter for single-item color-matched transport."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from bench.evaluation.report_profiles import WR07_REPORT_PROFILE
from bench.tasks.wr07.domain.contracts import TARGET_SLOTS

STAGE_ORDER = WR07_REPORT_PROFILE.stage_ids
REPORT_STAGE_ORDER = STAGE_ORDER
REPORT_STAGE_PROFILE_ID = WR07_REPORT_PROFILE.profile_id
TASK_EVALUATOR_METADATA_KEY = "task_evaluator"

@dataclass(frozen=True, slots=True)
class WR07EvaluationSnapshot:
    """Current physical evidence for policies, legacy events for recordings.

    On the policy path, ``item_picked`` means a grasped, lifted carton and
    ``item_placed`` means a released, supported carton, not a semantic latch.
    Expert-only memory/protocol fields must never gate policy evaluation.
    """

    step_id: int
    source_slot: str | None = None
    source_color: str | None = None
    initial_source_verified: bool = False
    item_picked: bool = False
    source_color_memory_latched: bool = False
    target_slot_attempted: str | None = None
    item_placed: bool = False
    target_color_matches: bool = False
    protocol_violation: bool = False
    domain_success: bool = False
    require_expert_protocol: bool = False
    details: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if int(self.step_id) < 0:
            raise ValueError("WR-07 evaluation step_id must be non-negative")
        object.__setattr__(self, "step_id", int(self.step_id))
        object.__setattr__(self, "details", dict(self.details or {}))

    def raw_stage_values(self) -> dict[str, bool]:
        matched = bool(self.item_placed and self.target_color_matches)
        expert_ready = not self.require_expert_protocol or bool(
            self.domain_success
            and self.source_color_memory_latched
            and not self.protocol_violation
        )
        return {
            "item_picked": bool(self.item_picked),
            "item_placed": bool(self.item_placed),
            "target_plate_matched": matched,
            "success": bool(matched and expert_ready),
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
            "applicable": self.applicable, "observed": self.observed,
            "qualified": self.qualified, "first_observed_step": self.first_observed_step,
            "qualified_step": self.qualified_step, "invalidated": self.invalidated,
            "reason": self.reason,
        }

@dataclass(frozen=True, slots=True)
class WR07EvalResult:
    success: bool
    terminal: bool
    failure_code: str | None
    stages: Mapping[str, StageEvalResult]
    details: Mapping[str, Any] = field(default_factory=dict)
    stop_execution: bool = False
    def to_diagnostics(self) -> dict[str, Any]:
        stages = {name: self.stages[name].to_dict() for name in STAGE_ORDER}
        payload = dict(self.details)
        payload.update({
            "success": self.success, "terminal": self.terminal,
            "stop_execution": self.stop_execution,
            "failure_code": self.failure_code, "stages": stages,
            "stage_checks": {name: value["qualified"] for name, value in stages.items()},
        })
        return payload
    def to_report_stages(self) -> dict[str, Any]:
        return {
            "schema_version": "membench.report_stages/v1",
            "task_id": WR07_REPORT_PROFILE.task_id,
            "profile_id": REPORT_STAGE_PROFILE_ID,
            "order": list(STAGE_ORDER),
            "stages": {
                name: {
                    "applicable": True,
                    "passed": self.stages[name].qualified,
                    "first_pass_step": self.stages[name].qualified_step if self.stages[name].qualified else None,
                }
                for name in STAGE_ORDER
            },
        }

class WR07Evaluator:
    def __init__(self, *, placement_dwell_steps: int = 2) -> None:
        if int(placement_dwell_steps) != placement_dwell_steps or placement_dwell_steps < 1:
            raise ValueError("WR-07 placement_dwell_steps must be a positive integer")
        self.placement_dwell_steps = int(placement_dwell_steps)
        self._latched = {name: False for name in STAGE_ORDER[:-1]}
        self._first: dict[str, int | None] = {name: None for name in STAGE_ORDER}
        self._qualified: dict[str, int | None] = {name: None for name in STAGE_ORDER}
        self._previous = {name: False for name in STAGE_ORDER}
        self._last: int | None = None
        self._cached: WR07EvalResult | None = None
        self._placement_run = 0
        self._match_run = 0
        self._wrong_target_run = 0
        self._wrong_target_slot: str | None = None
        self._failure_code: str | None = None
        self._finished = False

    def reset(self, snapshot: WR07EvaluationSnapshot) -> WR07EvalResult:
        self._latched = {name: False for name in STAGE_ORDER[:-1]}
        self._first = {name: None for name in STAGE_ORDER}
        self._qualified = {name: None for name in STAGE_ORDER}
        self._previous = snapshot.raw_stage_values()
        self._placement_run = 0
        self._match_run = 0
        self._wrong_target_run = 0
        self._wrong_target_slot = None
        self._failure_code = None
        self._finished = False
        self._last = snapshot.step_id
        self._cached = self._build(snapshot, False)
        return self._cached

    def observe(self, snapshot: WR07EvaluationSnapshot) -> WR07EvalResult:
        if self._last is None:
            return self.reset(snapshot)
        if snapshot.step_id < self._last:
            raise ValueError("WR-07 evaluation step moved backwards")
        if self._finished or snapshot.step_id == self._last:
            return self.result()
        raw = snapshot.raw_stage_values()
        for name, value in raw.items():
            if value and self._first[name] is None:
                self._first[name] = snapshot.step_id
        # A carton already resting at reset is not a putdown event. Policies
        # must first show a grasp-and-lift observation, then a later release.
        picked_before = self._latched["item_picked"]
        if (
            not picked_before
            and raw["item_picked"]
            and not self._previous["item_picked"]
            and (snapshot.require_expert_protocol or not raw["item_placed"])
        ):
            self._latched["item_picked"] = True
            self._qualified["item_picked"] = snapshot.step_id

        if snapshot.step_id != self._last + 1:
            self._placement_run = self._match_run = self._wrong_target_run = 0
        self._placement_run = (
            self._placement_run + 1 if picked_before and raw["item_placed"] else 0
        )
        self._match_run = (
            self._match_run + 1 if picked_before and raw["target_plate_matched"] else 0
        )
        # Keep recording's existing event timing; debounce physical policy
        # placement over two distinct control steps (0.1 s at 20 Hz).
        dwell = 1 if snapshot.require_expert_protocol else self.placement_dwell_steps
        if self._placement_run >= dwell and not self._latched["item_placed"]:
            self._latched["item_placed"] = True
            self._qualified["item_placed"] = snapshot.step_id
        if (
            self._latched["item_placed"]
            and self._match_run >= dwell
            and not self._latched["target_plate_matched"]
        ):
            self._latched["target_plate_matched"] = True
            self._qualified["target_plate_matched"] = snapshot.step_id

        # Confirm an actual wrong destination placement, not an airborne or
        # held carton passing over a tray. Require the same wrong tray for
        # the placement dwell; source/counter contacts are not destinations.
        wrong_slot = (
            snapshot.target_slot_attempted
            if (
                not snapshot.require_expert_protocol
                and picked_before
                and raw["item_placed"]
                and snapshot.target_slot_attempted in TARGET_SLOTS
                and not snapshot.target_color_matches
            )
            else None
        )
        self._wrong_target_run = (
            0 if wrong_slot is None else (
                self._wrong_target_run + 1 if wrong_slot == self._wrong_target_slot else 1
            )
        )
        self._wrong_target_slot = wrong_slot
        if self._wrong_target_run >= dwell:
            self._failure_code = "wrong_target_placement"

        # Policy success and wrong-target failure are final verdicts. The
        # rollout stops at that step; later observations cannot revise it.
        success = bool(
            self._failure_code is None
            and raw["success"]
            and all(self._latched.values())
            and self._match_run >= dwell
        )
        if success and self._qualified["success"] is None:
            self._qualified["success"] = snapshot.step_id
        self._previous, self._last = raw, snapshot.step_id
        self._cached = self._build(snapshot, success)
        # Expert recording retains its live protocol checks through the
        # remaining verification/return-arm operations.
        self._finished = self._cached.stop_execution
        return self._cached

    def _build(self, snapshot: WR07EvaluationSnapshot, success: bool) -> WR07EvalResult:
        raw = snapshot.raw_stage_values()
        stages = {}
        for name in STAGE_ORDER:
            qualified = success if name == "success" else self._latched[name]
            invalidated = bool(self._failure_code and not qualified)
            reason = self._failure_code if invalidated else (
                None if qualified else (
                    "not_observed" if not raw[name] else "waiting_for_qualified_event"
                )
            )
            stages[name] = StageEvalResult(
                True, self._first[name] is not None, qualified,
                self._first[name], self._qualified[name], invalidated, reason,
            )
        details = dict(snapshot.details)
        details["evaluation"] = {
            "profile_id": REPORT_STAGE_PROFILE_ID,
            "mode": "expert_recording" if snapshot.require_expert_protocol else "policy_physical",
            "placement_dwell_steps": (
                1 if snapshot.require_expert_protocol else self.placement_dwell_steps
            ),
            "placement_run": self._placement_run,
            "matching_plate_run": self._match_run,
            "wrong_target_slot": self._wrong_target_slot,
            "wrong_target_run": self._wrong_target_run,
            "terminal_step": snapshot.step_id if success or self._failure_code else None,
        }
        terminal = bool(success or self._failure_code)
        return WR07EvalResult(
            success, terminal, self._failure_code, stages, details,
            stop_execution=bool(not snapshot.require_expert_protocol and terminal),
        )

    def result(self) -> WR07EvalResult:
        if self._cached is None:
            raise RuntimeError("WR-07 evaluator has not been reset")
        return self._cached

def evaluate_initialized_runtime(runtime: Any) -> WR07EvalResult:
    initialized = getattr(runtime, "initialized", None) or runtime
    env = getattr(initialized, "env", None) or getattr(runtime, "env", None) or initialized
    metadata = getattr(initialized, "metadata", {}) or {}
    evaluator = metadata.get(TASK_EVALUATOR_METADATA_KEY) or getattr(env, "_wr07_evaluator", None)
    if not isinstance(evaluator, WR07Evaluator):
        raise RuntimeError("WR-07 runtime has no initialized task evaluator")
    return evaluator.observe(env.evaluation_snapshot())

def success_for_runtime(runtime: Any, info: Mapping[str, Any] | None = None) -> bool:
    del info
    return evaluate_initialized_runtime(runtime).success

def diagnostics_for_runtime(runtime: Any) -> Mapping[str, Any]:
    return evaluate_initialized_runtime(runtime).to_diagnostics()

def report_stages_for_runtime(runtime: Any) -> Mapping[str, Any]:
    return evaluate_initialized_runtime(runtime).to_report_stages()

__all__ = [
    "REPORT_STAGE_ORDER", "REPORT_STAGE_PROFILE_ID", "STAGE_ORDER",
    "TASK_EVALUATOR_METADATA_KEY", "StageEvalResult", "WR07EvalResult",
    "WR07EvaluationSnapshot", "WR07Evaluator", "diagnostics_for_runtime",
    "evaluate_initialized_runtime", "report_stages_for_runtime", "success_for_runtime",
]
