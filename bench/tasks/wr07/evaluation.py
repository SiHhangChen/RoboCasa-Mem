"""WR-07 report/evaluation adapter for single-item color-matched transport."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from bench.evaluation.report_profiles import WR07_REPORT_PROFILE

STAGE_ORDER = WR07_REPORT_PROFILE.stage_ids
REPORT_STAGE_ORDER = STAGE_ORDER
REPORT_STAGE_PROFILE_ID = WR07_REPORT_PROFILE.profile_id
TASK_EVALUATOR_METADATA_KEY = "task_evaluator"

@dataclass(frozen=True, slots=True)
class WR07EvaluationSnapshot:
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
    details: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if int(self.step_id) < 0:
            raise ValueError("WR-07 evaluation step_id must be non-negative")
        object.__setattr__(self, "step_id", int(self.step_id))
        object.__setattr__(self, "details", dict(self.details or {}))

    def raw_stage_values(self) -> dict[str, bool]:
        return {
            "item_picked": bool(self.item_picked),
            "memory_color_latched": bool(self.source_color_memory_latched),
            "target_plate_matched": bool(self.item_placed and self.target_color_matches),
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
    def to_diagnostics(self) -> dict[str, Any]:
        stages = {name: self.stages[name].to_dict() for name in STAGE_ORDER}
        payload = dict(self.details)
        payload.update({
            "success": self.success, "terminal": self.terminal,
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
    def __init__(self) -> None:
        self._latched = {name: False for name in STAGE_ORDER[:-1]}
        self._first: dict[str, int | None] = {name: None for name in STAGE_ORDER}
        self._qualified: dict[str, int | None] = {name: None for name in STAGE_ORDER}
        self._previous = {name: False for name in STAGE_ORDER}
        self._last: int | None = None
        self._cached: WR07EvalResult | None = None

    def reset(self, snapshot: WR07EvaluationSnapshot) -> WR07EvalResult:
        self._latched = {name: False for name in STAGE_ORDER[:-1]}
        self._first = {name: None for name in STAGE_ORDER}
        self._qualified = {name: None for name in STAGE_ORDER}
        self._previous = snapshot.raw_stage_values()
        self._last = snapshot.step_id
        self._cached = self._build(snapshot, False)
        return self._cached

    def observe(self, snapshot: WR07EvaluationSnapshot) -> WR07EvalResult:
        if self._last is None:
            return self.reset(snapshot)
        if snapshot.step_id == self._last:
            return self._cached  # type: ignore[return-value]
        if snapshot.step_id < self._last:
            raise ValueError("WR-07 evaluation step moved backwards")
        raw = snapshot.raw_stage_values()
        for name, value in raw.items():
            if value and self._first[name] is None:
                self._first[name] = snapshot.step_id
        for name in STAGE_ORDER[:-1]:
            if self._latched[name] or not raw[name] or self._previous[name]:
                continue
            self._latched[name] = True
            self._qualified[name] = snapshot.step_id
        success = bool(raw["success"] and all(self._latched.values()))
        if success and self._qualified["success"] is None:
            self._qualified["success"] = snapshot.step_id
        self._previous, self._last = raw, snapshot.step_id
        self._cached = self._build(snapshot, success)
        return self._cached

    def _build(self, snapshot: WR07EvaluationSnapshot, success: bool) -> WR07EvalResult:
        raw = snapshot.raw_stage_values()
        stages = {}
        for name in STAGE_ORDER:
            qualified = success if name == "success" else self._latched[name]
            stages[name] = StageEvalResult(
                True, self._first[name] is not None, qualified,
                self._first[name], self._qualified[name], False,
                None if qualified else ("not_observed" if not raw[name] else "waiting_for_qualified_event"),
            )
        return WR07EvalResult(success, success, None, stages, dict(snapshot.details))

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
