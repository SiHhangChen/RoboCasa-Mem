"""WA-05 integration for stage-conditioned policy evaluation."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from bench.evaluation.stage_rollout import StageSpec
from bench.tasks.wa05.evaluation import STAGE_ORDER, WA05EvalResult, evaluate_initialized_runtime


DEFAULT_STAGE_BUDGETS = {
    "cup_filled": 1200,
    "cup_staged": 800,
    "success": 800,
}


def build_stage_specs(
    budgets: Mapping[str, int] | None = None,
) -> tuple[StageSpec, ...]:
    configured = dict(DEFAULT_STAGE_BUDGETS)
    configured.update({str(key): int(value) for key, value in dict(budgets or {}).items()})
    unknown = sorted(set(configured) - set(STAGE_ORDER))
    if unknown:
        raise ValueError(f"unknown WA-05 stage budgets: {unknown}")
    return tuple(
        StageSpec(
            id=stage_id,
            completion_predicate=_completion(stage_id),
            failure_predicate=_failed,
            model_step_budget=configured[stage_id],
        )
        for stage_id in STAGE_ORDER
    )


def privileged_stage_state(runtime: Any) -> WA05EvalResult:
    shared = getattr(runtime, "metadata", {}).get("shared_runtime")
    return evaluate_initialized_runtime(shared or runtime)


def stage_completed(state: WA05EvalResult, stage_id: str) -> bool:
    try:
        return bool(state.stages[str(stage_id)].qualified)
    except KeyError as exc:
        raise KeyError(f"unknown WA-05 stage {stage_id!r}; expected one of {STAGE_ORDER}") from exc


def _completion(stage_id: str):
    return lambda state: stage_completed(state, stage_id)


def _failed(state: WA05EvalResult) -> bool:
    return bool(state.failure_code)


__all__ = [
    "DEFAULT_STAGE_BUDGETS",
    "build_stage_specs",
    "privileged_stage_state",
    "stage_completed",
]
