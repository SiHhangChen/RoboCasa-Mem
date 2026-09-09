"""WA-05 integration for stage-conditioned policy evaluation."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from bench.evaluation.stage_integration import StageEvaluationIntegration
from bench.evaluation.stage_rollout import StageSpec
from bench.tasks.wa05.evaluation import STAGE_ORDER, WA05EvalResult, evaluate_initialized_runtime


DEFAULT_STAGE_BUDGETS = {
    "cup_filled": 1200,
    "cup_staged": 800,
    "success": 800,
}


def build_stage_specs(
    budgets: Mapping[str, int] | None = None,
    target_stage: str | None = None,
) -> tuple[StageSpec, ...]:
    configured = dict(DEFAULT_STAGE_BUDGETS)
    configured.update({str(key): int(value) for key, value in dict(budgets or {}).items()})
    unknown = sorted(set(configured) - set(STAGE_ORDER))
    if unknown:
        raise ValueError(f"unknown WA-05 stage budgets: {unknown}")
    target = STAGE_ORDER[-1] if target_stage is None else str(target_stage)
    try:
        target_index = STAGE_ORDER.index(target)
    except ValueError as exc:
        raise KeyError(
            f"unknown WA-05 target stage {target!r}; expected one of {STAGE_ORDER}"
        ) from exc
    return tuple(
        StageSpec(
            id=stage_id,
            completion_predicate=_completion(stage_id),
            failure_predicate=_failed,
            model_step_budget=configured[stage_id],
        )
        for stage_id in STAGE_ORDER[: target_index + 1]
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


STAGE_EVALUATION = StageEvaluationIntegration(
    task_id="WA-05",
    target_stage_ids=STAGE_ORDER,
    capture_stage_ids=STAGE_ORDER,
    prefix_stage_by_target={
        "cup_filled": None,
        "cup_staged": "cup_filled",
        "success": "cup_staged",
    },
    completion_stage_by_target={stage_id: stage_id for stage_id in STAGE_ORDER},
    spec_builder=build_stage_specs,
    privileged_state_builder=privileged_stage_state,
    stage_completion_checker=stage_completed,
)


__all__ = [
    "DEFAULT_STAGE_BUDGETS",
    "STAGE_EVALUATION",
    "build_stage_specs",
    "privileged_stage_state",
    "stage_completed",
]
