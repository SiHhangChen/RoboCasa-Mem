"""TM02 integration for stage-conditioned policy evaluation."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from bench.evaluation.stage_integration import StageEvaluationIntegration
from bench.evaluation.stage_rollout import StageSpec
from bench.tasks.tm02.evaluation import (
    STAGE_ORDER,
    TM02EvalResult,
    evaluate_initialized_runtime,
)


DEFAULT_STAGE_BUDGETS = {
    "faucet_on": 800,
    "pot_filled_and_can_retrieved": 3000,
    "both_threads_completed": 1800,
    "success": 800,
}

B1_DIAGNOSTIC_STAGE_ORDER = (
    "faucet_on",
    "pot_filled",
    "cabinet_opened",
    "can_removed_from_cabinet",
    "can_released_in_place_region",
)

B1_DIAGNOSTIC_STAGE_BUDGETS = {
    "faucet_on": 220,
    "pot_filled": 400,
    "cabinet_opened": 1200,
    "can_removed_from_cabinet": 1100,
    "can_released_in_place_region": 1100,
}

B1_DIAGNOSTIC_PATHS = {
    "faucet_on": ("faucet_on",),
    "pot_filled": ("faucet_on", "pot_filled"),
    "cabinet_opened": ("faucet_on", "cabinet_opened"),
    "can_removed_from_cabinet": (
        "faucet_on",
        "cabinet_opened",
        "can_removed_from_cabinet",
    ),
    "can_released_in_place_region": (
        "faucet_on",
        "cabinet_opened",
        "can_removed_from_cabinet",
        "can_released_in_place_region",
    ),
}


def build_stage_specs(
    budgets: Mapping[str, int] | None = None,
    target_stage: str | None = None,
) -> tuple[StageSpec, ...]:
    configured = dict(DEFAULT_STAGE_BUDGETS)
    configured.update({str(key): int(value) for key, value in dict(budgets or {}).items()})
    unknown = sorted(set(configured) - set(STAGE_ORDER))
    if unknown:
        raise ValueError(f"unknown TM02 stage budgets: {unknown}")
    target = STAGE_ORDER[-1] if target_stage is None else str(target_stage)
    try:
        target_index = STAGE_ORDER.index(target)
    except ValueError as exc:
        raise KeyError(
            f"unknown TM02 target stage {target!r}; expected one of {STAGE_ORDER}"
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


def build_b1_diagnostic_stage_specs(
    budgets: Mapping[str, int] | None = None,
    target_stage: str | None = None,
) -> tuple[StageSpec, ...]:
    configured = dict(B1_DIAGNOSTIC_STAGE_BUDGETS)
    configured.update({str(key): int(value) for key, value in dict(budgets or {}).items()})
    unknown = sorted(set(configured) - set(B1_DIAGNOSTIC_STAGE_ORDER))
    if unknown:
        raise ValueError(f"unknown TM02 B1 diagnostic stage budgets: {unknown}")
    target = (
        B1_DIAGNOSTIC_STAGE_ORDER[-1]
        if target_stage is None
        else str(target_stage)
    )
    try:
        path = B1_DIAGNOSTIC_PATHS[target]
    except KeyError as exc:
        raise KeyError(
            f"unknown TM02 B1 diagnostic target {target!r}; "
            f"expected one of {B1_DIAGNOSTIC_STAGE_ORDER}"
        ) from exc
    return tuple(
        StageSpec(
            id=stage_id,
            completion_predicate=_completion(stage_id),
            failure_predicate=_failed,
            model_step_budget=configured[stage_id],
        )
        for stage_id in path
    )


def privileged_stage_state(runtime: Any) -> TM02EvalResult:
    shared = getattr(runtime, "metadata", {}).get("shared_runtime")
    return evaluate_initialized_runtime(shared or runtime)


def stage_completed(state: TM02EvalResult, stage_id: str) -> bool:
    diagnostic = _diagnostic_stage_value(state, str(stage_id))
    if diagnostic is not None:
        return diagnostic
    try:
        return bool(state.stages[str(stage_id)].qualified)
    except KeyError as exc:
        raise KeyError(
            f"unknown TM02 stage {stage_id!r}; expected one of "
            f"{(*STAGE_ORDER, *B1_DIAGNOSTIC_STAGE_ORDER[1:])}"
        ) from exc


def _diagnostic_stage_value(
    state: TM02EvalResult,
    stage_id: str,
) -> bool | None:
    if stage_id not in B1_DIAGNOSTIC_STAGE_ORDER[1:]:
        return None
    details = dict(state.details or {})
    semantic = dict(details.get("semantic") or {})
    physical = dict(details.get("physical") or {})
    if stage_id == "pot_filled":
        return bool(semantic.get("pot_fill_ready") and physical.get("pot_filled"))
    if stage_id == "cabinet_opened":
        return bool(semantic.get("cabinet_opened"))
    if stage_id == "can_removed_from_cabinet":
        return bool(semantic.get("can_removed_from_cabinet"))
    return bool(
        semantic.get("can_removed_from_cabinet")
        and physical.get("can_in_place_region")
        and physical.get("can_released")
    )


def _completion(stage_id: str):
    return lambda state: stage_completed(state, stage_id)


def _failed(state: TM02EvalResult) -> bool:
    return bool(state.failure_code)


STAGE_EVALUATION = StageEvaluationIntegration(
    task_id="TM02",
    target_stage_ids=STAGE_ORDER,
    capture_stage_ids=STAGE_ORDER,
    prefix_stage_by_target={
        "faucet_on": None,
        "pot_filled_and_can_retrieved": "faucet_on",
        "both_threads_completed": "pot_filled_and_can_retrieved",
        "success": "both_threads_completed",
    },
    completion_stage_by_target={stage_id: stage_id for stage_id in STAGE_ORDER},
    spec_builder=build_stage_specs,
    privileged_state_builder=privileged_stage_state,
    stage_completion_checker=stage_completed,
)

B1_DIAGNOSTIC_STAGE_EVALUATION = StageEvaluationIntegration(
    task_id="TM02",
    target_stage_ids=B1_DIAGNOSTIC_STAGE_ORDER,
    capture_stage_ids=B1_DIAGNOSTIC_STAGE_ORDER,
    prefix_stage_by_target={
        "faucet_on": None,
        "pot_filled": "faucet_on",
        "cabinet_opened": "faucet_on",
        "can_removed_from_cabinet": "cabinet_opened",
        "can_released_in_place_region": "can_removed_from_cabinet",
    },
    completion_stage_by_target={
        stage_id: stage_id for stage_id in B1_DIAGNOSTIC_STAGE_ORDER
    },
    spec_builder=build_b1_diagnostic_stage_specs,
    privileged_state_builder=privileged_stage_state,
    stage_completion_checker=stage_completed,
)


__all__ = [
    "B1_DIAGNOSTIC_PATHS",
    "B1_DIAGNOSTIC_STAGE_BUDGETS",
    "B1_DIAGNOSTIC_STAGE_EVALUATION",
    "B1_DIAGNOSTIC_STAGE_ORDER",
    "DEFAULT_STAGE_BUDGETS",
    "STAGE_EVALUATION",
    "build_b1_diagnostic_stage_specs",
    "build_stage_specs",
    "privileged_stage_state",
    "stage_completed",
]
