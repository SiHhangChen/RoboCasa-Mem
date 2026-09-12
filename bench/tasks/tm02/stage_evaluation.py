"""TM02 integration for stage-conditioned policy evaluation."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from bench.evaluation.report_profiles import TM02_REPORT_PROFILE_V1
from bench.evaluation.stage_integration import StageEvaluationIntegration
from bench.evaluation.stage_rollout import StageSpec
from bench.tasks.tm02.evaluation import (
    STAGE_ORDER,
    TM02EvalResult,
    evaluate_initialized_runtime,
)


DEFAULT_STAGE_BUDGETS = {
    # 2 x P95 of each report-v2 interval in the 50-seed frozen expert set.
    "faucet_on": 220,
    "pot_filled_and_can_retrieved": 3144,
    "one_thread_completed": 1644,
    "both_threads_completed": 1594,
    "success": 868,
}

LEGACY_STAGE_ORDER = TM02_REPORT_PROFILE_V1.stage_ids
LEGACY_STAGE_BUDGETS = {
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
    return _build_ordered_stage_specs(
        STAGE_ORDER, DEFAULT_STAGE_BUDGETS, budgets, target_stage,
    )


def build_legacy_stage_specs(
    budgets: Mapping[str, int] | None = None,
    target_stage: str | None = None,
) -> tuple[StageSpec, ...]:
    """Keep explicitly selected v1 stage campaigns on their original boundaries."""

    return _build_ordered_stage_specs(
        LEGACY_STAGE_ORDER, LEGACY_STAGE_BUDGETS, budgets, target_stage,
    )


def _build_ordered_stage_specs(
    order: tuple[str, ...],
    defaults: Mapping[str, int],
    budgets: Mapping[str, int] | None,
    target_stage: str | None,
) -> tuple[StageSpec, ...]:
    configured = dict(defaults)
    configured.update({str(key): int(value) for key, value in dict(budgets or {}).items()})
    unknown = sorted(set(configured) - set(order))
    if unknown:
        raise ValueError(f"unknown TM02 stage budgets: {unknown}")
    target = order[-1] if target_stage is None else str(target_stage)
    try:
        target_index = order.index(target)
    except ValueError as exc:
        raise KeyError(
            f"unknown TM02 target stage {target!r}; expected one of {order}"
        ) from exc
    return tuple(
        StageSpec(
            id=stage_id,
            completion_predicate=_completion(stage_id),
            failure_predicate=_failed,
            model_step_budget=configured[stage_id],
        )
        for stage_id in order[: target_index + 1]
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
        "one_thread_completed": "pot_filled_and_can_retrieved",
        "both_threads_completed": "one_thread_completed",
        "success": "both_threads_completed",
    },
    completion_stage_by_target={stage_id: stage_id for stage_id in STAGE_ORDER},
    spec_builder=build_stage_specs,
    privileged_state_builder=privileged_stage_state,
    stage_completion_checker=stage_completed,
)

LEGACY_STAGE_EVALUATION = StageEvaluationIntegration(
    task_id="TM02",
    target_stage_ids=LEGACY_STAGE_ORDER,
    capture_stage_ids=LEGACY_STAGE_ORDER,
    prefix_stage_by_target={
        "faucet_on": None,
        "pot_filled_and_can_retrieved": "faucet_on",
        "both_threads_completed": "pot_filled_and_can_retrieved",
        "success": "both_threads_completed",
    },
    completion_stage_by_target={stage_id: stage_id for stage_id in LEGACY_STAGE_ORDER},
    spec_builder=build_legacy_stage_specs,
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
    "LEGACY_STAGE_EVALUATION",
    "LEGACY_STAGE_ORDER",
    "STAGE_EVALUATION",
    "build_b1_diagnostic_stage_specs",
    "build_legacy_stage_specs",
    "build_stage_specs",
    "privileged_stage_state",
    "stage_completed",
]
