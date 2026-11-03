"""TS02 integration for stage-conditioned policy evaluation."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from bench.evaluation.stage_integration import StageEvaluationIntegration
from bench.evaluation.stage_rollout import StageSpec
from bench.tasks.ts02.evaluation import (
    STAGE_ORDER,
    TS02EvalResult,
    evaluate_initialized_runtime,
)


BEFORE_TARGET_DRAWER = "before_target_drawer"
TARGET_DRAWER_RETURNED = "target_drawer_returned"
TARGET_STAGE_ORDER = STAGE_ORDER
CAPTURE_STAGE_ORDER = (
    BEFORE_TARGET_DRAWER,
    *STAGE_ORDER,
    TARGET_DRAWER_RETURNED,
)

_PREFIX_STAGE_BY_TARGET = {
    "one_drawer_checked": None,
    "target_found": BEFORE_TARGET_DRAWER,
    "search_stopped_after_target": BEFORE_TARGET_DRAWER,
    "button_pressed": TARGET_DRAWER_RETURNED,
    "correct_press_count": TARGET_DRAWER_RETURNED,
}

DEFAULT_STAGE_BUDGETS = {
    "one_drawer_checked": 1200,
    "target_found": 1200,
    "search_stopped_after_target": 1200,
    "button_pressed": 800,
    "correct_press_count": 1200,
}


def build_stage_specs(
    budgets: Mapping[str, int] | None = None,
    target_stage: str | None = None,
) -> tuple[StageSpec, ...]:
    configured = dict(DEFAULT_STAGE_BUDGETS)
    configured.update({str(key): int(value) for key, value in dict(budgets or {}).items()})
    unknown = sorted(set(configured) - set(TARGET_STAGE_ORDER))
    if unknown:
        raise ValueError(f"unknown TS02 stage budgets: {unknown}")
    target = TARGET_STAGE_ORDER[-1] if target_stage is None else str(target_stage)
    if target not in TARGET_STAGE_ORDER:
        raise KeyError(
            f"unknown TS02 target stage {target!r}; expected one of {TARGET_STAGE_ORDER}"
        )
    return (
        StageSpec(
            id=target,
            completion_predicate=_completion(target),
            failure_predicate=_failed,
            model_step_budget=configured[target],
        ),
    )


def privileged_stage_state(runtime: Any) -> TS02EvalResult:
    shared = getattr(runtime, "metadata", {}).get("shared_runtime")
    return evaluate_initialized_runtime(shared or runtime)


def stage_completed(state: TS02EvalResult, stage_id: str) -> bool:
    stage = str(stage_id)
    if stage in {BEFORE_TARGET_DRAWER, TARGET_DRAWER_RETURNED}:
        return False
    try:
        return bool(state.stages[stage].qualified)
    except KeyError as exc:
        raise KeyError(
            f"unknown TS02 stage {stage!r}; expected one of {CAPTURE_STAGE_ORDER}"
        ) from exc


def build_expert_stage_boundaries(
    runtime: Any,
    plan: Any,
    stage_end_steps: Mapping[str, int],
) -> dict[str, int]:
    del runtime
    fixture_order = tuple(str(value) for value in plan.metadata["fixture_order"])
    target_fixture = str(plan.metadata["target_fixture"])
    if not fixture_order or fixture_order[-1] != target_fixture:
        raise ValueError("TS02 expert fixture_order must end at target_fixture")

    boundaries: dict[str, int] = {}
    if len(fixture_order) == 1:
        boundaries[BEFORE_TARGET_DRAWER] = 0
    else:
        previous = fixture_order[-2]
        stage_name = f"return_arm_after_{previous}"
        if stage_name in stage_end_steps:
            boundaries[BEFORE_TARGET_DRAWER] = int(stage_end_steps[stage_name])

    returned_stage = f"return_arm_after_{target_fixture}"
    if returned_stage in stage_end_steps:
        boundaries[TARGET_DRAWER_RETURNED] = int(stage_end_steps[returned_stage])
    return boundaries


def _completion(stage_id: str):
    return lambda state: stage_completed(state, stage_id)


def _failed(state: TS02EvalResult) -> bool:
    return bool(state.failure_code)


STAGE_EVALUATION = StageEvaluationIntegration(
    task_id="TS-02",
    target_stage_ids=TARGET_STAGE_ORDER,
    capture_stage_ids=CAPTURE_STAGE_ORDER,
    prefix_stage_by_target=_PREFIX_STAGE_BY_TARGET,
    completion_stage_by_target={stage_id: stage_id for stage_id in TARGET_STAGE_ORDER},
    spec_builder=build_stage_specs,
    privileged_state_builder=privileged_stage_state,
    stage_completion_checker=stage_completed,
    expert_stage_boundary_builder=build_expert_stage_boundaries,
)


__all__ = [
    "BEFORE_TARGET_DRAWER",
    "CAPTURE_STAGE_ORDER",
    "DEFAULT_STAGE_BUDGETS",
    "STAGE_EVALUATION",
    "TARGET_DRAWER_RETURNED",
    "TARGET_STAGE_ORDER",
    "build_expert_stage_boundaries",
    "build_stage_specs",
    "privileged_stage_state",
    "stage_completed",
]
