"""WA-02 integration for stage-conditioned policy evaluation."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from bench.evaluation.stage_integration import StageEvaluationIntegration
from bench.evaluation.stage_rollout import StageSpec
from bench.tasks.wa02.evaluation import (
    STAGE_ORDER,
    WA02EvalResult,
    evaluate_initialized_runtime,
)


TARGET_STAGE_ORDER = STAGE_ORDER
CAPTURE_STAGE_ORDER = STAGE_ORDER

_PREFIX_STAGE_BY_TARGET = {
    stage_id: (None if index == 0 else TARGET_STAGE_ORDER[index - 1])
    for index, stage_id in enumerate(TARGET_STAGE_ORDER)
}

DEFAULT_STAGE_BUDGETS = {
    "first_cup_pick": 1200,
    "first_cup_fill": 1600,
    "first_cup_ready": 1200,
    "second_cup_pick": 1600,
    "second_cup_fill": 1600,
    "second_cup_ready": 1200,
    "selection_success": 1000,
}


def build_stage_specs(
    budgets: Mapping[str, int] | None = None,
    target_stage: str | None = None,
) -> tuple[StageSpec, ...]:
    configured = dict(DEFAULT_STAGE_BUDGETS)
    configured.update({str(key): int(value) for key, value in dict(budgets or {}).items()})
    unknown = sorted(set(configured) - set(TARGET_STAGE_ORDER))
    if unknown:
        raise ValueError(f"unknown WA-02 stage budgets: {unknown}")
    target = TARGET_STAGE_ORDER[-1] if target_stage is None else str(target_stage)
    if target not in TARGET_STAGE_ORDER:
        raise KeyError(
            f"unknown WA-02 target stage {target!r}; expected one of {TARGET_STAGE_ORDER}"
        )
    return (
        StageSpec(
            id=target,
            completion_predicate=_completion(target),
            failure_predicate=_failed,
            model_step_budget=configured[target],
        ),
    )


def privileged_stage_state(runtime: Any) -> WA02EvalResult:
    shared = getattr(runtime, "metadata", {}).get("shared_runtime")
    return evaluate_initialized_runtime(shared or runtime)


def stage_completed(state: WA02EvalResult, stage_id: str) -> bool:
    stage = str(stage_id)
    try:
        return bool(state.stages[stage].qualified)
    except KeyError as exc:
        raise KeyError(
            f"unknown WA-02 stage {stage!r}; expected one of {CAPTURE_STAGE_ORDER}"
        ) from exc


def build_expert_stage_boundaries(
    runtime: Any,
    plan: Any,
    stage_end_steps: Mapping[str, int],
) -> dict[str, int]:
    del runtime
    order = tuple(str(value) for value in plan.metadata["fill_temperature_order"])
    if len(order) != 2:
        raise ValueError("WA-02 expert plan must contain two fill temperatures")
    requested_cup = str(plan.metadata["requested_cup_id"])

    operation_by_stage = {
        "first_cup_pick": _unique_stage(stage_end_steps, "return_to_regular_after_pick_first_"),
        "first_cup_fill": _unique_stage(
            stage_end_steps, f"lift_after_sink_exit_first_{order[0]}_"
        ),
        "first_cup_ready": _unique_stage(stage_end_steps, "return_after_service_first_"),
        "second_cup_pick": _unique_stage(stage_end_steps, "return_to_regular_after_pick_second_"),
        "second_cup_fill": _unique_stage(
            stage_end_steps, f"lift_after_sink_exit_second_{order[1]}_"
        ),
        "second_cup_ready": _unique_stage(stage_end_steps, "return_after_service_second_"),
        "selection_success": f"record_final_take_{requested_cup}",
    }
    return {
        stage_id: int(stage_end_steps[operation])
        for stage_id, operation in operation_by_stage.items()
        if operation in stage_end_steps
    }


def _unique_stage(stage_end_steps: Mapping[str, int], prefix: str) -> str:
    matches = [str(name) for name in stage_end_steps if str(name).startswith(prefix)]
    if len(matches) != 1:
        raise ValueError(
            f"WA-02 expected one expert stage starting with {prefix!r}, got {matches}"
        )
    return matches[0]


def _completion(stage_id: str):
    return lambda state: stage_completed(state, stage_id)


def _failed(state: WA02EvalResult) -> bool:
    return bool(state.failure_code)


STAGE_EVALUATION = StageEvaluationIntegration(
    task_id="WA-02",
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
    "CAPTURE_STAGE_ORDER",
    "DEFAULT_STAGE_BUDGETS",
    "STAGE_EVALUATION",
    "TARGET_STAGE_ORDER",
    "build_expert_stage_boundaries",
    "build_stage_specs",
    "privileged_stage_state",
    "stage_completed",
]
