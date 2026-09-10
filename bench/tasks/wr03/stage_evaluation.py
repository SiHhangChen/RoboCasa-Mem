"""WR-03 integration for the canonical-reset remembered-cabinet probe."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from bench.evaluation.stage_integration import StageEvaluationIntegration
from bench.evaluation.stage_rollout import StageSpec
from bench.tasks.wr03.evaluation import WR03EvalResult, evaluate_initialized_runtime


TARGET_STAGE_ORDER = ("canonical_reset", "open_remembered_cabinet")
CAPTURE_STAGE_ORDER = TARGET_STAGE_ORDER

DEFAULT_STAGE_BUDGETS = {
    "canonical_reset": 1200,
    "open_remembered_cabinet": 1200,
}


def build_stage_specs(
    budgets: Mapping[str, int] | None = None,
    target_stage: str | None = None,
) -> tuple[StageSpec, ...]:
    configured = dict(DEFAULT_STAGE_BUDGETS)
    configured.update({str(key): int(value) for key, value in dict(budgets or {}).items()})
    unknown = sorted(set(configured) - set(TARGET_STAGE_ORDER))
    if unknown:
        raise ValueError(f"unknown WR-03 stage budgets: {unknown}")
    target = TARGET_STAGE_ORDER[-1] if target_stage is None else str(target_stage)
    if target not in TARGET_STAGE_ORDER:
        raise KeyError(
            f"unknown WR-03 target stage {target!r}; expected one of {TARGET_STAGE_ORDER}"
        )
    return (
        StageSpec(
            id=target,
            completion_predicate=_completion(target),
            failure_predicate=_failed,
            model_step_budget=configured[target],
        ),
    )


def privileged_stage_state(runtime: Any) -> WR03EvalResult:
    shared = getattr(runtime, "metadata", {}).get("shared_runtime")
    return evaluate_initialized_runtime(shared or runtime)


def stage_completed(state: WR03EvalResult, stage_id: str) -> bool:
    stage = str(stage_id)
    try:
        return bool(state.stages[stage].qualified)
    except KeyError as exc:
        raise KeyError(
            f"unknown WR-03 stage {stage!r}; expected one of {CAPTURE_STAGE_ORDER}"
        ) from exc


def _completion(stage_id: str):
    return lambda state: stage_completed(state, stage_id)


def _failed(state: WR03EvalResult) -> bool:
    return bool(state.failure_code)


STAGE_EVALUATION = StageEvaluationIntegration(
    task_id="WR-03",
    target_stage_ids=TARGET_STAGE_ORDER,
    capture_stage_ids=CAPTURE_STAGE_ORDER,
    prefix_stage_by_target={
        "canonical_reset": None,
        "open_remembered_cabinet": "canonical_reset",
    },
    completion_stage_by_target={stage_id: stage_id for stage_id in TARGET_STAGE_ORDER},
    spec_builder=build_stage_specs,
    privileged_state_builder=privileged_stage_state,
    stage_completion_checker=stage_completed,
)


__all__ = [
    "CAPTURE_STAGE_ORDER",
    "DEFAULT_STAGE_BUDGETS",
    "STAGE_EVALUATION",
    "TARGET_STAGE_ORDER",
    "build_stage_specs",
    "privileged_stage_state",
    "stage_completed",
]
