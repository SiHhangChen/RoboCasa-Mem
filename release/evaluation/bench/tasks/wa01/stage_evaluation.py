"""WA-01 integration for stage-conditioned policy evaluation."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from bench.evaluation.stage_integration import StageEvaluationIntegration
from bench.evaluation.stage_rollout import StageSpec
from bench.tasks.wa01.evaluation import (
    STAGE_ORDER as EVALUATOR_STAGE_ORDER,
    WA01EvalResult,
    evaluate_initialized_runtime,
)


TARGET_STAGE_ORDER = (
    "sugar_added",
    "one_cup_ready",
    "two_cups_ready",
    "select",
    "success",
)
CAPTURE_STAGE_ORDER = EVALUATOR_STAGE_ORDER

_EVALUATOR_STAGE_BY_TARGET = {
    "sugar_added": "sugar",
    "one_cup_ready": "one_ready",
    "two_cups_ready": "ready",
    "select": "selection",
    "success": "success",
}
_TARGET_PATHS = {
    "sugar_added": ("sugar_added",),
    "one_cup_ready": ("sugar_added", "one_cup_ready"),
    "two_cups_ready": ("sugar_added", "one_cup_ready", "two_cups_ready"),
    "select": (
        "sugar_added",
        "one_cup_ready",
        "two_cups_ready",
        "select",
    ),
    "success": (
        "sugar_added",
        "one_cup_ready",
        "two_cups_ready",
        "select",
        "success",
    ),
}
_PREFIX_STAGE_BY_TARGET = {
    "sugar_added": None,
    "one_cup_ready": "sugar",
    "two_cups_ready": "one_ready",
    "select": "ready",
    # Success consumes the select result.  It intentionally starts from the
    # ready prefix so a policy that never selects cannot pass this stage.
    "success": "ready",
}

DEFAULT_STAGE_BUDGETS = {
    "sugar_added": 1500,
    "one_cup_ready": 2000,
    "two_cups_ready": 2000,
    "select": 1000,
    "success": 1000,
}


def build_stage_specs(
    budgets: Mapping[str, int] | None = None,
    target_stage: str | None = None,
) -> tuple[StageSpec, ...]:
    configured = dict(DEFAULT_STAGE_BUDGETS)
    configured.update({str(key): int(value) for key, value in dict(budgets or {}).items()})
    unknown = sorted(set(configured) - set(TARGET_STAGE_ORDER))
    if unknown:
        raise ValueError(f"unknown WA-01 stage budgets: {unknown}")
    target = TARGET_STAGE_ORDER[-1] if target_stage is None else str(target_stage)
    try:
        path = _TARGET_PATHS[target]
    except KeyError as exc:
        raise KeyError(
            f"unknown WA-01 target stage {target!r}; expected one of {TARGET_STAGE_ORDER}"
        ) from exc
    failure_predicate = _failure_predicate(target)
    return tuple(
        StageSpec(
            id=stage_id,
            completion_predicate=_completion(stage_id),
            failure_predicate=failure_predicate,
            model_step_budget=configured[stage_id],
            expert_prefix_required=not (
                target == "success" and stage_id == "select"
            ),
        )
        for stage_id in path
    )


def privileged_stage_state(runtime: Any) -> WA01EvalResult:
    shared = getattr(runtime, "metadata", {}).get("shared_runtime")
    return evaluate_initialized_runtime(shared or runtime)


def stage_completed(state: WA01EvalResult, stage_id: str) -> bool:
    stage = str(stage_id)
    evaluator_stage = _EVALUATOR_STAGE_BY_TARGET.get(stage, stage)
    try:
        return bool(state.stages[evaluator_stage].qualified)
    except KeyError as exc:
        expected = tuple(dict.fromkeys((*TARGET_STAGE_ORDER, *CAPTURE_STAGE_ORDER)))
        raise KeyError(f"unknown WA-01 stage {stage!r}; expected one of {expected}") from exc


def _completion(stage_id: str):
    return lambda state: stage_completed(state, stage_id)


def _failure_predicate(target_stage: str):
    """Build the target-aware failure predicate.

    ``select`` is intentionally a standalone stage: a physically valid
    selection qualifies it even when the selected cup is wrong.  The
    correctness failure is consumed by the following ``success`` stage.
    Earlier stages (and the success target) still fail on every evaluator
    failure code.
    """

    ignore_wrong_selection = str(target_stage) == "select"

    def failed(state: WA01EvalResult) -> bool:
        if not state.failure_code:
            return False
        return not (
            ignore_wrong_selection
            and state.failure_code == "wrong_coffee_selected"
        )

    return failed


STAGE_EVALUATION = StageEvaluationIntegration(
    task_id="WA-01",
    target_stage_ids=TARGET_STAGE_ORDER,
    capture_stage_ids=CAPTURE_STAGE_ORDER,
    prefix_stage_by_target=_PREFIX_STAGE_BY_TARGET,
    completion_stage_by_target=_EVALUATOR_STAGE_BY_TARGET,
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
