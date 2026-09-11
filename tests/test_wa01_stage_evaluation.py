from __future__ import annotations

from bench.tasks.wa01.evaluation import WA01EvaluationSnapshot, WA01Evaluator
from bench.tasks.wa01.stage_evaluation import (
    CAPTURE_STAGE_ORDER,
    STAGE_EVALUATION,
    TARGET_STAGE_ORDER,
    build_stage_specs,
    stage_completed,
)


def _snapshot(
    step: int,
    *,
    sugar: str | None = None,
    ready: tuple[str, ...] = (),
    selected: str | None = None,
) -> WA01EvaluationSnapshot:
    on_counter = {
        coffee_id: coffee_id in ready for coffee_id in ("coffee_A", "coffee_B")
    }
    return WA01EvaluationSnapshot(
        step_id=step,
        sugar_delivery_coffee_id=sugar,
        coffee_on_dining_counter=on_counter,
        coffee_upright=on_counter,
        selected_coffee_id=selected,
    )


def _ready_evaluator() -> WA01Evaluator:
    evaluator = WA01Evaluator()
    evaluator.reset(_snapshot(0))
    evaluator.observe(_snapshot(1, sugar="coffee_A"))
    evaluator.observe(_snapshot(2, sugar="coffee_A", ready=("coffee_A",)))
    evaluator.observe(
        _snapshot(3, sugar="coffee_A", ready=("coffee_A", "coffee_B"))
    )
    return evaluator


def test_wa01_stage_specs_reuse_authoritative_evaluator() -> None:
    evaluator = _ready_evaluator()
    ready = evaluator.result()
    success = evaluator.observe(
        _snapshot(
            4,
            sugar="coffee_A",
            ready=("coffee_A", "coffee_B"),
            selected="coffee_A",
        )
    )

    assert TARGET_STAGE_ORDER == (
        "sugar_added",
        "one_cup_ready",
        "two_cups_ready",
        "select",
        "success",
    )
    assert CAPTURE_STAGE_ORDER == (
        "sugar",
        "one_ready",
        "ready",
        "selection",
        "success",
    )
    assert stage_completed(ready, "two_cups_ready")
    assert stage_completed(success, "success")
    assert STAGE_EVALUATION.completion_stage_by_target["select"] == "selection"
    assert STAGE_EVALUATION.completion_stage_by_target["success"] == "success"


def test_wa01_success_uses_selection_directly_from_ready() -> None:
    success = build_stage_specs(target_stage="success")

    assert tuple(spec.id for spec in success) == (
        "sugar_added",
        "one_cup_ready",
        "two_cups_ready",
        "select",
        "success",
    )
    assert success[-2].expert_prefix_required is False
    assert success[-1].expert_prefix_required is True
    assert STAGE_EVALUATION.prefix_stage_by_target["success"] == "ready"


def test_wa01_wrong_selection_fails_success() -> None:
    evaluator = _ready_evaluator()
    wrong = evaluator.observe(
        _snapshot(
            4,
            sugar="coffee_A",
            ready=("coffee_A", "coffee_B"),
            selected="coffee_B",
        )
    )

    success = build_stage_specs(target_stage="success")[-1]
    assert not success.completion_predicate(wrong)
    assert success.failure_predicate(wrong)

    select = build_stage_specs(target_stage="select")[-1]
    assert select.completion_predicate(wrong)
    assert not select.failure_predicate(wrong)
