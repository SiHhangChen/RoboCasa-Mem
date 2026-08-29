from __future__ import annotations

from bench.tasks.wa05.evaluation import WA05EvaluationSnapshot, WA05Evaluator
from bench.tasks.wa05.stage_evaluation import build_stage_specs, stage_completed


def _snapshot(
    step: int,
    *,
    filled: str | None = None,
    staged: bool = False,
    pressed: str | None = None,
    success: bool = False,
    failure: str | None = None,
) -> WA05EvaluationSnapshot:
    return WA05EvaluationSnapshot(
        step_id=step,
        cup_id="cup_A",
        service_slot="slot_1",
        filled_temperature=filled,
        staged_before_answer=staged,
        pressed_temperature=pressed,
        domain_success=success,
        absorbing_failure=failure is not None,
        failure_code=failure,
    )


def test_wa05_stage_specs_reuse_authoritative_evaluator() -> None:
    evaluator = WA05Evaluator(cup_id="cup_A", service_slot="slot_1")
    evaluator.reset(_snapshot(0))
    filled = evaluator.observe(_snapshot(1, filled="hot"))
    staged = evaluator.observe(_snapshot(2, filled="hot", staged=True))
    success = evaluator.observe(
        _snapshot(3, filled="hot", staged=True, pressed="hot", success=True)
    )
    specs = build_stage_specs()
    assert tuple(spec.id for spec in specs) == ("cup_filled", "cup_staged", "success")
    assert stage_completed(filled, "cup_filled")
    assert stage_completed(staged, "cup_staged")
    assert stage_completed(success, "success")


def test_wa05_absorbing_failure_fails_all_stage_specs() -> None:
    evaluator = WA05Evaluator(cup_id="cup_A", service_slot="slot_1")
    evaluator.reset(_snapshot(0))
    failed = evaluator.observe(_snapshot(1, failure="button_pressed_before_fill"))
    assert all(spec.failure_predicate(failed) for spec in build_stage_specs())
