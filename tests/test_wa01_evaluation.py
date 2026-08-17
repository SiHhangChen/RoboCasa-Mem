from __future__ import annotations

from types import SimpleNamespace

from bench.tasks.wa01.evaluation import (
    STAGE_ORDER,
    TASK_EVALUATOR_METADATA_KEY,
    WA01EvaluationSnapshot,
    WA01Evaluator,
    diagnostics_for_runtime,
    success_for_runtime,
)


def _snapshot(
    step: int,
    *,
    sugar: str | None = None,
    ready: tuple[str, ...] = (),
    selected: str | None = None,
    mode: str = "unmarked",
    target: str | None = None,
) -> WA01EvaluationSnapshot:
    on_counter = {
        coffee_id: coffee_id in ready for coffee_id in ("coffee_A", "coffee_B")
    }
    upright = dict(on_counter)
    return WA01EvaluationSnapshot(
        step_id=step,
        sugar_target_mode=mode,
        target_sugared_coffee_id=target,
        sugar_delivery_coffee_id=sugar,
        coffee_on_dining_counter=on_counter,
        coffee_upright=upright,
        selected_coffee_id=selected,
        details={
            "sugar": {
                "sugared_coffee_id": sugar,
                "assignment_correct": None,
            },
            "coffee_pair": {"together_before_selection": len(ready) == 2},
            "selection": {
                "selected_coffee_id": selected,
                "correct": selected is not None and selected == sugar,
            },
        },
    )


def _evaluator(*, mode: str = "unmarked", target: str | None = None) -> WA01Evaluator:
    evaluator = WA01Evaluator(
        sugar_target_mode=mode,
        target_sugared_coffee_id=target,
    )
    evaluator.reset(_snapshot(0, mode=mode, target=target))
    return evaluator


def test_wa01_evaluator_qualifies_the_ordered_stage_funnel() -> None:
    evaluator = _evaluator()

    evaluator.observe(_snapshot(1, sugar="coffee_A"))
    evaluator.observe(_snapshot(2, sugar="coffee_A", ready=("coffee_A",)))
    evaluator.observe(
        _snapshot(3, sugar="coffee_A", ready=("coffee_A", "coffee_B"))
    )
    result = evaluator.observe(
        _snapshot(
            4,
            sugar="coffee_A",
            ready=("coffee_A", "coffee_B"),
            selected="coffee_A",
        )
    )

    assert result.success
    assert result.terminal
    assert result.failure_code is None
    assert {name: result.stages[name].qualified_step for name in STAGE_ORDER} == {
        "sugar": 1,
        "one_ready": 2,
        "ready": 3,
        "selection": 4,
        "correct": 4,
        "success": 4,
    }


def test_wa01_evaluator_rejects_skipped_stages_and_duplicate_queries() -> None:
    evaluator = _evaluator()
    both = ("coffee_A", "coffee_B")

    evaluator.observe(_snapshot(1, ready=both))
    sugar_result = evaluator.observe(_snapshot(2, sugar="coffee_A", ready=both))
    duplicate = evaluator.observe(_snapshot(2, sugar="coffee_A", ready=both))
    jumped = evaluator.observe(
        _snapshot(3, sugar="coffee_A", ready=both, selected="coffee_A")
    )

    assert duplicate is sugar_result
    assert sugar_result.stages["sugar"].qualified
    assert not sugar_result.stages["one_ready"].qualified
    assert not sugar_result.stages["ready"].qualified
    assert jumped.stages["selection"].observed
    assert jumped.stages["selection"].invalidated
    assert jumped.stages["selection"].reason == "prerequisite_not_qualified:ready"
    assert jumped.stages["correct"].observed
    assert jumped.stages["success"].observed
    assert not any(stage.qualified for name, stage in jumped.stages.items() if name != "sugar")
    assert not jumped.success
    assert jumped.failure_code == "selection_before_ready"


def test_wa01_evaluator_can_recover_only_after_a_new_ordered_observation() -> None:
    evaluator = _evaluator()
    both = ("coffee_A", "coffee_B")

    evaluator.observe(_snapshot(1, ready=both))
    evaluator.observe(_snapshot(2, sugar="coffee_A", ready=both))
    evaluator.observe(_snapshot(3, sugar="coffee_A"))
    evaluator.observe(_snapshot(4, sugar="coffee_A", ready=("coffee_A",)))
    evaluator.observe(_snapshot(5, sugar="coffee_A", ready=both))
    result = evaluator.observe(
        _snapshot(6, sugar="coffee_A", ready=both, selected="coffee_A")
    )

    assert result.success
    assert result.stages["one_ready"].qualified_step == 4
    assert result.stages["ready"].qualified_step == 5
    assert not result.stages["one_ready"].invalidated
    assert not result.stages["ready"].invalidated
    rejected = result.details["evaluation"]["rejected_stage_observations"]
    assert rejected["one_ready"] == [
        {"step": 1, "reason": "prerequisite_not_qualified:sugar"}
    ]


def test_wa01_evaluator_reports_wrong_selection_without_qualifying_correct() -> None:
    evaluator = _evaluator()
    both = ("coffee_A", "coffee_B")

    evaluator.observe(_snapshot(1, sugar="coffee_A"))
    evaluator.observe(_snapshot(2, sugar="coffee_A", ready=("coffee_A",)))
    evaluator.observe(_snapshot(3, sugar="coffee_A", ready=both))
    result = evaluator.observe(
        _snapshot(4, sugar="coffee_A", ready=both, selected="coffee_B")
    )

    assert result.stages["selection"].qualified
    assert not result.stages["correct"].qualified
    assert not result.success
    assert result.failure_code == "wrong_coffee_selected"


def test_wa01_marked_mode_keeps_assignment_separate_from_selection_correctness() -> None:
    evaluator = _evaluator(mode="marked", target="coffee_A")
    both = ("coffee_A", "coffee_B")

    evaluator.observe(_snapshot(1, sugar="coffee_B", mode="marked", target="coffee_A"))
    evaluator.observe(
        _snapshot(
            2,
            sugar="coffee_B",
            ready=("coffee_A",),
            mode="marked",
            target="coffee_A",
        )
    )
    evaluator.observe(
        _snapshot(3, sugar="coffee_B", ready=both, mode="marked", target="coffee_A")
    )
    result = evaluator.observe(
        _snapshot(
            4,
            sugar="coffee_B",
            ready=both,
            selected="coffee_B",
            mode="marked",
            target="coffee_A",
        )
    )

    assert result.stages["correct"].qualified
    assert not result.stages["success"].qualified
    assert result.failure_code == "wrong_sugar_assignment"
    assert result.to_diagnostics()["success_checks"]["sugar_assignment_correct"] is False


def test_wa01_runtime_callbacks_share_one_evaluator_result() -> None:
    evaluator = _evaluator()

    class FakeEnv:
        def __init__(self) -> None:
            self.snapshot = _snapshot(1, sugar="coffee_A")

        def evaluation_snapshot(self) -> WA01EvaluationSnapshot:
            return self.snapshot

    env = FakeEnv()
    runtime = SimpleNamespace(
        env=env,
        metadata={TASK_EVALUATOR_METADATA_KEY: evaluator},
    )

    diagnostics = diagnostics_for_runtime(runtime)
    assert diagnostics["stage_checks"]["sugar"]
    assert not success_for_runtime(runtime, {})
    assert diagnostics_for_runtime(runtime) == diagnostics
