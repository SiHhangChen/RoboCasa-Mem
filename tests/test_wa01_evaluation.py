from __future__ import annotations

from types import SimpleNamespace

from bench.evaluation.report_stages import validate_report_stages
from bench.tasks.wa01.evaluation import (
    REPORT_STAGE_ORDER,
    STAGE_ORDER,
    TASK_EVALUATOR_METADATA_KEY,
    WA01EvaluationSnapshot,
    WA01Evaluator,
    diagnostics_for_runtime,
    report_stages_for_runtime,
    success_for_runtime,
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
    upright = dict(on_counter)
    return WA01EvaluationSnapshot(
        step_id=step,
        sugar_delivery_coffee_id=sugar,
        coffee_on_dining_counter=on_counter,
        coffee_upright=upright,
        selected_coffee_id=selected,
        details={
            "sugar": {
                "sugared_coffee_id": sugar,
            },
            "coffee_pair": {"together_before_selection": len(ready) == 2},
            "selection": {
                "selected_coffee_id": selected,
                "correct": selected is not None and selected == sugar,
            },
        },
    )


def _evaluator() -> WA01Evaluator:
    evaluator = WA01Evaluator()
    evaluator.reset(_snapshot(0))
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
        "success": 4,
    }
    report = result.to_report_stages()
    assert report["order"] == [
        "sugar_added",
        "one_cup_ready",
        "two_cups_ready",
        "select",
        "success",
    ]
    assert {
        name: report["stages"][name]["first_pass_step"]
        for name in REPORT_STAGE_ORDER
    } == {
        "sugar_added": 1,
        "one_cup_ready": 2,
        "two_cups_ready": 3,
        "select": 4,
        "success": 4,
    }
    assert validate_report_stages(
        report,
        expected_task_id="WA-01",
        expected_profile_id="WA-01/report-v2",
        expected_success=True,
    ) == report


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


def test_wa01_evaluator_reports_wrong_selection_as_select_only() -> None:
    evaluator = _evaluator()
    both = ("coffee_A", "coffee_B")

    evaluator.observe(_snapshot(1, sugar="coffee_A"))
    evaluator.observe(_snapshot(2, sugar="coffee_A", ready=("coffee_A",)))
    evaluator.observe(_snapshot(3, sugar="coffee_A", ready=both))
    result = evaluator.observe(
        _snapshot(4, sugar="coffee_A", ready=both, selected="coffee_B")
    )

    assert result.stages["selection"].observed
    assert result.stages["selection"].qualified
    assert not result.stages["success"].qualified
    assert result.stages["selection"].reason is None
    assert not result.success
    assert result.failure_code == "wrong_coffee_selected"
    diagnostics = result.to_diagnostics()
    assert diagnostics["selection"]["observed_correct"] is False
    assert diagnostics["selection"]["correct"] is False
    assert diagnostics["success_checks"]["correct_sugared_selection"] is False


def test_wa01_ready_without_selection_does_not_pass_select_or_success() -> None:
    evaluator = _evaluator()
    both = ("coffee_A", "coffee_B")

    evaluator.observe(_snapshot(1, sugar="coffee_A"))
    evaluator.observe(_snapshot(2, sugar="coffee_A", ready=("coffee_A",)))
    result = evaluator.observe(_snapshot(3, sugar="coffee_A", ready=both))

    assert result.stages["ready"].qualified
    assert not result.stages["selection"].qualified
    assert not result.stages["success"].qualified
    report = result.to_report_stages()
    assert report["stages"]["select"]["passed"] is False
    assert report["stages"]["success"]["passed"] is False


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
    assert report_stages_for_runtime(runtime)["order"] == list(
        REPORT_STAGE_ORDER
    )
    assert diagnostics_for_runtime(runtime) == diagnostics


def test_wa01_runtime_registers_public_report_stages() -> None:
    from bench.tasks.wa01 import runtime

    assert runtime.TASK_RUNTIME_API.report_stage_profile_id == "WA-01/report-v2"
    assert callable(runtime.TASK_RUNTIME_API.report_stages)
