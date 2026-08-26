from __future__ import annotations

from types import SimpleNamespace

from bench.evaluation.report_stages import validate_report_stages
from bench.tasks.wa03.evaluation import (
    REPORT_STAGE_ORDER,
    STAGE_ORDER,
    TASK_EVALUATOR_METADATA_KEY,
    WA03EvaluationSnapshot,
    WA03Evaluator,
    diagnostics_for_runtime,
    report_stages_for_runtime,
    success_for_runtime,
)

COFFEES = ("coffee_A", "coffee_B", "coffee_C")


def _slot_map(ready: tuple[str, ...]) -> dict[str, str | None]:
    """Map each cup to a distinct dining slot when ready, else None."""

    slots = ("left", "center", "right")
    return {coffee_id: (slots[index] if coffee_id in ready else None) for index, coffee_id in enumerate(COFFEES)}


def _snapshot(
    step: int,
    *,
    sugar: str | None = None,
    ready: tuple[str, ...] = (),
    selected: str | None = None,
    mode: str = "unmarked",
    target: str | None = None,
) -> WA03EvaluationSnapshot:
    on_counter = {coffee_id: coffee_id in ready for coffee_id in COFFEES}
    upright = dict(on_counter)
    return WA03EvaluationSnapshot(
        step_id=step,
        sugar_target_mode=mode,
        target_sugared_coffee_id=target,
        sugar_delivery_coffee_id=sugar,
        coffee_on_dining_counter=on_counter,
        coffee_upright=upright,
        coffee_in_dining_region=_slot_map(ready),
        selected_coffee_id=selected,
        details={
            "sugar": {
                "sugared_coffee_id": sugar,
                "assignment_correct": None,
            },
            "coffee_set": {"together_before_selection": len(ready) == 3},
            "selection": {
                "selected_coffee_id": selected,
                "correct": selected is not None and selected == sugar,
            },
        },
    )


def _evaluator(*, mode: str = "unmarked", target: str | None = None) -> WA03Evaluator:
    evaluator = WA03Evaluator(
        sugar_target_mode=mode,
        target_sugared_coffee_id=target,
    )
    evaluator.reset(_snapshot(0, mode=mode, target=target))
    return evaluator


def test_wa03_evaluator_qualifies_the_ordered_stage_funnel() -> None:
    evaluator = _evaluator()

    evaluator.observe(_snapshot(1, sugar="coffee_A"))
    evaluator.observe(_snapshot(2, sugar="coffee_A", ready=("coffee_A",)))
    evaluator.observe(_snapshot(3, sugar="coffee_A", ready=("coffee_A", "coffee_B")))
    evaluator.observe(_snapshot(4, sugar="coffee_A", ready=COFFEES))
    result = evaluator.observe(
        _snapshot(5, sugar="coffee_A", ready=COFFEES, selected="coffee_A")
    )

    assert result.success
    assert result.terminal
    assert result.failure_code is None
    assert {name: result.stages[name].qualified_step for name in STAGE_ORDER} == {
        "sugar": 1,
        "one_ready": 2,
        "two_ready": 3,
        "ready": 4,
        "selection": 5,
        "correct": 5,
        "success": 5,
    }
    report = result.to_report_stages()
    assert report["order"] == [
        "sugar_added",
        "one_cup_ready",
        "two_cups_ready",
        "three_cups_ready",
        "success",
    ]
    assert {
        name: report["stages"][name]["first_pass_step"]
        for name in REPORT_STAGE_ORDER
    } == {
        "sugar_added": 1,
        "one_cup_ready": 2,
        "two_cups_ready": 3,
        "three_cups_ready": 4,
        "success": 5,
    }
    assert validate_report_stages(
        report,
        expected_task_id="WA-03",
        expected_profile_id="WA-03/report-v1",
        expected_success=True,
    ) == report


def test_wa03_ready_requires_three_cups_each_in_its_own_region() -> None:
    evaluator = _evaluator()
    evaluator.observe(_snapshot(1, sugar="coffee_A"))
    evaluator.observe(_snapshot(2, sugar="coffee_A", ready=("coffee_A",)))
    evaluator.observe(_snapshot(3, sugar="coffee_A", ready=("coffee_A", "coffee_B")))

    # The robot carries one cup at a time, so one_ready and two_ready qualified
    # on earlier steps.  Now all three cups are on the counter and upright, but
    # coffee_B shares coffee_A's region while center is empty: not ready.
    snapshot = WA03EvaluationSnapshot(
        step_id=4,
        sugar_target_mode="unmarked",
        target_sugared_coffee_id=None,
        sugar_delivery_coffee_id="coffee_A",
        coffee_on_dining_counter={coffee_id: True for coffee_id in COFFEES},
        coffee_upright={coffee_id: True for coffee_id in COFFEES},
        coffee_in_dining_region={"coffee_A": "left", "coffee_B": "left", "coffee_C": "right"},
        selected_coffee_id=None,
    )
    assert not snapshot.ready
    result = evaluator.observe(snapshot)
    assert result.stages["one_ready"].qualified
    assert result.stages["two_ready"].qualified
    assert not result.stages["ready"].qualified


def test_wa03_two_ready_qualifies_before_all_three_are_in_regions() -> None:
    evaluator = _evaluator()

    evaluator.observe(_snapshot(1, sugar="coffee_A"))
    evaluator.observe(_snapshot(2, sugar="coffee_A", ready=("coffee_A",)))
    result = evaluator.observe(_snapshot(3, sugar="coffee_A", ready=("coffee_A", "coffee_B")))

    assert result.stages["one_ready"].qualified
    assert result.stages["two_ready"].qualified
    assert not result.stages["ready"].qualified


def test_wa03_evaluator_rejects_skipped_stages_and_duplicate_queries() -> None:
    evaluator = _evaluator()

    evaluator.observe(_snapshot(1, ready=COFFEES))
    sugar_result = evaluator.observe(_snapshot(2, sugar="coffee_A", ready=COFFEES))
    duplicate = evaluator.observe(_snapshot(2, sugar="coffee_A", ready=COFFEES))
    jumped = evaluator.observe(
        _snapshot(3, sugar="coffee_A", ready=COFFEES, selected="coffee_A")
    )

    assert duplicate is sugar_result
    assert sugar_result.stages["sugar"].qualified
    assert not sugar_result.stages["one_ready"].qualified
    assert not sugar_result.stages["two_ready"].qualified
    assert not sugar_result.stages["ready"].qualified
    assert jumped.stages["selection"].observed
    assert jumped.stages["selection"].invalidated
    assert jumped.stages["selection"].reason == "prerequisite_not_qualified:ready"
    assert jumped.stages["correct"].observed
    assert jumped.stages["success"].observed
    assert not any(stage.qualified for name, stage in jumped.stages.items() if name != "sugar")
    assert not jumped.success
    assert jumped.failure_code == "selection_before_ready"


def test_wa03_evaluator_can_recover_only_after_a_new_ordered_observation() -> None:
    evaluator = _evaluator()
    one = ("coffee_A",)
    two = ("coffee_A", "coffee_B")

    evaluator.observe(_snapshot(1, ready=COFFEES))
    evaluator.observe(_snapshot(2, sugar="coffee_A", ready=COFFEES))
    evaluator.observe(_snapshot(3, sugar="coffee_A"))
    evaluator.observe(_snapshot(4, sugar="coffee_A", ready=one))
    evaluator.observe(_snapshot(5, sugar="coffee_A", ready=two))
    evaluator.observe(_snapshot(6, sugar="coffee_A", ready=COFFEES))
    result = evaluator.observe(
        _snapshot(7, sugar="coffee_A", ready=COFFEES, selected="coffee_A")
    )

    assert result.success
    assert result.stages["one_ready"].qualified_step == 4
    assert result.stages["two_ready"].qualified_step == 5
    assert result.stages["ready"].qualified_step == 6
    assert not result.stages["one_ready"].invalidated
    assert not result.stages["two_ready"].invalidated
    assert not result.stages["ready"].invalidated
    rejected = result.details["evaluation"]["rejected_stage_observations"]
    assert rejected["one_ready"] == [
        {"step": 1, "reason": "prerequisite_not_qualified:sugar"}
    ]


def test_wa03_evaluator_reports_wrong_selection_without_qualifying_correct() -> None:
    evaluator = _evaluator()

    evaluator.observe(_snapshot(1, sugar="coffee_A"))
    evaluator.observe(_snapshot(2, sugar="coffee_A", ready=("coffee_A",)))
    evaluator.observe(_snapshot(3, sugar="coffee_A", ready=("coffee_A", "coffee_B")))
    evaluator.observe(_snapshot(4, sugar="coffee_A", ready=COFFEES))
    result = evaluator.observe(
        _snapshot(5, sugar="coffee_A", ready=COFFEES, selected="coffee_B")
    )
    assert result.stages["selection"].qualified
    assert not result.stages["correct"].qualified
    assert not result.success
    assert result.failure_code == "wrong_coffee_selected"


def test_wa03_marked_mode_keeps_assignment_separate_from_selection_correctness() -> None:
    evaluator = _evaluator(mode="marked", target="coffee_A")

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
        _snapshot(3, sugar="coffee_B", ready=("coffee_A", "coffee_B"), mode="marked", target="coffee_A")
    )
    evaluator.observe(
        _snapshot(4, sugar="coffee_B", ready=COFFEES, mode="marked", target="coffee_A")
    )
    result = evaluator.observe(
        _snapshot(
            5,
            sugar="coffee_B",
            ready=COFFEES,
            selected="coffee_B",
            mode="marked",
            target="coffee_A",
        )
    )

    assert result.stages["correct"].qualified
    assert not result.stages["success"].qualified
    assert result.failure_code == "wrong_sugar_assignment"
    assert result.to_diagnostics()["success_checks"]["sugar_assignment_correct"] is False
    report = result.to_report_stages()
    assert {
        name: stage["passed"] for name, stage in report["stages"].items()
    } == {
        "sugar_added": True,
        "one_cup_ready": True,
        "two_cups_ready": True,
        "three_cups_ready": True,
        "success": False,
    }


def test_wa03_runtime_callbacks_share_one_evaluator_result() -> None:
    evaluator = _evaluator()

    class FakeEnv:
        def __init__(self) -> None:
            self.snapshot = _snapshot(1, sugar="coffee_A")

        def evaluation_snapshot(self) -> WA03EvaluationSnapshot:
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


def test_wa03_runtime_registers_public_report_stages() -> None:
    from bench.tasks.wa03 import runtime

    assert runtime.TASK_RUNTIME_API.report_stage_profile_id == "WA-03/report-v1"
    assert callable(runtime.TASK_RUNTIME_API.report_stages)
