from __future__ import annotations

from types import SimpleNamespace

import bench.tasks.wr01.runtime as wr01_runtime
from bench.evaluation.report_stages import validate_report_stages
from bench.tasks.wr01.evaluation import (
    REPORT_STAGE_ORDER,
    STAGE_ORDER,
    TASK_EVALUATOR_METADATA_KEY,
    WR01EvaluationSnapshot,
    WR01Evaluator,
    diagnostics_for_runtime,
    report_stages_for_runtime,
    success_for_runtime,
)


def _snapshot(
    step: int,
    *,
    order: tuple[str, str] = ("blue_can", "red_can"),
    target: str = "blue_can",
    target_level: str = "lower",
    grasped: tuple[str, ...] = (),
    stored: dict[str, str] | None = None,
    upper_closed: bool = False,
    lower_closed: bool = False,
    opened: str | None = None,
    wrong_opened: bool = False,
    retrieved: str | None = None,
    distractor_stored: bool = True,
) -> WR01EvaluationSnapshot:
    return WR01EvaluationSnapshot(
        step_id=step,
        storage_order=order,
        target_item_id=target,
        target_level=target_level,
        item_grasped={item_id: item_id in grasped for item_id in order},
        stored_levels_by_item=stored or {},
        upper_door_closed=upper_closed,
        lower_door_closed=lower_closed,
        opened_level=opened,
        wrong_retrieval_level_opened=wrong_opened,
        retrieved_item_id=retrieved,
        distractor_still_stored=distractor_stored,
    )


def _evaluator() -> WR01Evaluator:
    evaluator = WR01Evaluator(
        storage_order=("blue_can", "red_can"),
        target_item_id="blue_can",
        target_level="lower",
    )
    evaluator.reset(_snapshot(0))
    return evaluator


def _complete_through_closed_cabinets(evaluator: WR01Evaluator) -> None:
    first_stored = {"blue_can": "lower"}
    both_stored = {"blue_can": "lower", "red_can": "upper"}
    evaluator.observe(_snapshot(1, grasped=("blue_can",)))
    evaluator.observe(_snapshot(2, stored=first_stored))
    evaluator.observe(_snapshot(3, grasped=("red_can",), stored=first_stored))
    evaluator.observe(_snapshot(4, stored=both_stored))
    evaluator.observe(_snapshot(5, stored=both_stored, upper_closed=True))
    evaluator.observe(
        _snapshot(
            6,
            stored=both_stored,
            upper_closed=True,
            lower_closed=True,
        )
    )


def test_wr01_evaluator_qualifies_the_eight_ordered_stages() -> None:
    evaluator = _evaluator()
    both_stored = {"blue_can": "lower", "red_can": "upper"}
    _complete_through_closed_cabinets(evaluator)
    evaluator.observe(
        _snapshot(
            7,
            stored=both_stored,
            upper_closed=True,
            lower_closed=True,
            opened="lower",
        )
    )
    result = evaluator.observe(
        _snapshot(
            8,
            stored=both_stored,
            upper_closed=True,
            lower_closed=True,
            opened="lower",
            retrieved="blue_can",
        )
    )

    assert result.success
    assert result.terminal
    assert result.failure_code is None
    assert {
        name: result.stages[name].qualified_step for name in STAGE_ORDER
    } == {name: index for index, name in enumerate(STAGE_ORDER, start=1)}
    diagnostics = result.to_diagnostics()
    assert diagnostics["stage_checks"] == {name: True for name in STAGE_ORDER}
    assert diagnostics["evaluation"]["stage_bindings"] == {
        "first_can": "blue_can",
        "first_store_level": "lower",
        "second_can": "red_can",
        "second_store_level": "upper",
        "retrieval_target": "blue_can",
        "retrieval_level": "lower",
    }

    report = result.to_report_stages()
    assert report == {
        "schema_version": "membench.report_stages/v1",
        "task_id": "WR-01",
        "profile_id": "WR-01/report-v1",
        "order": [
            "one_can_stored",
            "two_cans_stored",
            "storage_hidden",
            "correct_cabinet_opened",
            "success",
        ],
        "stages": {
            "one_can_stored": {
                "applicable": True,
                "passed": True,
                "first_pass_step": 2,
            },
            "two_cans_stored": {
                "applicable": True,
                "passed": True,
                "first_pass_step": 4,
            },
            "storage_hidden": {
                "applicable": True,
                "passed": True,
                "first_pass_step": 6,
            },
            "correct_cabinet_opened": {
                "applicable": True,
                "passed": True,
                "first_pass_step": 7,
            },
            "success": {
                "applicable": True,
                "passed": True,
                "first_pass_step": 8,
            },
        },
    }
    assert not set(STAGE_ORDER) & set(report["stages"])
    assert validate_report_stages(
        report,
        expected_task_id="WR-01",
        expected_profile_id="WR-01/report-v1",
        expected_success=True,
    ) == report


def test_wr01_evaluator_rejects_a_skipped_pick_until_a_new_grasp_edge() -> None:
    evaluator = _evaluator()
    first_stored = {"blue_can": "lower"}

    skipped = evaluator.observe(_snapshot(1, grasped=("red_can",)))
    evaluator.observe(_snapshot(2))
    evaluator.observe(_snapshot(3, grasped=("blue_can",)))
    evaluator.observe(_snapshot(4, stored=first_stored))
    recovered = evaluator.observe(
        _snapshot(5, grasped=("red_can",), stored=first_stored)
    )

    assert skipped.stages["second_can_pick"].observed
    assert skipped.stages["second_can_pick"].invalidated
    assert skipped.stages["second_can_pick"].reason == (
        "prerequisite_not_qualified:first_can_store"
    )
    assert recovered.stages["second_can_pick"].qualified
    assert recovered.stages["second_can_pick"].qualified_step == 5


def test_wr01_evaluator_makes_the_wrong_retrieval_cabinet_terminal() -> None:
    evaluator = _evaluator()
    both_stored = {"blue_can": "lower", "red_can": "upper"}
    _complete_through_closed_cabinets(evaluator)

    result = evaluator.observe(
        _snapshot(
            7,
            stored=both_stored,
            lower_closed=True,
            opened="upper",
            wrong_opened=True,
        )
    )

    assert result.terminal
    assert not result.success
    assert result.failure_code == "wrong_retrieval_cabinet"
    assert result.stages["open_cabinet"].observed
    assert not result.stages["open_cabinet"].qualified
    assert result.stages["open_cabinet"].reason == "wrong_retrieval_cabinet"

    report = result.to_report_stages()
    assert [
        report["stages"][name]["passed"] for name in REPORT_STAGE_ORDER
    ] == [True, True, True, False, False]
    assert report["stages"]["correct_cabinet_opened"]["first_pass_step"] is None
    assert report["stages"]["success"]["first_pass_step"] is None


def test_wr01_runtime_callbacks_share_one_evaluator_result() -> None:
    evaluator = _evaluator()

    class FakeEnv:
        snapshot = _snapshot(1, grasped=("blue_can",))

        def evaluation_snapshot(self) -> WR01EvaluationSnapshot:
            return self.snapshot

    runtime = SimpleNamespace(
        env=FakeEnv(),
        metadata={TASK_EVALUATOR_METADATA_KEY: evaluator},
    )

    diagnostics = diagnostics_for_runtime(runtime)
    assert diagnostics["stage_checks"]["first_can_pick"]
    assert not success_for_runtime(runtime, {})
    assert diagnostics_for_runtime(runtime) == diagnostics
    report = report_stages_for_runtime(runtime)
    assert report["order"] == list(REPORT_STAGE_ORDER)
    assert not any(stage["passed"] for stage in report["stages"].values())


def test_wr01_runtime_registers_public_report_stage_projector() -> None:
    assert wr01_runtime.TASK_RUNTIME_API.report_stage_profile_id == (
        "WR-01/report-v1"
    )
    assert callable(wr01_runtime.TASK_RUNTIME_API.report_stages)
