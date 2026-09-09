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
    close_order: tuple[str, str] = ("upper", "lower"),
    storage_hidden: bool = False,
    canonical_reset: bool = False,
    final_placement_verified: bool = False,
) -> WR01EvaluationSnapshot:
    return WR01EvaluationSnapshot(
        step_id=step,
        storage_order=order,
        target_item_id=target,
        target_level=target_level,
        close_order=close_order,
        item_grasped={item_id: item_id in grasped for item_id in order},
        stored_levels_by_item=stored or {},
        storage_hidden=storage_hidden,
        upper_door_closed=upper_closed,
        lower_door_closed=lower_closed,
        opened_level=opened,
        wrong_retrieval_level_opened=wrong_opened,
        retrieved_item_id=retrieved,
        distractor_still_stored=distractor_stored,
        canonical_reset=canonical_reset,
        final_placement_verified=final_placement_verified,
    )


def _evaluator() -> WR01Evaluator:
    evaluator = WR01Evaluator(
        storage_order=("blue_can", "red_can"),
        target_item_id="blue_can",
        target_level="lower",
    )
    evaluator.reset(_snapshot(0))
    return evaluator


def _complete_through_handoff(evaluator: WR01Evaluator) -> dict[str, str]:
    both_stored = {"blue_can": "lower", "red_can": "upper"}
    evaluator.observe(_snapshot(1, stored=both_stored, upper_closed=True))
    evaluator.observe(
        _snapshot(2, stored=both_stored, upper_closed=True, lower_closed=True)
    )
    evaluator.observe(
        _snapshot(
            3,
            stored=both_stored,
            storage_hidden=True,
            upper_closed=True,
            lower_closed=True,
            canonical_reset=True,
        )
    )
    return both_stored


def test_wr01_evaluator_qualifies_the_new_ordered_stages() -> None:
    evaluator = _evaluator()
    both_stored = _complete_through_handoff(evaluator)
    evaluator.observe(
        _snapshot(
            4,
            stored=both_stored,
            upper_closed=True,
            lower_closed=True,
            storage_hidden=True,
            canonical_reset=True,
            opened="lower",
        )
    )
    result = evaluator.observe(
        _snapshot(
            5,
            stored=both_stored,
            upper_closed=True,
            lower_closed=True,
            storage_hidden=True,
            canonical_reset=True,
            opened="lower",
            retrieved="blue_can",
        )
    )
    result = evaluator.observe(
        _snapshot(
            6,
            stored=both_stored,
            upper_closed=True,
            lower_closed=True,
            storage_hidden=True,
            canonical_reset=True,
            opened="lower",
            retrieved="blue_can",
            final_placement_verified=True,
        )
    )

    assert result.success
    assert result.terminal
    assert result.failure_code is None
    assert {
        name: result.stages[name].qualified_step for name in STAGE_ORDER
    } == {name: index for name, index in zip(STAGE_ORDER, range(1, 7))}
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
                "first_pass_step": 1,
            },
            "two_cans_stored": {
                "applicable": True,
                "passed": True,
                "first_pass_step": 2,
            },
            "storage_hidden": {
                "applicable": True,
                "passed": True,
                "first_pass_step": 3,
            },
            "correct_cabinet_opened": {
                "applicable": True,
                "passed": True,
                "first_pass_step": 4,
            },
            "success": {
                "applicable": True,
                "passed": True,
                "first_pass_step": 6,
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


def test_wr01_evaluator_rejects_a_skipped_close_until_a_new_close_edge() -> None:
    evaluator = _evaluator()
    both_stored = {"blue_can": "lower", "red_can": "upper"}

    skipped = evaluator.observe(
        _snapshot(1, stored=both_stored, lower_closed=True)
    )
    assert skipped.stages["close_second_door"].observed
    assert skipped.stages["close_second_door"].invalidated
    assert skipped.stages["close_second_door"].reason == (
        "prerequisite_not_qualified:close_first_door"
    )
    evaluator.observe(_snapshot(2, stored=both_stored, upper_closed=True, lower_closed=True))
    evaluator.observe(_snapshot(3, stored=both_stored, upper_closed=True))
    recovered = evaluator.observe(
        _snapshot(4, stored=both_stored, upper_closed=True, lower_closed=True)
    )
    assert recovered.stages["close_second_door"].qualified
    assert recovered.stages["close_second_door"].qualified_step == 4


def test_wr01_evaluator_makes_the_wrong_retrieval_cabinet_terminal() -> None:
    evaluator = _evaluator()
    both_stored = _complete_through_handoff(evaluator)

    result = evaluator.observe(
        _snapshot(
            4,
            stored=both_stored,
            storage_hidden=True,
            canonical_reset=True,
            lower_closed=True,
            upper_closed=True,
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
    both_stored = _complete_through_handoff(evaluator)

    class FakeEnv:
        snapshot = _snapshot(
            3,
            stored=both_stored,
            storage_hidden=True,
            upper_closed=True,
            lower_closed=True,
            canonical_reset=True,
        )

        def evaluation_snapshot(self) -> WR01EvaluationSnapshot:
            return self.snapshot

    runtime = SimpleNamespace(
        env=FakeEnv(),
        metadata={TASK_EVALUATOR_METADATA_KEY: evaluator},
    )

    diagnostics = diagnostics_for_runtime(runtime)
    assert diagnostics["stage_checks"]["canonical_reset"]
    assert not success_for_runtime(runtime, {})
    assert diagnostics_for_runtime(runtime) == diagnostics
    report = report_stages_for_runtime(runtime)
    assert report["order"] == list(REPORT_STAGE_ORDER)
    assert report["stages"]["success"]["passed"] is False


def test_wr01_runtime_registers_public_report_stage_projector() -> None:
    assert wr01_runtime.TASK_RUNTIME_API.report_stage_profile_id == (
        "WR-01/report-v1"
    )
    assert callable(wr01_runtime.TASK_RUNTIME_API.report_stages)
