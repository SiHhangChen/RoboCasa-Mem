from __future__ import annotations

from types import SimpleNamespace

import pytest

import bench.tasks.wr03.runtime as wr03_runtime
from bench.evaluation.report_stages import validate_report_stages
from bench.tasks.wr03.evaluation import (
    REPORT_STAGE_ORDER,
    STAGE_ORDER,
    TASK_EVALUATOR_METADATA_KEY,
    WR03EvaluationSnapshot,
    WR03Evaluator,
    report_stages_for_runtime,
)


def _snapshot(
    step: int,
    *,
    close_order: tuple[str, str] = ("upper", "lower"),
    grasped: bool = False,
    stored: str | None = None,
    closed_levels: frozenset[str] = frozenset(),
    hidden: bool = False,
    lower_closed: bool = False,
    upper_closed: bool = False,
    opened: str | None = None,
    wrong_opened: bool = False,
    retrieved: str | None = None,
    outside: bool = False,
    other_closed: bool = False,
    violation: bool = False,
    canonical_reset: bool = False,
    placed: bool = False,
) -> WR03EvaluationSnapshot:
    return WR03EvaluationSnapshot(
        step_id=step,
        close_order=close_order,
        can_grasped=grasped,
        stored_level=stored,
        closed_levels_after_storage=closed_levels,
        storage_hidden=hidden,
        lower_door_closed=lower_closed,
        upper_door_closed=upper_closed,
        opened_level=opened,
        wrong_retrieval_level_opened=wrong_opened,
        retrieved_item_id=retrieved,
        can_outside_both=outside,
        other_door_closed=other_closed,
        protocol_violation=violation,
        canonical_reset=canonical_reset,
        final_placement_verified=placed,
        details={"step_id": step},
    )


def _evaluator(
    close_order: tuple[str, str] = ("upper", "lower"),
) -> WR03Evaluator:
    evaluator = WR03Evaluator(close_order=close_order)
    evaluator.reset(_snapshot(0, stored="lower", close_order=close_order))
    return evaluator


def test_wr03_snapshot_rejects_hidden_storage_without_both_close_events() -> None:
    with pytest.raises(ValueError, match="both post-storage door-close events"):
        _snapshot(
            1,
            stored="lower",
            hidden=True,
            closed_levels=frozenset({"upper"}),
        )


def _complete(evaluator: WR03Evaluator) -> None:
    evaluator.observe(
        _snapshot(
            1,
            stored="lower",
            upper_closed=True,
            closed_levels=frozenset({"upper"}),
        )
    )
    evaluator.observe(
        _snapshot(
            2,
            stored="lower",
            hidden=True,
            lower_closed=True,
            upper_closed=True,
            closed_levels=frozenset({"upper", "lower"}),
        )
    )
    evaluator.observe(
        _snapshot(
            3,
            stored="lower",
            hidden=True,
            lower_closed=True,
            upper_closed=True,
            canonical_reset=True,
            closed_levels=frozenset({"upper", "lower"}),
        )
    )
    evaluator.observe(
        _snapshot(
            4,
            stored="lower",
            hidden=True,
            lower_closed=False,
            upper_closed=True,
            opened="lower",
            canonical_reset=True,
            closed_levels=frozenset({"upper", "lower"}),
        )
    )
    evaluator.observe(
        _snapshot(
            5,
            stored="lower",
            hidden=True,
            lower_closed=False,
            upper_closed=True,
            opened="lower",
            canonical_reset=True,
            closed_levels=frozenset({"upper", "lower"}),
            retrieved="red_can",
            outside=True,
            other_closed=True,
        )
    )


def test_wr03_evaluator_qualifies_ordered_stages_and_final_success() -> None:
    evaluator = _evaluator()
    _complete(evaluator)
    result = evaluator.observe(
        _snapshot(
            6,
            stored="lower",
            hidden=True,
            lower_closed=False,
            upper_closed=True,
            opened="lower",
            canonical_reset=True,
            closed_levels=frozenset({"upper", "lower"}),
            retrieved="red_can",
            outside=True,
            other_closed=True,
            placed=True,
        )
    )

    assert result.success
    assert result.failure_code is None
    assert {
        name: result.stages[name].qualified_step for name in STAGE_ORDER
    } == {
        "close_first_door": 1,
        "close_second_door": 2,
        "canonical_reset": 3,
        "open_remembered_cabinet": 4,
        "retrieve_can": 5,
        "place_on_table": 6,
    }
    diagnostics = result.to_diagnostics()
    assert diagnostics["evaluation"]["step_id"] == 6
    assert diagnostics["stage_checks"] == {name: True for name in STAGE_ORDER}

    report = result.to_report_stages()
    assert report == {
        "schema_version": "membench.report_stages/v1",
        "task_id": "WR-03",
        "profile_id": "WR-03/report-v1",
        "order": [
            "can_stored",
            "storage_hidden",
            "correct_cabinet_opened",
            "success",
        ],
        "stages": {
            "can_stored": {
                "applicable": True,
                "passed": True,
                "first_pass_step": 1,
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
    assert not {
        "close_first_door",
        "close_second_door",
        "open_remembered_cabinet",
        "retrieve_can",
        "place_on_table",
    } & set(report["stages"])
    assert validate_report_stages(
        report,
        expected_task_id="WR-03",
        expected_profile_id="WR-03/report-v1",
        expected_success=True,
    ) == report


def test_wr03_evaluator_rejects_open_before_memory_barrier() -> None:
    evaluator = _evaluator()
    result = evaluator.observe(
        _snapshot(
            1,
            stored="lower",
            opened="lower",
            closed_levels=frozenset(),
        )
    )

    assert result.stages["open_remembered_cabinet"].observed
    assert not result.stages["open_remembered_cabinet"].qualified
    assert result.stages["open_remembered_cabinet"].reason == (
        "prerequisite_not_qualified:canonical_reset"
    )


def test_wr03_evaluator_qualifies_both_close_stages_from_one_sample() -> None:
    evaluator = _evaluator()

    result = evaluator.observe(
        _snapshot(
            1,
            stored="lower",
            hidden=True,
            lower_closed=True,
            upper_closed=True,
            closed_levels=frozenset({"upper", "lower"}),
        )
    )

    assert result.stages["close_first_door"].qualified_step == 1
    assert result.stages["close_second_door"].qualified_step == 1
    assert not result.stages["canonical_reset"].qualified


def test_wr03_evaluator_accepts_either_door_close_order() -> None:
    close_order = ("lower", "upper")
    evaluator = _evaluator(close_order)
    lower_first = evaluator.observe(
        _snapshot(
            1,
            close_order=close_order,
            stored="lower",
            lower_closed=True,
            closed_levels=frozenset({"lower"}),
        )
    )
    hidden = evaluator.observe(
        _snapshot(
            2,
            close_order=close_order,
            stored="lower",
            hidden=True,
            lower_closed=True,
            upper_closed=True,
            closed_levels=frozenset({"upper", "lower"}),
        )
    )

    assert lower_first.stages["close_first_door"].qualified_step == 1
    assert not lower_first.stages["close_second_door"].qualified
    assert hidden.stages["close_second_door"].qualified_step == 2
    assert not hidden.stages["canonical_reset"].qualified
    assert hidden.to_report_stages()["stages"]["storage_hidden"] == {
        "applicable": True,
        "passed": False,
        "first_pass_step": None,
    }


def test_wr03_closed_door_events_do_not_replace_the_memory_barrier() -> None:
    evaluator = _evaluator()
    result = evaluator.observe(
        _snapshot(
            1,
            stored="lower",
            lower_closed=True,
            upper_closed=True,
            closed_levels=frozenset({"upper", "lower"}),
        )
    )

    assert result.stages["close_first_door"].qualified
    assert result.stages["close_second_door"].qualified
    assert not result.stages["canonical_reset"].qualified
    assert not result.to_report_stages()["stages"]["storage_hidden"]["passed"]


def test_wr03_evaluator_makes_wrong_retrieval_cabinet_terminal() -> None:
    evaluator = _evaluator()
    evaluator.observe(
        _snapshot(
            1,
            stored="lower",
            upper_closed=True,
            closed_levels=frozenset({"upper"}),
        )
    )
    evaluator.observe(
        _snapshot(
            2,
            stored="lower",
            hidden=True,
            lower_closed=True,
            upper_closed=True,
            closed_levels=frozenset({"upper", "lower"}),
        )
    )
    evaluator.observe(
        _snapshot(
            3,
            stored="lower",
            hidden=True,
            lower_closed=True,
            upper_closed=True,
            canonical_reset=True,
            closed_levels=frozenset({"upper", "lower"}),
        )
    )
    result = evaluator.observe(
        _snapshot(
            4,
            stored="lower",
            hidden=True,
            lower_closed=True,
            upper_closed=False,
            opened="upper",
            wrong_opened=True,
            canonical_reset=True,
            closed_levels=frozenset({"upper", "lower"}),
        )
    )

    assert result.terminal
    assert not result.success
    assert result.failure_code == "wrong_retrieval_cabinet"
    assert [
        result.to_report_stages()["stages"][name]["passed"]
        for name in REPORT_STAGE_ORDER
    ] == [True, True, False, False]


def test_wr03_success_is_revoked_if_final_state_regresses() -> None:
    evaluator = _evaluator()
    _complete(evaluator)
    success = evaluator.observe(
        _snapshot(
            6,
            stored="lower",
            hidden=True,
            opened="lower",
            canonical_reset=True,
            closed_levels=frozenset({"upper", "lower"}),
            retrieved="red_can",
            outside=True,
            other_closed=True,
            placed=True,
        )
    )
    regressed = evaluator.observe(
        _snapshot(
            7,
            stored="lower",
            hidden=True,
            opened="lower",
            canonical_reset=True,
            closed_levels=frozenset({"upper", "lower"}),
            retrieved="red_can",
            outside=True,
            other_closed=False,
            placed=True,
        )
    )

    assert success.success
    assert not regressed.success
    assert regressed.stages["retrieve_can"].qualified
    assert not regressed.to_report_stages()["stages"]["success"]["passed"]


def test_wr03_runtime_report_callback_uses_initialized_evaluator() -> None:
    evaluator = _evaluator()

    class FakeEnv:
        snapshot = _snapshot(1, grasped=True)

        def evaluation_snapshot(self) -> WR03EvaluationSnapshot:
            return self.snapshot

    runtime = SimpleNamespace(
        env=FakeEnv(),
        metadata={TASK_EVALUATOR_METADATA_KEY: evaluator},
    )

    report = report_stages_for_runtime(runtime)
    assert report["order"] == list(REPORT_STAGE_ORDER)
    assert not any(stage["passed"] for stage in report["stages"].values())


def test_wr03_runtime_registers_public_report_stage_projector() -> None:
    assert wr03_runtime.TASK_RUNTIME_API.report_stage_profile_id == (
        "WR-03/report-v1"
    )
    assert callable(wr03_runtime.TASK_RUNTIME_API.report_stages)
