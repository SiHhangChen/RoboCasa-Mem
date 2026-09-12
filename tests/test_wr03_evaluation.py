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
    stored: str | None = "lower",
    hidden: bool = False,
    lower_closed: bool = False,
    upper_closed: bool = False,
    opened: str | None = None,
    retrieved: str | None = None,
    outside: bool = False,
    violation: bool = False,
    placed: bool = False,
    details: dict | None = None,
) -> WR03EvaluationSnapshot:
    return WR03EvaluationSnapshot(
        step_id=step,
        stored_level=stored,
        storage_hidden=hidden,
        lower_door_closed=lower_closed,
        upper_door_closed=upper_closed,
        opened_level=opened,
        retrieved_item_id=retrieved,
        can_outside_both=outside,
        protocol_violation=violation,
        final_placement_verified=placed,
        details={"step_id": step, **(details or {})},
    )


def _evaluator(*, stored: str = "lower") -> WR03Evaluator:
    evaluator = WR03Evaluator()
    evaluator.reset(_snapshot(0, stored=stored))
    return evaluator


def _close_both(evaluator: WR03Evaluator, *, stored: str = "lower") -> None:
    evaluator.observe(_snapshot(1, stored=stored, upper_closed=True))
    evaluator.observe(
        _snapshot(2, stored=stored, hidden=True, lower_closed=True, upper_closed=True)
    )


def _retrieve(evaluator: WR03Evaluator, *, stored: str = "lower") -> None:
    _close_both(evaluator, stored=stored)
    # Both doors may stay open throughout opening, retrieval, and placement.
    evaluator.observe(_snapshot(3, stored=stored, hidden=True, opened=stored))
    evaluator.observe(
        _snapshot(4, stored=stored, hidden=True, opened=stored,
                  retrieved="red_can", outside=True)
    )


def _placement(step: int = 5, *, stored: str = "lower", **kwargs):
    return _snapshot(
        step, stored=stored, hidden=True, opened=stored, retrieved="red_can",
        outside=True, placed=True, **kwargs,
    )


def test_wr03_snapshot_rejects_hidden_storage_without_a_stored_can() -> None:
    with pytest.raises(ValueError, match="requires a stored can"):
        _snapshot(1, stored=None, hidden=True)


@pytest.mark.parametrize("stored", ["lower", "upper"])
def test_wr03_evaluator_qualifies_five_stages_without_neutral_reset(stored) -> None:
    evaluator = _evaluator(stored=stored)
    _retrieve(evaluator, stored=stored)
    assert not evaluator.result().success
    result = evaluator.observe(_placement(stored=stored))

    assert result.success and result.terminal
    assert result.failure_code is None
    assert STAGE_ORDER == (
        "close_first_door", "close_second_door", "open_remembered_cabinet",
        "retrieve_can", "place_on_table",
    )
    assert {
        name: result.stages[name].qualified_step for name in STAGE_ORDER
    } == dict(zip(STAGE_ORDER, range(1, 6)))
    diagnostics = result.to_diagnostics()
    assert diagnostics["evaluation"]["step_id"] == 5
    assert diagnostics["evaluation"]["door_order_policy"] == "unordered"
    assert diagnostics["stage_checks"] == {name: True for name in STAGE_ORDER}
    assert "canonical_reset" not in diagnostics["stage_checks"]

    report = result.to_report_stages()
    assert report == {
        "schema_version": "membench.report_stages/v1",
        "task_id": "WR-03",
        "profile_id": "WR-03/report-v2",
        "order": [
            "one_door_closed", "both_doors_closed",
            "correct_cabinet_opened", "success",
        ],
        "stages": {
            name: {"applicable": True, "passed": True, "first_pass_step": step}
            for name, step in zip(REPORT_STAGE_ORDER, (1, 2, 3, 5))
        },
    }
    assert validate_report_stages(
        report, expected_task_id="WR-03",
        expected_profile_id="WR-03/report-v2", expected_success=True,
    ) == report


@pytest.mark.parametrize("stored", ["lower", "upper"])
@pytest.mark.parametrize("first", ["lower", "upper"])
def test_wr03_evaluator_accepts_either_close_order_without_configuration(stored, first):
    evaluator = _evaluator(stored=stored)
    first_closed = evaluator.observe(
        _snapshot(1, stored=stored, **{f"{first}_closed": True})
    )
    both_closed = evaluator.observe(
        _snapshot(2, stored=stored, hidden=True, lower_closed=True, upper_closed=True)
    )

    assert first_closed.stages["close_first_door"].qualified_step == 1
    assert not first_closed.stages["close_second_door"].qualified
    assert both_closed.stages["close_second_door"].qualified_step == 2
    assert both_closed.to_report_stages()["stages"]["both_doors_closed"] == {
        "applicable": True, "passed": True, "first_pass_step": 2,
    }


def test_wr03_evaluator_qualifies_both_close_stages_from_one_sample() -> None:
    evaluator = _evaluator()
    result = evaluator.observe(
        _snapshot(1, hidden=True, lower_closed=True, upper_closed=True)
    )
    assert result.stages["close_first_door"].qualified_step == 1
    assert result.stages["close_second_door"].qualified_step == 1
    assert not result.stages["open_remembered_cabinet"].qualified


def test_wr03_closing_each_door_separately_does_not_count_as_both_closed() -> None:
    evaluator = _evaluator()
    for step, first in enumerate(("upper", "lower", "upper", "lower"), start=1):
        result = evaluator.observe(_snapshot(step, **{f"{first}_closed": True}))
        assert result.stages["close_first_door"].qualified
        assert not result.stages["close_second_door"].qualified


def test_wr03_late_storage_evidence_does_not_require_reclosing_the_doors() -> None:
    evaluator = WR03Evaluator()
    evaluator.reset(_snapshot(0, stored=None))
    waiting = evaluator.observe(
        _snapshot(1, stored=None, lower_closed=True, upper_closed=True)
    )
    assert not waiting.stages["close_first_door"].qualified

    result = evaluator.observe(
        _snapshot(2, hidden=True, lower_closed=True, upper_closed=True)
    )
    assert result.stages["close_first_door"].qualified_step == 2
    assert result.stages["close_second_door"].qualified_step == 2


def test_wr03_opening_without_memory_barrier_never_qualifies() -> None:
    evaluator = _evaluator()
    evaluator.observe(_snapshot(1, lower_closed=True, upper_closed=True))
    result = evaluator.observe(_snapshot(2, opened="lower"))
    assert result.stages["close_second_door"].qualified
    assert not result.stages["open_remembered_cabinet"].observed
    assert not result.stages["open_remembered_cabinet"].qualified
    assert not result.terminal


def test_wr03_memory_barrier_does_not_replace_closing_stage_prerequisites() -> None:
    evaluator = _evaluator()
    result = evaluator.observe(_snapshot(1, hidden=True, opened="lower"))
    assert not result.stages["open_remembered_cabinet"].qualified
    assert result.stages["open_remembered_cabinet"].reason == (
        "prerequisite_not_qualified:close_second_door"
    )


@pytest.mark.parametrize("stored,empty", [("lower", "upper"), ("upper", "lower")])
def test_wr03_wrong_door_first_is_recoverable_and_other_door_can_stay_open(stored, empty):
    evaluator = _evaluator(stored=stored)
    _close_both(evaluator, stored=stored)
    wrong = evaluator.observe(
        _snapshot(3, stored=stored, hidden=True, opened=empty,
                  details={"retrieval": {"wrong_level_opened": True}})
    )
    assert not wrong.terminal
    assert wrong.failure_code is None
    assert [
        wrong.to_report_stages()["stages"][name]["passed"]
        for name in REPORT_STAGE_ORDER
    ] == [True, True, False, False]

    opened = evaluator.observe(
        _snapshot(4, stored=stored, hidden=True, opened=stored)
    )
    assert opened.stages["open_remembered_cabinet"].qualified_step == 4
    evaluator.observe(
        _snapshot(5, stored=stored, hidden=True, opened=stored,
                  retrieved="red_can", outside=True)
    )
    assert evaluator.observe(_placement(6, stored=stored)).success


def test_wr03_retrieval_requires_opening_the_target_cabinet() -> None:
    evaluator = _evaluator()
    _close_both(evaluator)
    result = evaluator.observe(
        _snapshot(3, hidden=True, retrieved="red_can", outside=True, placed=True)
    )
    assert not result.stages["retrieve_can"].qualified
    assert result.stages["retrieve_can"].reason == (
        "prerequisite_not_qualified:open_remembered_cabinet"
    )
    assert not result.stages["place_on_table"].qualified
    assert not result.success


def test_wr03_retrieval_requires_actual_extraction() -> None:
    evaluator = _evaluator()
    _close_both(evaluator)
    evaluator.observe(_snapshot(3, hidden=True, opened="lower"))
    result = evaluator.observe(
        _snapshot(4, hidden=True, opened="lower", retrieved="red_can", placed=True)
    )
    assert result.terminal and not result.success
    assert result.failure_code == "can_not_extracted"
    assert not result.stages["place_on_table"].qualified
    assert evaluator.observe(_placement()).failure_code == "can_not_extracted"


def test_wr03_premature_placement_requires_a_new_edge_after_prerequisites() -> None:
    evaluator = _evaluator()
    evaluator.observe(_snapshot(1, upper_closed=True, placed=True))
    evaluator.observe(
        _snapshot(2, hidden=True, lower_closed=True, upper_closed=True, placed=True)
    )
    evaluator.observe(_snapshot(3, hidden=True, opened="lower", placed=True))
    evaluator.observe(_placement(4))
    # The earlier placement was rejected, not retroactively qualified.
    assert not evaluator.result().stages["place_on_table"].qualified
    evaluator.observe(
        _snapshot(5, hidden=True, opened="lower", retrieved="red_can", outside=True)
    )
    assert evaluator.observe(_placement(6)).success


@pytest.mark.parametrize("at_reset", [False, True])
def test_wr03_genuine_protocol_violation_is_terminal(at_reset) -> None:
    evaluator = WR03Evaluator()
    if at_reset:
        result = evaluator.reset(_snapshot(0, violation=True))
    else:
        evaluator.reset(_snapshot(0))
        _retrieve(evaluator)
        result = evaluator.observe(_placement(violation=True))
    assert result.terminal and not result.success
    assert result.failure_code == "protocol_violation"
    assert not result.stages["place_on_table"].qualified
    assert evaluator.observe(_placement(10)) is result


def test_wr03_terminal_success_is_cached() -> None:
    evaluator = _evaluator()
    _retrieve(evaluator)
    success = evaluator.observe(_placement())
    regressed = evaluator.observe(_snapshot(6))
    assert success.success
    assert regressed is success
    assert regressed.to_report_stages()["stages"]["success"]["passed"]


def test_wr03_same_step_is_idempotent_and_backwards_steps_are_rejected() -> None:
    evaluator = _evaluator()
    result = evaluator.observe(_snapshot(1, upper_closed=True))
    assert evaluator.observe(_snapshot(1, lower_closed=True, upper_closed=True)) is result
    with pytest.raises(ValueError, match="moved backwards"):
        evaluator.observe(_snapshot(0))


def test_wr03_conditions_present_at_reset_are_not_qualified() -> None:
    evaluator = WR03Evaluator()
    result = evaluator.reset(
        _snapshot(0, hidden=True, lower_closed=True, upper_closed=True)
    )
    assert result.stages["close_first_door"].observed
    assert not result.stages["close_first_door"].qualified
    assert result.stages["close_first_door"].reason == "present_at_reset"
    repeated = evaluator.observe(
        _snapshot(1, hidden=True, lower_closed=True, upper_closed=True)
    )
    assert not repeated.stages["close_second_door"].qualified


def test_wr03_diagnostics_keep_authoritative_raw_checks_and_rejections() -> None:
    evaluator = _evaluator()
    result = evaluator.observe(
        _snapshot(1, placed=True, details={
            "raw_stage_checks": {"can_pick": True},
            "rejected_stage_observations": {"legacy": []},
        })
    )
    diagnostics = result.to_diagnostics()["evaluation"]
    assert set(diagnostics["raw_stage_checks"]) == set(STAGE_ORDER)
    assert diagnostics["raw_stage_checks"]["place_on_table"]
    assert diagnostics["rejected_stage_observations"]["place_on_table"] == [{
        "step": 1, "reason": "prerequisite_not_qualified:retrieve_can",
    }]


def test_wr03_runtime_report_callback_uses_initialized_evaluator() -> None:
    evaluator = _evaluator()

    class FakeEnv:
        def evaluation_snapshot(self) -> WR03EvaluationSnapshot:
            return _snapshot(1)

    runtime = SimpleNamespace(
        env=FakeEnv(), metadata={TASK_EVALUATOR_METADATA_KEY: evaluator},
    )
    report = report_stages_for_runtime(runtime)
    assert report["order"] == list(REPORT_STAGE_ORDER)
    assert not any(stage["passed"] for stage in report["stages"].values())


def test_wr03_runtime_registers_public_report_stage_projector() -> None:
    assert wr03_runtime.TASK_RUNTIME_API.report_stage_profile_id == "WR-03/report-v2"
    assert callable(wr03_runtime.TASK_RUNTIME_API.report_stages)
