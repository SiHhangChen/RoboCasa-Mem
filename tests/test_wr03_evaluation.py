from __future__ import annotations

from bench.tasks.wr03.evaluation import (
    STAGE_ORDER,
    WR03EvaluationSnapshot,
    WR03Evaluator,
)


def _snapshot(
    step: int,
    *,
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
) -> WR03EvaluationSnapshot:
    return WR03EvaluationSnapshot(
        step_id=step,
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
        details={"step_id": step},
    )


def _evaluator() -> WR03Evaluator:
    evaluator = WR03Evaluator()
    evaluator.reset(_snapshot(0))
    return evaluator


def _complete(evaluator: WR03Evaluator) -> None:
    evaluator.observe(_snapshot(1, grasped=True))
    evaluator.observe(_snapshot(2, stored="lower"))
    evaluator.observe(
        _snapshot(
            3,
            stored="lower",
            upper_closed=True,
            closed_levels=frozenset({"upper"}),
        )
    )
    evaluator.observe(
        _snapshot(
            4,
            stored="lower",
            hidden=True,
            lower_closed=True,
            upper_closed=True,
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
            closed_levels=frozenset({"upper", "lower"}),
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
            closed_levels=frozenset({"upper", "lower"}),
            retrieved="red_can",
            outside=True,
            other_closed=True,
        )
    )

    assert result.success
    assert result.failure_code is None
    assert {
        name: result.stages[name].qualified_step for name in STAGE_ORDER
    } == {
        "can_pick": 1,
        "can_store": 2,
        "close_upper_cabinet": 3,
        "close_lower_cabinet": 4,
        "open_remembered_cabinet": 5,
        "retrieve_can": 6,
    }
    diagnostics = result.to_diagnostics()
    assert diagnostics["evaluation"]["step_id"] == 6
    assert diagnostics["stage_checks"] == {name: True for name in STAGE_ORDER}


def test_wr03_evaluator_rejects_open_before_memory_barrier() -> None:
    evaluator = _evaluator()
    evaluator.observe(_snapshot(1, grasped=True))
    evaluator.observe(_snapshot(2, stored="lower"))
    result = evaluator.observe(
        _snapshot(
            3,
            stored="lower",
            opened="lower",
            closed_levels=frozenset(),
        )
    )

    assert result.stages["open_remembered_cabinet"].observed
    assert not result.stages["open_remembered_cabinet"].qualified
    assert result.stages["open_remembered_cabinet"].reason == (
        "prerequisite_not_qualified:close_lower_cabinet"
    )


def test_wr03_evaluator_qualifies_both_close_stages_from_one_sample() -> None:
    evaluator = _evaluator()
    evaluator.observe(_snapshot(1, grasped=True))
    evaluator.observe(_snapshot(2, stored="lower"))

    result = evaluator.observe(
        _snapshot(
            3,
            stored="lower",
            hidden=True,
            lower_closed=True,
            upper_closed=True,
            closed_levels=frozenset({"upper", "lower"}),
        )
    )

    assert result.stages["close_upper_cabinet"].qualified_step == 3
    assert result.stages["close_lower_cabinet"].qualified_step == 3


def test_wr03_evaluator_makes_wrong_retrieval_cabinet_terminal() -> None:
    evaluator = _evaluator()
    evaluator.observe(_snapshot(1, grasped=True))
    evaluator.observe(_snapshot(2, stored="lower"))
    evaluator.observe(
        _snapshot(
            3,
            stored="lower",
            hidden=True,
            upper_closed=True,
            closed_levels=frozenset({"upper"}),
        )
    )
    evaluator.observe(
        _snapshot(
            4,
            stored="lower",
            hidden=True,
            lower_closed=True,
            upper_closed=True,
            closed_levels=frozenset({"upper", "lower"}),
        )
    )
    result = evaluator.observe(
        _snapshot(
            6,
            stored="lower",
            hidden=True,
            lower_closed=True,
            upper_closed=False,
            opened="upper",
            wrong_opened=True,
            closed_levels=frozenset({"upper", "lower"}),
        )
    )

    assert result.terminal
    assert not result.success
    assert result.failure_code == "wrong_retrieval_cabinet"


def test_wr03_success_is_revoked_if_final_state_regresses() -> None:
    evaluator = _evaluator()
    _complete(evaluator)
    success = evaluator.observe(
        _snapshot(
            6,
            stored="lower",
            hidden=True,
            opened="lower",
            closed_levels=frozenset({"upper", "lower"}),
            retrieved="red_can",
            outside=True,
            other_closed=True,
        )
    )
    regressed = evaluator.observe(
        _snapshot(
            7,
            stored="lower",
            hidden=True,
            opened="lower",
            closed_levels=frozenset({"upper", "lower"}),
            retrieved="red_can",
            outside=True,
            other_closed=False,
        )
    )

    assert success.success
    assert not regressed.success
    assert regressed.stages["retrieve_can"].qualified
