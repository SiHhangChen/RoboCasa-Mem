from __future__ import annotations

import pytest

from bench.tasks.wr03.domain.semantics import (
    PhysicalState,
    SemanticState,
    evaluate_success,
    latch_storage_hidden,
    record_retrieval_door_open,
    record_retrieval_pick,
    record_storage,
    record_storage_door_closed,
)
from bench.tasks.wr03.evaluation import WR03EvaluationSnapshot, WR03Evaluator
from bench.tasks.wr03.physical_events import PhysicalSample, WR03PhysicalObserver


def _sample(
    step: int,
    *,
    can: str | None = None,
    lower_closed: bool = False,
    upper_closed: bool = False,
    lower_open: bool = False,
    upper_open: bool = False,
) -> PhysicalSample:
    return PhysicalSample(
        step_id=step,
        can_in_lower=can == "lower",
        can_in_upper=can == "upper",
        lower_door_closed=lower_closed,
        upper_door_closed=upper_closed,
        lower_door_open=lower_open,
        upper_door_open=upper_open,
    )


def _repeat(observer: WR03PhysicalObserver, start: int, count: int, **kwargs) -> int:
    for step in range(start, start + count):
        observer.observe(_sample(step, **kwargs))
    return start + count


def test_wr03_observer_reconstructs_lower_storage_and_retrieval() -> None:
    state = SemanticState()
    observer = WR03PhysicalObserver(state, dwell_steps=2)
    observer.reset(_sample(0, lower_open=True, upper_open=True))

    step = _repeat(observer, 1, 2, can="lower", lower_open=True, upper_open=True)
    assert state.stored_level == "lower"

    step = _repeat(
        observer,
        step,
        2,
        can="lower",
        lower_closed=True,
        upper_closed=True,
    )
    assert state.closed_levels_after_storage == {"lower", "upper"}
    assert state.storage_hidden

    step = _repeat(
        observer,
        step,
        2,
        can="lower",
        lower_open=True,
        upper_closed=True,
    )
    assert state.retrieval_level_opened == "lower"

    _repeat(
        observer,
        step,
        2,
        lower_open=True,
        upper_closed=True,
    )
    assert state.retrieved_item_id == "red_can"
    report = evaluate_success(
        state,
        PhysicalState(
            can_in_lower=False,
            can_in_upper=False,
            lower_door_closed=False,
            upper_door_closed=True,
        ),
    )
    assert report.success


def test_wr03_observer_recovers_after_opening_the_empty_cabinet_first() -> None:
    state = SemanticState()
    observer = WR03PhysicalObserver(state, dwell_steps=1)
    observer.reset(_sample(0, lower_open=True, upper_open=True))
    observer.observe(_sample(1, can="upper", lower_open=True, upper_open=True))
    observer.observe(
        _sample(2, can="upper", lower_closed=True, upper_closed=True)
    )
    assert state.storage_hidden

    observer.observe(
        _sample(3, can="upper", lower_open=True, upper_closed=True)
    )
    assert state.retrieval_level_opened is None
    assert state.wrong_retrieval_level_opened
    assert not state.protocol_violation

    opened = observer.observe(
        _sample(4, can="upper", lower_open=True, upper_open=True)
    )
    assert state.retrieval_level_opened == "upper"
    assert opened.diagnostics["phase"] == WR03PhysicalObserver.RETRIEVE
    observer.observe(_sample(5, lower_open=True, upper_open=True))
    assert state.retrieved_item_id == "red_can"
    report = evaluate_success(state, PhysicalState(False, False, False, False))
    assert report.success
    assert "no_wrong_retrieval_door" not in report.checks()
    assert "other_door_remains_closed" not in report.checks()


def test_wr03_observer_requires_consecutive_containment_but_no_closing_dwell() -> None:
    state = SemanticState()
    observer = WR03PhysicalObserver(state, dwell_steps=3)
    observer.reset(_sample(0, lower_open=True, upper_open=True))
    observer.observe(_sample(1, can="lower", lower_open=True, upper_open=True))
    observer.observe(_sample(2, lower_open=True, upper_open=True))
    assert state.stored_level is None

    step = _repeat(observer, 3, 2, can="lower", lower_open=True, upper_open=True)
    assert state.stored_level is None
    observer.observe(_sample(step, can="lower", lower_open=True, upper_open=True))
    assert state.stored_level == "lower"
    step += 1
    observer.observe(
        _sample(step, can="lower", lower_closed=True, upper_closed=True)
    )
    assert state.storage_hidden
    assert state.closed_levels_after_storage == {"lower", "upper"}
    assert not state.canonical_reset_complete

    observer.observe(_sample(step + 1, can="lower", lower_open=True))
    observer.observe(_sample(step + 2, can="lower"))
    assert state.retrieval_level_opened is None


def test_wr03_observer_deduplicates_same_simulator_step() -> None:
    state = SemanticState()
    observer = WR03PhysicalObserver(state, dwell_steps=2)
    observer.reset(_sample(0, lower_open=True, upper_open=True))
    first = observer.observe(
        _sample(1, can="lower", lower_open=True, upper_open=True)
    )
    duplicate = observer.observe(
        _sample(1, can="lower", lower_open=True, upper_open=True)
    )
    assert not first.duplicate_step
    assert duplicate.duplicate_step
    assert state.stored_level is None


def test_wr03_repeated_close_ack_is_idempotent_after_memory_barrier() -> None:
    state = SemanticState(
        stored_level="lower",
        closed_levels_after_storage={"lower", "upper"},
        storage_hidden=True,
    )

    assert record_storage_door_closed(state, "lower")
    assert not state.protocol_violation


@pytest.mark.parametrize("stored", ["lower", "upper"])
def test_wr03_target_opening_requires_three_consecutive_observations(stored) -> None:
    state = SemanticState(stored_level=stored)
    observer = WR03PhysicalObserver(state)
    observer.reset(_sample(0, can=stored, lower_open=True, upper_open=True))
    observer.observe(_sample(1, can=stored, lower_closed=True, upper_closed=True))
    assert state.storage_hidden

    for step in (2, 3):
        observer.observe(_sample(step, can=stored, lower_open=True, upper_open=True))
    assert state.retrieval_level_opened is None
    # An interrupted opening must reset the target's dwell even if the empty
    # door stays open; it must not force the observer into retrieval either.
    empty = "upper" if stored == "lower" else "lower"
    observer.observe(_sample(4, can=stored, **{f"{empty}_open": True}))
    for step in (5, 6):
        observer.observe(_sample(step, can=stored, lower_open=True, upper_open=True))
        assert state.retrieval_level_opened is None
    observer.observe(_sample(7, can=stored, lower_open=True, upper_open=True))
    assert state.retrieval_level_opened == stored


def test_wr03_retrieval_requires_three_consecutive_outside_observations() -> None:
    state = SemanticState(stored_level="lower", storage_hidden=True,
                          retrieval_level_opened="lower")
    observer = WR03PhysicalObserver(state)
    observer.reset(_sample(0, can="lower", lower_open=True, upper_open=True))
    _repeat(observer, 1, 2, lower_open=True, upper_open=True)
    assert state.retrieved_item_id is None
    observer.observe(_sample(3, can="lower", lower_open=True, upper_open=True))
    _repeat(observer, 4, 2, lower_open=True, upper_open=True)
    assert state.retrieved_item_id is None
    observer.observe(_sample(6, lower_open=True, upper_open=True))
    assert state.retrieved_item_id == "red_can"


def test_wr03_memory_barrier_requires_current_storage_and_simultaneous_closure() -> None:
    state = SemanticState(stored_level="lower")
    observer = WR03PhysicalObserver(state)
    observer.reset(_sample(0, can="lower", lower_open=True, upper_open=True))
    _repeat(observer, 1, 3, can="lower", lower_closed=True, upper_open=True)
    _repeat(observer, 4, 3, can="lower", lower_open=True, upper_closed=True)
    assert state.closed_levels_after_storage == {"lower", "upper"}
    assert not state.storage_hidden

    observer.observe(_sample(7, can="upper", lower_closed=True, upper_closed=True))
    assert not state.storage_hidden
    observer.observe(_sample(8, lower_closed=True, upper_closed=True))
    assert not state.storage_hidden
    observer.observe(_sample(9, can="lower", lower_closed=True, upper_closed=True))
    assert state.storage_hidden


def test_wr03_memory_barrier_needs_no_historical_close_events_or_neutral_pose() -> None:
    state = SemanticState(stored_level="lower")
    assert not state.closed_levels_after_storage
    assert not state.canonical_reset_complete
    assert latch_storage_hidden(state, can_still_stored=True, cabinets_closed=True)
    assert record_storage_door_closed(state, "lower")
    assert record_storage_door_closed(state, "upper")
    assert not state.protocol_violation


def test_wr03_conflicting_storage_assignment_is_still_a_protocol_violation() -> None:
    state = SemanticState(stored_level="lower")
    assert not record_storage(state, "red_can", "upper")
    assert state.protocol_violation
    assert not latch_storage_hidden(state, can_still_stored=True, cabinets_closed=True)


@pytest.mark.parametrize("event", ["door", "pick"])
def test_wr03_semantic_retrieval_without_memory_barrier_is_still_invalid(event) -> None:
    state = SemanticState(stored_level="lower")
    if event == "door":
        assert not record_retrieval_door_open(state, "lower")
    else:
        assert not record_retrieval_pick(state, "red_can", physically_extracted=True)
    assert state.protocol_violation


@pytest.mark.parametrize("stored", ["lower", "upper"])
@pytest.mark.parametrize("first_closed", ["lower", "upper"])
def test_wr03_observer_and_evaluator_complete_after_wrong_door_first(stored, first_closed):
    state = SemanticState(stored_level=stored)
    observer = WR03PhysicalObserver(state)
    evaluator = WR03Evaluator()

    def snapshot(sample, *, placed=False):
        return WR03EvaluationSnapshot(
            step_id=sample.step_id,
            stored_level=state.stored_level,
            storage_hidden=state.storage_hidden,
            lower_door_closed=sample.lower_door_closed,
            upper_door_closed=sample.upper_door_closed,
            opened_level=state.retrieval_level_opened,
            retrieved_item_id=state.retrieved_item_id,
            can_outside_both=sample.can_outside_both,
            protocol_violation=state.protocol_violation,
            final_placement_verified=placed,
        )

    def observe(step, *, can=stored, placed=False, **kwargs):
        sample = _sample(step, can=can, **kwargs)
        observer.observe(sample)
        return evaluator.observe(snapshot(sample, placed=placed))

    initial = _sample(0, can=stored, lower_open=True, upper_open=True)
    observer.reset(initial)
    evaluator.reset(snapshot(initial))
    first = observe(1, **{f"{first_closed}_closed": True})
    assert first.stages["close_first_door"].qualified_step == 1
    closed = observe(2, lower_closed=True, upper_closed=True)
    assert state.storage_hidden
    assert closed.stages["close_second_door"].qualified_step == 2
    assert not state.canonical_reset_complete
    empty = "upper" if stored == "lower" else "lower"
    for step in (3, 4, 5):
        result = observe(step, **{f"{empty}_open": True})
    assert state.wrong_retrieval_level_opened
    assert not result.terminal
    for step in (6, 7, 8):
        result = observe(step, lower_open=True, upper_open=True)
    assert result.stages["open_remembered_cabinet"].qualified_step == 8
    for step in (9, 10, 11):
        result = observe(step, can=None, lower_open=True, upper_open=True)
    assert result.stages["retrieve_can"].qualified_step == 11
    assert not result.success
    result = observe(12, can=None, placed=True, lower_open=True, upper_open=True)
    assert result.success
