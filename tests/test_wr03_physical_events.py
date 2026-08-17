from __future__ import annotations

from bench.tasks.wr03.domain.semantics import (
    PhysicalState,
    SemanticState,
    evaluate_success,
    record_storage_door_closed,
)
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


def test_wr03_observer_latches_wrong_retrieval_door() -> None:
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
    assert state.retrieval_level_opened == "lower"
    assert state.wrong_retrieval_level_opened


def test_wr03_observer_ignores_one_step_containment_and_door_transients() -> None:
    state = SemanticState()
    observer = WR03PhysicalObserver(state, dwell_steps=3)
    observer.reset(_sample(0, lower_open=True, upper_open=True))
    observer.observe(_sample(1, can="lower", lower_open=True, upper_open=True))
    observer.observe(_sample(2, lower_open=True, upper_open=True))
    assert state.stored_level is None

    step = _repeat(observer, 3, 3, can="upper", lower_open=True, upper_open=True)
    assert state.stored_level == "upper"
    observer.observe(
        _sample(step, can="upper", lower_closed=True, upper_closed=True)
    )
    observer.observe(_sample(step + 1, can="upper"))
    assert not state.storage_hidden


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
