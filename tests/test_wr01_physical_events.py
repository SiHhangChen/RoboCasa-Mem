from __future__ import annotations

from bench.tasks.wr01.domain.semantics import PhysicalState, SemanticState, evaluate_success
from bench.tasks.wr01.physical_events import PhysicalSample, WR01PhysicalObserver


def _sample(
    step: int,
    *,
    red: str | None = None,
    blue: str | None = None,
    lower_closed: bool = False,
    upper_closed: bool = False,
    lower_open: bool = False,
    upper_open: bool = False,
) -> PhysicalSample:
    return PhysicalSample(
        step_id=step,
        red_can_in_lower=red == "lower",
        red_can_in_upper=red == "upper",
        blue_can_in_lower=blue == "lower",
        blue_can_in_upper=blue == "upper",
        lower_door_closed=lower_closed,
        upper_door_closed=upper_closed,
        lower_door_open=lower_open,
        upper_door_open=upper_open,
    )


def _repeat(observer: WR01PhysicalObserver, start: int, count: int, **kwargs) -> int:
    for step in range(start, start + count):
        observer.observe(_sample(step, **kwargs))
    return start + count


def test_wr01_observer_reconstructs_two_item_memory_protocol() -> None:
    state = SemanticState("blue_can", "lower")
    observer = WR01PhysicalObserver(state, dwell_steps=2)
    observer.reset(_sample(0, lower_open=True, upper_open=True))

    step = _repeat(observer, 1, 2, red="upper", blue="lower", lower_open=True, upper_open=True)
    step = _repeat(observer, step, 2, red="upper", blue="lower", lower_closed=True, upper_closed=True)
    step = _repeat(observer, step, 2, red="upper", blue="lower", lower_closed=True, upper_closed=True)
    step = _repeat(observer, step, 2, red="upper", lower_open=True, upper_closed=True)
    _repeat(observer, step, 2, red="upper", lower_open=True, upper_closed=True)

    assert state.stored_levels_by_item == {"red_can": "upper", "blue_can": "lower"}
    assert state.storage_hidden
    assert state.retrieval_level_opened == "lower"
    assert state.retrieved_item_id == "blue_can"
    assert observer.phase == observer.COMPLETE
    assert evaluate_success(state, PhysicalState(distractor_still_stored=True)).success


def test_wr01_observer_latches_wrong_retrieval_door() -> None:
    state = SemanticState("blue_can", "lower")
    observer = WR01PhysicalObserver(state, dwell_steps=1)
    observer.reset(_sample(0, lower_open=True, upper_open=True))
    observer.observe(_sample(1, red="upper", blue="lower", lower_open=True, upper_open=True))
    observer.observe(_sample(2, red="upper", blue="lower", lower_closed=True, upper_closed=True))
    observer.observe(_sample(3, red="upper", blue="lower", lower_closed=True, upper_closed=True))
    observer.observe(_sample(4, red="upper", blue="lower", upper_open=True, lower_closed=True))
    assert state.storage_hidden
    assert state.retrieval_level_opened == "upper"
    assert state.wrong_retrieval_level_opened


def test_wr01_observer_deduplicates_simulator_step() -> None:
    state = SemanticState("blue_can", "lower")
    observer = WR01PhysicalObserver(state, dwell_steps=2)
    observer.reset(_sample(0, lower_open=True, upper_open=True))
    observer.observe(_sample(1, red="upper", blue="lower", lower_open=True, upper_open=True))
    duplicate = observer.observe(_sample(1, red="upper", blue="lower", lower_open=True, upper_open=True))
    assert duplicate.duplicate_step
    assert state.stored_levels_by_item == {}
