from __future__ import annotations

import pytest

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
    assert not state.canonical_reset_complete
    report = evaluate_success(state, PhysicalState(distractor_still_stored=True))
    assert report.success
    assert "canonical_reset_complete" not in report.checks()


@pytest.mark.parametrize(
    ("red", "blue", "lower_closed", "upper_closed"),
    [
        ("upper", "lower", True, False),
        ("upper", "lower", False, True),
        (None, "lower", True, True),
        ("upper", None, True, True),
    ],
)
def test_wr01_memory_barrier_requires_both_items_and_closed_doors(
    red: str | None,
    blue: str | None,
    lower_closed: bool,
    upper_closed: bool,
) -> None:
    state = SemanticState(
        "blue_can", "lower",
        stored_levels_by_item={"red_can": "upper", "blue_can": "lower"},
    )
    observer = WR01PhysicalObserver(state, dwell_steps=1)
    observer.reset(_sample(0, red="upper", blue="lower", lower_open=True, upper_open=True))
    _repeat(
        observer, 1, 3, red=red, blue=blue,
        lower_closed=lower_closed, upper_closed=upper_closed,
    )

    assert not state.storage_hidden
    assert observer.phase == observer.CLOSE


def test_wr01_observer_can_open_the_other_cabinet_then_retrieve_the_target() -> None:
    state = SemanticState("blue_can", "lower")
    observer = WR01PhysicalObserver(state, dwell_steps=1)
    observer.reset(_sample(0, lower_open=True, upper_open=True))
    observer.observe(_sample(1, red="upper", blue="lower", lower_open=True, upper_open=True))
    observer.observe(_sample(2, red="upper", blue="lower", lower_closed=True, upper_closed=True))
    observer.observe(_sample(3, red="upper", blue="lower", lower_closed=True, upper_closed=True))
    observer.observe(_sample(4, red="upper", blue="lower", upper_open=True, lower_closed=True))
    assert state.storage_hidden
    assert state.retrieval_level_opened is None
    assert state.wrong_retrieval_level_opened
    assert observer.phase == observer.HIDDEN

    observer.observe(_sample(5, red="upper", blue="lower", upper_open=True, lower_open=True))
    assert state.retrieval_level_opened == "lower"
    assert observer.phase == observer.RETRIEVE
    observer.observe(_sample(6, red="upper", upper_open=True, lower_open=True))
    assert state.retrieved_item_id == "blue_can"
    report = evaluate_success(state, PhysicalState(distractor_still_stored=True))
    assert report.success
    assert "no_wrong_retrieval_door" not in report.checks()


@pytest.mark.parametrize("target_level", ["lower", "upper"])
@pytest.mark.parametrize("both_open", [False, True])
def test_wr01_observer_accepts_the_target_door_with_or_without_the_other_door_open(
    target_level: str, both_open: bool,
) -> None:
    other = "upper" if target_level == "lower" else "lower"
    state = SemanticState(
        "blue_can", target_level,
        stored_levels_by_item={"blue_can": target_level, "red_can": other},
        storage_hidden=True,
    )
    observer = WR01PhysicalObserver(state, dwell_steps=2)
    observer.reset(_sample(0, red=other, blue=target_level, lower_closed=True, upper_closed=True))

    _repeat(
        observer, 1, 2, red=other, blue=target_level,
        lower_open=both_open or target_level == "lower",
        upper_open=both_open or target_level == "upper",
        lower_closed=not both_open and target_level != "lower",
        upper_closed=not both_open and target_level != "upper",
    )

    assert state.retrieval_level_opened == target_level
    assert observer.phase == observer.RETRIEVE


def test_wr01_target_door_open_dwell_resets_when_the_door_closes() -> None:
    state = SemanticState(
        "blue_can", "lower",
        stored_levels_by_item={"blue_can": "lower", "red_can": "upper"},
        storage_hidden=True,
    )
    observer = WR01PhysicalObserver(state, dwell_steps=2)
    observer.reset(_sample(0, red="upper", blue="lower", lower_closed=True, upper_closed=True))
    observer.observe(_sample(1, red="upper", blue="lower", lower_open=True, upper_closed=True))
    observer.observe(_sample(2, red="upper", blue="lower", lower_closed=True, upper_closed=True))
    observer.observe(_sample(3, red="upper", blue="lower", lower_open=True, upper_closed=True))
    assert state.retrieval_level_opened is None
    observer.observe(_sample(4, red="upper", blue="lower", lower_open=True, upper_closed=True))
    assert state.retrieval_level_opened == "lower"


def test_wr01_opening_only_the_other_cabinet_does_not_qualify_retrieval() -> None:
    state = SemanticState(
        "blue_can", "lower",
        stored_levels_by_item={"blue_can": "lower", "red_can": "upper"},
        storage_hidden=True,
    )
    observer = WR01PhysicalObserver(state, dwell_steps=1)
    observer.reset(_sample(0, red="upper", blue="lower", lower_closed=True, upper_closed=True))
    _repeat(observer, 1, 3, red="upper", upper_open=True, lower_closed=True)
    assert state.retrieval_level_opened is None
    assert state.retrieved_item_id is None
    assert not evaluate_success(state, PhysicalState(distractor_still_stored=True)).success


def test_wr01_observer_deduplicates_simulator_step() -> None:
    state = SemanticState("blue_can", "lower")
    observer = WR01PhysicalObserver(state, dwell_steps=2)
    observer.reset(_sample(0, lower_open=True, upper_open=True))
    observer.observe(_sample(1, red="upper", blue="lower", lower_open=True, upper_open=True))
    duplicate = observer.observe(_sample(1, red="upper", blue="lower", lower_open=True, upper_open=True))
    assert duplicate.duplicate_step
    assert state.stored_levels_by_item == {}
