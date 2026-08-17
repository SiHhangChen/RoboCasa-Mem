from __future__ import annotations

from bench.tasks.wr04.domain.semantics import (
    SemanticState,
    evaluate_success,
    record_initial_level,
)
from bench.tasks.wr04.physical_events import PhysicalSample, WR04PhysicalObserver
from bench.runtime.contracts import RuntimeStep, TaskDiagnostics


def _state(level: str = "shelf_lower") -> SemanticState:
    state = SemanticState()
    assert record_initial_level(state, level, physically_verified=True)
    return state


def _sample(
    step: int,
    *,
    lower: bool = False,
    upper: bool = False,
    sink: bool = False,
    lower_supported: bool = False,
    upper_supported: bool = False,
) -> PhysicalSample:
    return PhysicalSample(
        step_id=step,
        item_at_fixed_sink=sink,
        item_in_lower=lower,
        item_in_upper=upper,
        item_supported_by_lower=lower_supported,
        item_supported_by_upper=upper_supported,
    )


def _observe_dwell(
    observer: WR04PhysicalObserver,
    start: int,
    **sample: bool,
) -> int:
    for step in range(start, start + observer.dwell_steps):
        observer.observe(_sample(step, **sample))
    return start + observer.dwell_steps


def _observe_terminal_verification(
    observer: WR04PhysicalObserver,
    start: int,
    **sample: bool,
) -> int:
    for step in range(start, start + observer.terminal_verification_steps):
        observer.observe(_sample(step, **sample))
    return start + observer.terminal_verification_steps


def test_wr04_policy_neutral_observer_latches_complete_sequence() -> None:
    state = _state()
    observer = WR04PhysicalObserver(
        state, dwell_steps=2, terminal_verification_steps=2
    )
    observer.reset(_sample(0, lower=True, lower_supported=True))

    step = _observe_dwell(observer, 1)
    assert state.item_removed_from_home

    step = _observe_dwell(observer, step, sink=True)
    assert state.item_staged_at_sink
    assert state.memory_barrier_latched

    step = _observe_dwell(observer, step)
    step = _observe_dwell(observer, step, lower=True, lower_supported=True)
    assert state.first_return_level == "shelf_lower"
    assert state.returned_item_id is None
    _observe_terminal_verification(
        observer, step, lower=True, lower_supported=True
    )
    assert not state.wrong_return_level_attempted
    assert state.returned_item_id == "ketchup"
    assert set(observer.stage_first_steps) == {
        "pickup",
        "staged",
        "barrier",
        "return_correct",
        "returned",
        "success",
    }


def test_wr04_terminal_verification_rejects_restarted_operation() -> None:
    state = _state()
    observer = WR04PhysicalObserver(
        state,
        dwell_steps=2,
        terminal_verification_steps=5,
    )
    observer.reset(_sample(0, lower=True, lower_supported=True))
    step = _observe_dwell(observer, 1)
    step = _observe_dwell(observer, step, sink=True)
    step = _observe_dwell(observer, step)
    step = _observe_dwell(observer, step, lower=True, lower_supported=True)

    observer.observe(_sample(step, lower=True, lower_supported=True))
    observer.observe(_sample(step + 1, lower=True, lower_supported=True))
    assert state.returned_item_id is None

    observer.observe(_sample(step + 2))
    report = observer.observe(_sample(step + 3)).diagnostics
    assert report["phase"] == "invalid"
    assert report["violation_reason"] == (
        "item_disturbed_during_terminal_verification"
    )
    assert not evaluate_success(
        state,
        _sample(step + 3).final_physical_state(),
    ).success


def test_wr04_initial_and_transient_states_do_not_create_events() -> None:
    state = _state()
    observer = WR04PhysicalObserver(
        state, dwell_steps=3, terminal_verification_steps=2
    )
    observer.reset(_sample(0, lower=True, lower_supported=True))

    observer.observe(_sample(1))
    observer.observe(_sample(2, lower=True, lower_supported=True))
    observer.observe(_sample(3))

    assert not state.item_removed_from_home
    assert observer.stage_first_steps == {}


def test_wr04_sink_stage_latches_environment_memory_barrier() -> None:
    state = _state()
    observer = WR04PhysicalObserver(
        state, dwell_steps=2, terminal_verification_steps=2
    )
    observer.reset(_sample(0, lower=True, lower_supported=True))

    step = _observe_dwell(observer, 1)
    _observe_dwell(observer, step, sink=True)

    assert state.item_staged_at_sink
    assert state.item_removed_from_home
    assert state.memory_barrier_latched
    assert observer.stage_first_steps == {
        "pickup": 2,
        "staged": 4,
        "barrier": 4,
    }


def test_wr04_first_wrong_return_level_is_latched() -> None:
    state = _state()
    observer = WR04PhysicalObserver(
        state, dwell_steps=1, terminal_verification_steps=2
    )
    observer.reset(_sample(0, lower=True, lower_supported=True))
    observer.observe(_sample(1))
    observer.observe(_sample(2, sink=True))
    observer.observe(_sample(3))

    observer.observe(_sample(4, upper=True, upper_supported=True))
    observer.observe(_sample(5, lower=True, lower_supported=True))

    assert state.first_return_level == "shelf_upper"
    assert state.wrong_return_level_attempted
    assert state.returned_item_id is None
    assert "return_correct" not in observer.stage_first_steps
    assert "success" not in observer.stage_first_steps


def test_wr04_duplicate_step_is_not_counted_toward_dwell() -> None:
    state = _state()
    observer = WR04PhysicalObserver(
        state, dwell_steps=2, terminal_verification_steps=2
    )
    observer.reset(_sample(0, lower=True, lower_supported=True))

    observer.observe(_sample(1))
    duplicate = observer.observe(_sample(1))
    assert duplicate.duplicate_step
    assert not state.item_removed_from_home

    observer.observe(_sample(2))
    assert state.item_removed_from_home


def _complete_return(state: SemanticState, *, original: str = "shelf_lower", dwell: int = 2) -> WR04PhysicalObserver:
    observer = WR04PhysicalObserver(
        state, dwell_steps=dwell, terminal_verification_steps=2
    )
    observer.reset(_sample(0, lower=original == "shelf_lower", lower_supported=original == "shelf_lower"))
    step = _observe_dwell(observer, 1)
    step = _observe_dwell(observer, step, sink=True)
    step = _observe_dwell(observer, step)
    step = _observe_dwell(observer, step, lower=True, lower_supported=True)
    _observe_terminal_verification(
        observer, step, lower=True, lower_supported=True
    )
    return observer


def test_wr04_removing_returned_item_is_a_stage_regression() -> None:
    state = _state()
    observer = _complete_return(state, dwell=2)
    assert state.returned_item_id == "ketchup"
    assert not state.stage_regressions

    # After a successful return, taking the item back off the shelf must not
    # count as a second successful episode: the returned stage is only valid
    # once.
    start = observer.stage_first_steps["returned"] + 1
    for step in range(start, start + observer.dwell_steps):
        observer.observe(_sample(step))
    assert "item_removed_after_success" in state.stage_regressions
    assert state.protocol_violation


def test_wr04_returning_to_sink_after_departure_is_a_stage_regression() -> None:
    state = _state()
    observer = WR04PhysicalObserver(
        state, dwell_steps=1, terminal_verification_steps=2
    )
    observer.reset(_sample(0, lower=True, lower_supported=True))
    observer.observe(_sample(1))
    observer.observe(_sample(2, sink=True))
    observer.observe(_sample(3))
    assert not state.stage_regressions

    # Once the sponge leaves the staged sink, entering the sink again repeats
    # a completed stage and invalidates the episode.
    observer.observe(_sample(4, sink=True))
    assert "sink_reentered_after_departure" in state.stage_regressions
    assert state.protocol_violation


def test_wr04_phase_history_is_strict_and_single_pass() -> None:
    state = _state()
    observer = _complete_return(state, dwell=1)
    report = observer.observe(
        _sample(6, lower=True, lower_supported=True)
    ).diagnostics
    assert [entry["phase"] for entry in report["phase_history"]] == [
        "home",
        "to_sink",
        "at_sink",
        "to_return",
        "verify_return",
        "complete",
    ]


def test_runtime_step_honors_task_semantic_terminal() -> None:
    transition = RuntimeStep(
        observation={},
        diagnostics=TaskDiagnostics(
            success=False,
            terminal=True,
            failure_code="sink_reentered_after_departure",
        ),
    )
    assert transition.terminal
