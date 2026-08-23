from __future__ import annotations

import pytest

from bench.tasks.tm01.domain.contracts import (
    COMPLETION_BUTTON_NAMES,
    DRAWER_NAMES,
    ObserverConfig,
    StageEvent,
)
from bench.tasks.tm01.domain.semantics import (
    PhysicalState,
    SemanticState,
    diagnostics,
    evaluate_success,
    record_a1,
    record_a2,
    record_close_attempt,
    record_completion_button_attempt,
    record_inspection_attempt,
    record_open_attempt,
)
from bench.tasks.tm01.physical_events import PhysicalSample, TM01PhysicalObserver


def _complete_drawer_cycle(state: SemanticState, drawer: str) -> None:
    assert record_open_attempt(state, drawer, drawer_open=True).passed
    assert record_inspection_attempt(state, drawer, drawer_open=True).passed
    assert record_close_attempt(state, drawer, drawer_closed=True).passed


def _final_physical_state() -> PhysicalState:
    return PhysicalState(
        knob_qpos_rad=0.0,
        target_burner_on=False,
        all_other_burners_off=True,
        drawer_open_fraction={name: 0.0 for name in DRAWER_NAMES},
        pot_stable_on_burner=True,
    )


@pytest.mark.parametrize("turn_off_after_first", (True, False))
def test_tm01_attempt_semantics_preserve_both_legal_routes(
    turn_off_after_first: bool,
) -> None:
    first, second = DRAWER_NAMES
    state = SemanticState()

    assert record_a1(state).passed
    _complete_drawer_cycle(state, first)
    if turn_off_after_first:
        assert record_a2(state).passed
    _complete_drawer_cycle(state, second)
    if not turn_off_after_first:
        assert record_a2(state).passed
    assert record_completion_button_attempt(
        state,
        button_id=COMPLETION_BUTTON_NAMES[0],
        all_drawers_closed=True,
        button_physically_pressed=True,
    ).passed

    expected = (
        StageEvent.A1,
        StageEvent.B1,
        StageEvent.A2,
        StageEvent.B2,
        StageEvent.COMPLETION_BUTTON,
    )
    if not turn_off_after_first:
        expected = (
            StageEvent.A1,
            StageEvent.B1,
            StageEvent.B2,
            StageEvent.A2,
            StageEvent.COMPLETION_BUTTON,
        )
    assert tuple(state.event_history) == expected
    assert state.open_attempt_history == [first, second]
    assert state.inspect_history == [first, second]
    assert state.close_attempt_history == [first, second]
    assert evaluate_success(state, _final_physical_state()).success


def test_tm01_close_before_inspect_is_an_absorbing_failure() -> None:
    state = SemanticState()
    assert record_a1(state).passed
    assert record_open_attempt(state, DRAWER_NAMES[0], drawer_open=True).passed

    report = record_close_attempt(
        state,
        DRAWER_NAMES[0],
        drawer_closed=True,
    )

    assert not report.passed
    assert report.reason == "close_before_inspect"
    assert state.invalid_close_detected
    assert state.close_before_inspect_detected
    assert state.failure_reason == "close_before_inspect"
    assert state.absorbing_failure


def test_tm01_duplicate_drawer_open_is_attributed() -> None:
    first = DRAWER_NAMES[0]
    state = SemanticState()
    assert record_a1(state).passed
    _complete_drawer_cycle(state, first)

    report = record_open_attempt(state, first, drawer_open=True)

    assert not report.passed
    assert report.reason == "duplicate_drawer_check"
    assert state.invalid_open_detected
    assert state.duplicate_drawer_check
    assert state.open_attempt_history == [first, first]


def test_tm01_completion_attempt_before_both_threads_finish_fails() -> None:
    state = SemanticState()
    assert record_a1(state).passed
    _complete_drawer_cycle(state, DRAWER_NAMES[0])

    report = record_completion_button_attempt(
        state,
        button_id=COMPLETION_BUTTON_NAMES[0],
        all_drawers_closed=True,
        button_physically_pressed=True,
    )

    assert not report.passed
    assert report.reason == "completion_button_pressed_before_task_complete"
    assert state.completion_button_attempt_count == 1
    assert state.completion_button_press_count == 1
    assert state.invalid_completion_button_press
    assert diagnostics(state)["failure_reason"] == report.reason


def _observer_config() -> ObserverConfig:
    return ObserverConfig(
        drawer_open_threshold=0.70,
        drawer_closed_threshold=0.08,
        inspect_dwell_steps=2,
        button_pressed_threshold_m=0.004,
        button_released_threshold_m=0.001,
        button_release_dwell_steps=2,
        off_commit_max_rad=0.10,
        burner_on_min_rad=0.35,
        on_commit_min_rad=0.50,
        commit_dwell_steps=2,
        min_gripper_contact_steps=1,
        contact_evidence_window_steps=5,
        min_interaction_motion_rad=0.40,
        max_unexplained_delta_rad_per_step=0.25,
        pot_burner_xy_threshold_m=0.08,
        pot_translation_guard_m=0.02,
        pot_tilt_guard_deg=5.0,
    )


def _sample(
    step: int,
    *,
    left: float = 0.0,
    right: float = 0.0,
) -> PhysicalSample:
    return PhysicalSample(
        step_id=step,
        knob_qpos_rad=1.0,
        knob_qvel_rad_s=1.0,
        knob_gripper_contact=True,
        knob_any_robot_contact=True,
        target_burner_on=True,
        other_burners_on=False,
        drawer_open_fraction={
            DRAWER_NAMES[0]: left,
            DRAWER_NAMES[1]: right,
        },
        completion_button_press_depth_m={
            name: 0.0 for name in COMPLETION_BUTTON_NAMES
        },
    )


def _drawer_ready_observer() -> TM01PhysicalObserver:
    observer = TM01PhysicalObserver(_observer_config())
    initial = _sample(0)
    initial = PhysicalSample(
        **{
            **initial.__dict__,
            "knob_qpos_rad": 0.0,
            "target_burner_on": False,
        }
    )
    observer.reset(initial)
    assert record_a1(observer.state).passed
    return observer


def test_tm01_observer_commits_open_inspect_close_attempts() -> None:
    observer = _drawer_ready_observer()

    observer.observe(_sample(1, left=0.20))
    observer.observe(_sample(2, left=0.80))
    observer.observe(_sample(3, left=0.80))
    update = observer.observe(_sample(4, left=0.0))

    assert update.semantic_events == (StageEvent.B1.value,)
    assert observer.state.open_attempt_history == [DRAWER_NAMES[0]]
    assert observer.state.inspect_history == [DRAWER_NAMES[0]]
    assert observer.state.close_attempt_history == [DRAWER_NAMES[0]]
    assert observer.state.completed_drawer_history == [DRAWER_NAMES[0]]


def test_tm01_observer_rejects_an_opening_that_never_reaches_threshold() -> None:
    observer = _drawer_ready_observer()

    first = observer.observe(_sample(1, left=0.20))
    second = observer.observe(_sample(2, left=0.0))

    assert first.diagnostics["pending_drawer"] == DRAWER_NAMES[0]
    assert second.failure_labels == ("drawer_not_open",)
    assert observer.state.open_attempt_history == [DRAWER_NAMES[0]]
    assert observer.state.invalid_open_detected
    assert observer.state.failure_reason == "drawer_not_open"


def test_tm01_observer_does_not_commit_a2_while_drawer_is_pending() -> None:
    observer = _drawer_ready_observer()
    observer.state.b1_complete = True
    observer.state.boiling_ready = True
    observer.state.event_history.append(StageEvent.B1)

    observer.observe(_sample(1, right=0.20))
    update = observer.observe(
        PhysicalSample(
            **{
                **_sample(2, right=0.20).__dict__,
                "knob_qpos_rad": 0.0,
                "knob_qvel_rad_s": -1.0,
                "target_burner_on": False,
            }
        )
    )

    assert update.semantic_events == ()
    assert "A2_during_active_drawer" in observer.state.failure_labels


def test_tm01_observer_rejects_close_before_inspection_dwell() -> None:
    observer = _drawer_ready_observer()

    observer.observe(_sample(1, left=0.80))
    update = observer.observe(_sample(2, left=0.0))

    assert update.failure_labels == ("close_before_inspect",)
    assert observer.state.close_before_inspect_detected
    assert observer.state.close_attempt_history == [DRAWER_NAMES[0]]
