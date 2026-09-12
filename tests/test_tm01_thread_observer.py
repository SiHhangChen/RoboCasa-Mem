"""Replay physical samples through the unchanged observer and the v2 evaluator."""

from dataclasses import replace

import pytest

from bench.tasks.tm01.config import load_task_config, resolve_observer_config
from bench.tasks.tm01.domain.contracts import COMPLETION_BUTTON_NAMES, DRAWER_NAMES
from bench.tasks.tm01.domain.semantics import PhysicalState, evaluate_success
from bench.tasks.tm01.evaluation import TM01EvaluationSnapshot, TM01Evaluator
from bench.tasks.tm01.physical_events import PhysicalSample, TM01PhysicalObserver


@pytest.mark.parametrize("a2_first", (True, False))
@pytest.mark.parametrize("first_drawer", DRAWER_NAMES)
def test_tm01_physical_sample_routes_preserve_independent_thread_progress(
    a2_first: bool, first_drawer: str,
) -> None:
    config = resolve_observer_config(load_task_config())
    observer = TM01PhysicalObserver(config)
    sample = PhysicalSample(
        step_id=0, knob_qpos_rad=0.0, knob_qvel_rad_s=0.0,
        knob_gripper_contact=False, knob_any_robot_contact=False,
        target_burner_on=False, other_burners_on=False,
        drawer_open_fraction=dict.fromkeys(DRAWER_NAMES, 0.0),
        completion_button_press_depth_m=dict.fromkeys(COMPLETION_BUTTON_NAMES, 0.0),
    )
    observer.reset(sample)
    evaluator = TM01Evaluator()
    evaluator.reset(TM01EvaluationSnapshot(step_id=0))

    def tick(**changes):
        nonlocal sample
        sample = replace(sample, step_id=sample.step_id + 1, **changes)
        observer.observe(sample)
        assert not observer.state.absorbing_failure, observer.state.failure_labels
        assert not observer.invalid_environment_labels
        state = observer.state
        physical = PhysicalState(
            knob_qpos_rad=sample.knob_qpos_rad,
            target_burner_on=sample.target_burner_on,
            all_other_burners_off=not sample.other_burners_on,
            drawer_open_fraction=sample.drawer_open_fraction,
            pot_stable_on_burner=sample.pot_stable_on_burner,
        )
        return evaluator.observe(TM01EvaluationSnapshot(
            step_id=sample.step_id, burner_on_completed=state.a1_complete,
            one_drawer_inspected=state.b1_complete,
            boiling_thread_completed=state.a2_complete,
            drawer_thread_completed=state.b2_complete,
            completion_button_pressed=state.completion_button_pressed,
            domain_success=evaluate_success(state, physical).success,
            event_steps=observer.event_steps,
        ))

    for angle in (0.2, 0.4, 0.6, 0.8, 1.0):
        tick(knob_qpos_rad=angle, knob_qvel_rad_s=1.0,
             knob_gripper_contact=True, knob_any_robot_contact=True,
             target_burner_on=angle >= config.burner_on_min_rad)
    assert observer.state.a1_complete

    def check_drawer(drawer: str):
        fractions = {name: 0.8 if name == drawer else 0.0 for name in DRAWER_NAMES}
        for _ in range(config.inspect_dwell_steps):
            tick(drawer_open_fraction=fractions, knob_gripper_contact=False,
                 knob_any_robot_contact=False, knob_qvel_rad_s=0.0)
        return tick(drawer_open_fraction=dict.fromkeys(DRAWER_NAMES, 0.0))

    def turn_off():
        for angle in (0.8, 0.6, 0.4, 0.2, 0.0, 0.0, 0.0):
            result = tick(knob_qpos_rad=angle, knob_qvel_rad_s=-1.0,
                          knob_gripper_contact=True, knob_any_robot_contact=True,
                          target_burner_on=angle >= config.burner_on_min_rad)
        return result

    first = check_drawer(first_drawer)
    assert first.stages["one_drawer_inspected"].qualified
    assert not first.stages["one_thread_completed"].qualified
    other_drawer = next(name for name in DRAWER_NAMES if name != first_drawer)
    partial = turn_off() if a2_first else check_drawer(other_drawer)
    assert partial.stages["one_thread_completed"].qualified
    assert not partial.stages["both_threads_completed"].qualified
    complete = check_drawer(other_drawer) if a2_first else turn_off()
    assert complete.stages["both_threads_completed"].qualified
    assert not complete.success
    result = tick(completion_button_press_depth_m={
        COMPLETION_BUTTON_NAMES[0]: config.button_pressed_threshold_m,
        COMPLETION_BUTTON_NAMES[1]: 0.0,
    })
    assert result.success
    expected_tail = ["A2", "B2"] if a2_first else ["B2", "A2"]
    assert [event.value for event in observer.state.event_history] == [
        "A1", "B1", *expected_tail, "completion_button",
    ]
    assert observer.state.completed_drawer_history == [first_drawer, other_drawer]
    steps = observer.event_steps
    assert result.stages["one_thread_completed"].qualified_step == min(steps["A2"], steps["B2"])
    assert result.stages["both_threads_completed"].qualified_step == max(steps["A2"], steps["B2"])
