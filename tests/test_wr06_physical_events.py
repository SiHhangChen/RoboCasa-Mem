"""Focused WR-06 physical-event protocol tests (mirror of the WR-04 suite)."""

from __future__ import annotations

from bench.tasks.wr06.domain.semantics import (
    SemanticState,
    evaluate_success,
    record_initial_level,
)
from bench.tasks.wr06.physical_events import PhysicalSample, WR06PhysicalObserver
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
    plate: bool = False,
    lower_supported: bool = False,
    upper_supported: bool = False,
    grasped: bool = False,
) -> PhysicalSample:
    return PhysicalSample(
        step_id=step,
        item_at_fixed_plate=plate,
        item_in_lower=lower,
        item_in_upper=upper,
        item_supported_by_lower=lower_supported,
        item_supported_by_upper=upper_supported,
        item_grasped=grasped,
    )


def _observe_dwell(
    observer: WR06PhysicalObserver,
    start: int,
    **sample: bool,
) -> int:
    for step in range(start, start + observer.dwell_steps):
        observer.observe(_sample(step, **sample))
    return start + observer.dwell_steps


def _observe_terminal_verification(
    observer: WR06PhysicalObserver,
    start: int,
    **sample: bool,
) -> int:
    for step in range(start, start + observer.terminal_verification_steps):
        observer.observe(_sample(step, **sample))
    return start + observer.terminal_verification_steps


def _arm_return_verification(observer: WR06PhysicalObserver) -> None:
    assert observer.begin_return_placement()
    assert observer.arm_return_verification()


def test_wr06_policy_neutral_observer_latches_complete_sequence() -> None:
    state = _state()
    observer = WR06PhysicalObserver(
        state, dwell_steps=2, terminal_verification_steps=2
    )
    observer.reset(_sample(0, lower=True, lower_supported=True))

    step = _observe_dwell(observer, 1)
    assert state.item_removed_from_home

    step = _observe_dwell(observer, step, plate=True)
    assert state.item_staged_at_plate
    assert state.memory_barrier_latched

    step = _observe_dwell(observer, step, grasped=True)
    _arm_return_verification(observer)
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


def test_wr06_terminal_verification_rejects_restarted_operation() -> None:
    state = _state()
    observer = WR06PhysicalObserver(
        state,
        dwell_steps=2,
        terminal_verification_steps=5,
    )
    observer.reset(_sample(0, lower=True, lower_supported=True))
    step = _observe_dwell(observer, 1)
    step = _observe_dwell(observer, step, plate=True)
    step = _observe_dwell(observer, step, grasped=True)
    _arm_return_verification(observer)
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


def test_wr06_initial_and_transient_states_do_not_create_events() -> None:
    state = _state()
    observer = WR06PhysicalObserver(
        state, dwell_steps=3, terminal_verification_steps=2
    )
    observer.reset(_sample(0, lower=True, lower_supported=True))

    observer.observe(_sample(1))
    observer.observe(_sample(2, lower=True, lower_supported=True))
    observer.observe(_sample(3))

    assert not state.item_removed_from_home
    assert observer.stage_first_steps == {}


def test_wr06_plate_stage_latches_environment_memory_barrier() -> None:
    state = _state()
    observer = WR06PhysicalObserver(
        state, dwell_steps=2, terminal_verification_steps=2
    )
    observer.reset(_sample(0, lower=True, lower_supported=True))

    step = _observe_dwell(observer, 1)
    _observe_dwell(observer, step, plate=True)

    assert state.item_staged_at_plate
    assert state.item_removed_from_home
    assert state.memory_barrier_latched
    assert observer.stage_first_steps == {
        "pickup": 2,
        "staged": 4,
        "barrier": 4,
    }


def test_wr06_held_bottle_at_plate_does_not_complete_staging() -> None:
    state = _state()
    observer = WR06PhysicalObserver(
        state, dwell_steps=2, terminal_verification_steps=2
    )
    observer.reset(_sample(0, lower=True, lower_supported=True))

    step = _observe_dwell(observer, 1)
    step = _observe_dwell(observer, step, plate=True, grasped=True)
    assert not state.item_staged_at_plate
    assert not state.memory_barrier_latched

    observer.observe(_sample(step, plate=True, grasped=False))
    assert not state.item_staged_at_plate
    observer.observe(_sample(step + 1, plate=True, grasped=False))
    assert state.item_staged_at_plate
    assert state.memory_barrier_latched


def test_wr06_first_wrong_return_level_is_latched() -> None:
    state = _state()
    observer = WR06PhysicalObserver(
        state, dwell_steps=1, terminal_verification_steps=2
    )
    observer.reset(_sample(0, lower=True, lower_supported=True))
    observer.observe(_sample(1))
    observer.observe(_sample(2, plate=True))
    observer.observe(_sample(3, grasped=True))
    _arm_return_verification(observer)

    observer.observe(_sample(4, upper=True, upper_supported=True))
    observer.observe(_sample(5, lower=True, lower_supported=True))

    assert state.first_return_level == "shelf_upper"
    assert state.wrong_return_level_attempted
    assert state.returned_item_id is None
    assert "return_correct" not in observer.stage_first_steps
    assert "success" not in observer.stage_first_steps


def test_wr06_duplicate_step_is_not_counted_toward_dwell() -> None:
    state = _state()
    observer = WR06PhysicalObserver(
        state, dwell_steps=2, terminal_verification_steps=2
    )
    observer.reset(_sample(0, lower=True, lower_supported=True))

    observer.observe(_sample(1))
    duplicate = observer.observe(_sample(1))
    assert duplicate.duplicate_step
    assert not state.item_removed_from_home

    observer.observe(_sample(2))
    assert state.item_removed_from_home


def _complete_return(state: SemanticState, *, original: str = "shelf_lower", dwell: int = 2) -> WR06PhysicalObserver:
    observer = WR06PhysicalObserver(
        state, dwell_steps=dwell, terminal_verification_steps=2
    )
    observer.reset(_sample(0, lower=original == "shelf_lower", lower_supported=original == "shelf_lower"))
    step = _observe_dwell(observer, 1)
    step = _observe_dwell(observer, step, plate=True)
    step = _observe_dwell(observer, step, grasped=True)
    _arm_return_verification(observer)
    step = _observe_dwell(observer, step, lower=True, lower_supported=True)
    _observe_terminal_verification(
        observer, step, lower=True, lower_supported=True
    )
    return observer


def test_wr06_removing_returned_item_is_a_stage_regression() -> None:
    state = _state()
    observer = _complete_return(state, dwell=2)
    assert state.returned_item_id == "ketchup"
    assert not state.stage_regressions

    start = observer.stage_first_steps["returned"] + 1
    for step in range(start, start + observer.dwell_steps):
        observer.observe(_sample(step))
    assert "item_removed_after_success" in state.stage_regressions
    assert state.protocol_violation


def test_wr06_returning_to_plate_after_departure_is_a_stage_regression() -> None:
    state = _state()
    observer = WR06PhysicalObserver(
        state, dwell_steps=1, terminal_verification_steps=2
    )
    observer.reset(_sample(0, lower=True, lower_supported=True))
    observer.observe(_sample(1))
    observer.observe(_sample(2, plate=True))
    observer.observe(_sample(3, grasped=True))
    assert not state.stage_regressions

    observer.observe(_sample(4, plate=True, grasped=True))
    assert "plate_reentered_after_departure" in state.stage_regressions
    assert state.protocol_violation


def test_wr06_bumped_plate_bottle_does_not_count_as_reentry() -> None:
    """An open-gripper bump must not create a fake return violation."""
    state = _state()
    observer = WR06PhysicalObserver(
        state, dwell_steps=3, terminal_verification_steps=2
    )
    observer.reset(_sample(0, lower=True, lower_supported=True))
    step = _observe_dwell(observer, 1)
    step = _observe_dwell(observer, step, plate=True, grasped=True)

    # The original shelf pick is still holding the bottle as it first reaches
    # the plate. It must not count as the later plate-side re-pick.
    step = _observe_dwell(observer, step, plate=True, grasped=True)
    step = _observe_dwell(observer, step, plate=True, grasped=False)

    # The bottle leaves the staging radius briefly, without a confirmed
    # grasp, and then falls back into that radius as in seed 135.
    for current in range(step, step + 3):
        observer.observe(_sample(current, grasped=False))
    for current in range(step + 3, step + 6):
        observer.observe(_sample(current, plate=True, grasped=False))

    assert observer._phase == observer.AT_PLATE
    assert not state.protocol_violation
    assert "plate_reentered_after_departure" not in state.stage_regressions


def test_wr06_return_verification_arms_from_physical_supported_release() -> None:
    state = _state()
    observer = WR06PhysicalObserver(
        state, dwell_steps=2, terminal_verification_steps=2
    )
    observer.reset(_sample(0, lower=True, lower_supported=True))
    step = _observe_dwell(observer, 1)
    step = _observe_dwell(observer, step, plate=True)
    step = _observe_dwell(observer, step, grasped=True)

    step = _observe_dwell(
        observer,
        step,
        lower=True,
        lower_supported=True,
        grasped=True,
    )
    assert state.first_return_level is None
    assert observer._phase == observer.TO_RETURN

    step = _observe_dwell(observer, step, lower=True, lower_supported=True)
    assert state.first_return_level == "shelf_lower"
    assert observer._return_verification_armed
    _observe_terminal_verification(
        observer, step, lower=True, lower_supported=True
    )
    assert state.returned_item_id == "ketchup"


def test_wr06_return_handler_defers_completion_to_physical_observer(
    monkeypatch,
) -> None:
    from bench.tasks.wr06.execution import handlers

    class FakeEnvironment:
        demo_original_level = "shelf_lower"

        def return_region_pose(self, level: str) -> list[float]:
            assert level == "shelf_lower"
            return [0.0, 0.0, 0.0]

    class FakeController:
        def __init__(self, env: FakeEnvironment) -> None:
            self.env = env

    captured: dict[str, object] = {}

    def fake_place(controller, **kwargs):
        captured.update(kwargs)
        return {"pass": True}

    monkeypatch.setattr(handlers, "place_at_accessible", fake_place)
    env = FakeEnvironment()
    operation = handlers.build_operation_handlers(FakeController(env))["return_item_to_level"]
    result = operation("ketchup", return_level="shelf_lower")

    assert result["pass"] is True
    assert result["return_verification_mode"] == "physical_supported_release"
    assert result["return_completion_pending"] is True
    assert captured["target_pos_world"] == [0.0, 0.0, 0.0]


def test_wr06_phase_history_is_strict_and_single_pass() -> None:
    state = _state()
    observer = _complete_return(state, dwell=1)
    report = observer.observe(
        _sample(6, lower=True, lower_supported=True)
    ).diagnostics
    assert [entry["phase"] for entry in report["phase_history"]] == [
        "home",
        "to_plate",
        "at_plate",
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
            failure_code="plate_reentered_after_departure",
        ),
    )
    assert transition.terminal
