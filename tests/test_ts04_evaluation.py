"""Task-level evaluation and physical-observer tests for TS04."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import bench.tasks.ts04.runtime as ts04_runtime
from bench.tasks.ts04.domain.contracts import FIXTURE_NAMES
from bench.tasks.ts04.evaluation import (
    REPORT_STAGE_ORDER,
    TS04EvaluationSnapshot,
    TS04Evaluator,
    diagnostics_for_runtime,
    report_stages_for_runtime,
    success_for_runtime,
)
from bench.tasks.ts04.physical_events import PhysicalSample, TS04PhysicalObserver


def _snapshot(
    step: int,
    *,
    target: str,
    history: tuple[str, ...] = (),
    search_after_target: bool = False,
    button_steps: tuple[int, ...] = (),
    actual_button_count: int | None = None,
    success: bool = False,
    failure_reason: str | None = None,
) -> TS04EvaluationSnapshot:
    target_found = target in history
    return TS04EvaluationSnapshot(
        step_id=step,
        fixture_names=FIXTURE_NAMES,
        target_fixture=target,
        completed_fixture_history=history,
        completed_fixture_steps={
            fixture: index
            for index, fixture in enumerate(history, start=1)
        },
        target_found=target_found,
        search_after_target_detected=search_after_target,
        valid_button_press_steps=button_steps,
        actual_button_press_count=(
            len(button_steps)
            if actual_button_count is None
            else actual_button_count
        ),
        required_button_press_count=len(history) if target_found else 0,
        domain_success=success,
        failure_reason=failure_reason,
    )


def _evaluator(target: str) -> TS04Evaluator:
    evaluator = TS04Evaluator(
        fixture_names=FIXTURE_NAMES,
        target_fixture=target,
    )
    evaluator.reset(_snapshot(0, target=target))
    return evaluator


@pytest.mark.parametrize("search_length", (1, 2))
def test_five_stage_funnel_requires_one_press_per_checked_drawer(
    search_length: int,
) -> None:
    target = FIXTURE_NAMES[search_length - 1]
    evaluator = _evaluator(target)
    history: tuple[str, ...] = ()

    for close_step, fixture in enumerate(
        FIXTURE_NAMES[:search_length],
        start=1,
    ):
        history = (*history, fixture)
        result = evaluator.observe(
            _snapshot(close_step, target=target, history=history)
        )

    assert result.stages["one_drawer_checked"].qualified
    assert result.stages["one_drawer_checked"].qualified_step == 1
    assert result.stages["target_found"].qualified
    assert result.stages["target_found"].qualified_step == search_length
    assert result.stages["search_stopped_after_target"].qualified
    assert (
        result.stages["search_stopped_after_target"].qualified_step
        == search_length
    )
    assert not result.stages["button_pressed"].qualified
    assert not result.success

    button_steps = tuple(range(search_length + 1, 2 * search_length + 1))
    result = evaluator.observe(
        _snapshot(
            2 * search_length,
            target=target,
            history=history,
            button_steps=button_steps,
            success=True,
        )
    )

    assert result.stages["button_pressed"].qualified
    assert result.stages["button_pressed"].qualified_step == search_length + 1
    assert result.stages["correct_press_count"].qualified
    assert result.stages["correct_press_count"].qualified_step == 2 * search_length
    assert result.success
    assert result.details["required_button_presses"] == search_length


def test_second_button_press_revokes_final_verdict() -> None:
    target = FIXTURE_NAMES[0]
    evaluator = _evaluator(target)
    evaluator.observe(_snapshot(1, target=target, history=(target,)))
    evaluator.observe(
        _snapshot(
            2,
            target=target,
            history=(target,),
            button_steps=(2,),
            success=True,
        )
    )

    extra = evaluator.observe(
        _snapshot(
            3,
            target=target,
            history=(target,),
            button_steps=(2,),
            actual_button_count=2,
            failure_reason="completion_button_pressed_too_many_times",
        )
    )

    assert extra.stages["button_pressed"].qualified
    assert not extra.stages["correct_press_count"].qualified
    assert extra.stages["correct_press_count"].observed
    assert extra.stages["correct_press_count"].invalidated
    assert (
        extra.stages["correct_press_count"].reason
        == "completion_button_count_does_not_match_search_count"
    )
    assert not extra.success


def test_search_after_target_revokes_stop_and_downstream_stages() -> None:
    target = FIXTURE_NAMES[0]
    evaluator = _evaluator(target)
    evaluator.observe(_snapshot(1, target=target, history=(target,)))
    evaluator.observe(
        _snapshot(
            2,
            target=target,
            history=(target,),
            button_steps=(2,),
            success=True,
        )
    )

    continued = evaluator.observe(
        _snapshot(
            3,
            target=target,
            history=(target,),
            search_after_target=True,
            button_steps=(2,),
            failure_reason="search_continued_after_target_found",
        )
    )

    assert continued.stages["target_found"].qualified
    for name in (
        "search_stopped_after_target",
        "button_pressed",
        "correct_press_count",
    ):
        assert not continued.stages[name].qualified
        assert continued.stages[name].invalidated
        assert (
            continued.to_report_stages()["stages"][name]["first_pass_step"]
            is None
        )


def test_public_report_contains_exactly_five_stages() -> None:
    target = FIXTURE_NAMES[0]
    evaluator = _evaluator(target)
    evaluator.observe(_snapshot(1, target=target, history=(target,)))
    result = evaluator.observe(
        _snapshot(
            2,
            target=target,
            history=(target,),
            button_steps=(2,),
            success=True,
        )
    )

    report = result.to_report_stages()

    assert report["profile_id"] == "TS04/report-v1"
    assert report["order"] == list(REPORT_STAGE_ORDER)
    assert tuple(report["stages"]) == REPORT_STAGE_ORDER
    assert "success" not in report["stages"]
    assert report["stages"]["correct_press_count"] == {
        "applicable": True,
        "passed": True,
        "first_pass_step": 2,
    }


def test_snapshot_enforces_search_length_press_requirement() -> None:
    target = FIXTURE_NAMES[0]

    with pytest.raises(ValueError, match="required_button_press_count"):
        TS04EvaluationSnapshot(
            step_id=1,
            fixture_names=FIXTURE_NAMES,
            target_fixture=target,
            completed_fixture_history=(target,),
            completed_fixture_steps={target: 1},
            target_found=True,
            required_button_press_count=2,
        )

    second = FIXTURE_NAMES[1]
    snapshot = TS04EvaluationSnapshot(
        step_id=2,
        fixture_names=FIXTURE_NAMES,
        target_fixture=second,
        completed_fixture_history=FIXTURE_NAMES,
        completed_fixture_steps={FIXTURE_NAMES[0]: 1, second: 2},
        target_found=True,
        required_button_press_count=2,
    )
    assert snapshot.required_button_press_count == 2


def test_same_step_is_idempotent_and_backward_step_is_rejected() -> None:
    target = FIXTURE_NAMES[0]
    evaluator = TS04Evaluator(
        fixture_names=FIXTURE_NAMES,
        target_fixture=target,
    )
    evaluator.reset(_snapshot(1, target=target))
    initial = evaluator.result()

    assert evaluator.observe(
        _snapshot(1, target=target, history=(target,))
    ) is initial
    with pytest.raises(ValueError, match="moved backwards"):
        evaluator.observe(_snapshot(0, target=target))


def _observer(target: str) -> TS04PhysicalObserver:
    return TS04PhysicalObserver(
        fixture_names=FIXTURE_NAMES,
        target_fixture=target,
        drawer_open_threshold=0.70,
        drawer_closed_threshold=0.08,
        button_pressed_threshold_m=0.004,
        button_released_threshold_m=0.001,
        button_release_dwell_steps=2,
    )


def _physical_sample(
    step: int,
    *,
    left: float = 0.0,
    right: float = 0.0,
    button: float = 0.0,
) -> PhysicalSample:
    return PhysicalSample(
        step_id=step,
        drawer_open_fraction={
            FIXTURE_NAMES[0]: left,
            FIXTURE_NAMES[1]: right,
        },
        completion_button_press_depth_m=button,
    )


def test_physical_observer_completes_free_policy_trajectory() -> None:
    target = FIXTURE_NAMES[1]
    observer = _observer(target)
    observer.reset(_physical_sample(0))

    observer.observe(_physical_sample(1, left=0.75))
    observer.observe(_physical_sample(2))
    observer.observe(_physical_sample(3, right=0.75))
    observer.observe(_physical_sample(4))

    assert observer.state.completed_fixture_history == list(FIXTURE_NAMES)
    assert observer.state.target_found
    assert observer.completed_fixture_steps == {
        FIXTURE_NAMES[0]: 2,
        FIXTURE_NAMES[1]: 4,
    }

    observer.observe(_physical_sample(5, button=0.006))

    assert observer.state.completion_button_press_count == 1
    assert not observer.state.completion_button_pressed
    observer.observe(_physical_sample(6))
    observer.observe(_physical_sample(7))
    observer.observe(_physical_sample(8, button=0.006))

    assert observer.state.completion_button_press_count == 2
    assert observer.state.completion_button_pressed
    assert observer.valid_button_press_steps == [5, 8]


def test_physical_observer_debounces_hold_and_detects_extra_press() -> None:
    target = FIXTURE_NAMES[0]
    observer = _observer(target)
    observer.reset(_physical_sample(0))
    observer.observe(_physical_sample(1, left=0.75))
    observer.observe(_physical_sample(2))

    observer.observe(_physical_sample(3, button=0.006))
    observer.observe(_physical_sample(4, button=0.006))
    observer.observe(_physical_sample(5, button=0.006))
    assert observer.state.completion_button_press_count == 1

    observer.observe(_physical_sample(6))
    observer.observe(_physical_sample(7))
    observer.observe(_physical_sample(8, button=0.006))

    assert observer.state.completion_button_press_count == 2
    assert observer.state.invalid_completion_button_press
    assert (
        observer.state.failure_reason
        == "completion_button_pressed_too_many_times"
    )
    assert observer.valid_button_press_steps == [3]


def test_physical_observer_detects_search_after_target() -> None:
    target = FIXTURE_NAMES[0]
    observer = _observer(target)
    observer.reset(_physical_sample(0))
    observer.observe(_physical_sample(1, left=0.75))
    observer.observe(_physical_sample(2))

    observer.observe(_physical_sample(3, right=0.75))

    assert observer.state.target_found
    assert observer.state.search_after_target_detected
    assert observer.state.failure_reason == "search_continued_after_target_found"


def test_runtime_callbacks_share_one_evaluator_result() -> None:
    target = FIXTURE_NAMES[0]
    evaluator = _evaluator(target)
    snapshot = _snapshot(
        2,
        target=target,
        history=(target,),
        button_steps=(2,),
        success=True,
    )

    class _Env:
        def evaluation_snapshot(self) -> TS04EvaluationSnapshot:
            return snapshot

    runtime = SimpleNamespace(
        env=_Env(),
        metadata={"task_evaluator": evaluator},
    )

    assert success_for_runtime(runtime)
    assert diagnostics_for_runtime(runtime)["success"]
    assert (
        report_stages_for_runtime(runtime)["stages"]["correct_press_count"][
            "passed"
        ]
        is True
    )


def test_runtime_initialize_mounts_fresh_evaluator(monkeypatch) -> None:
    calls: dict[str, object] = {"reset": 0}

    class _FakeEvaluator:
        def __init__(self, **kwargs: object) -> None:
            calls["kwargs"] = dict(kwargs)

        def reset(self, snapshot: object) -> None:
            calls["reset"] = int(calls["reset"]) + 1
            calls["snapshot"] = snapshot

    monkeypatch.setattr(ts04_runtime, "TS04Evaluator", _FakeEvaluator)
    monkeypatch.setattr(
        ts04_runtime,
        "_initialize_drawers",
        lambda descriptor, env, observation: (
            "updated",
            "initialization",
            {},
            "targets",
        ),
    )

    class _FakeEnv:
        fixture_names = FIXTURE_NAMES
        target_fixture = FIXTURE_NAMES[1]

        def evaluation_snapshot(self) -> str:
            return "SNAPSHOT"

    env = _FakeEnv()
    updated, initialization, metadata, targets = ts04_runtime._initialize(
        object(),
        env,
        {},
    )

    assert updated == "updated"
    assert initialization == "initialization"
    assert targets == "targets"
    assert metadata["task_evaluator"] is env._ts04_evaluator
    assert calls == {
        "reset": 1,
        "snapshot": "SNAPSHOT",
        "kwargs": {
            "fixture_names": FIXTURE_NAMES,
            "target_fixture": FIXTURE_NAMES[1],
        },
    }


def test_runtime_registers_public_report_stage_projector() -> None:
    assert ts04_runtime.TASK_RUNTIME_API.report_stage_profile_id == "TS04/report-v1"
    assert callable(ts04_runtime.TASK_RUNTIME_API.report_stages)
