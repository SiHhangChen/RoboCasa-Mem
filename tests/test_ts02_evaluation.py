"""Task-level evaluation tests for TS02 drawer search."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import bench.tasks.ts02.runtime as ts02_runtime
from bench.tasks.ts02.domain.contracts import FIXTURE_NAMES
from bench.tasks.ts02.evaluation import (
    REPORT_STAGE_ORDER,
    TS02EvaluationSnapshot,
    TS02Evaluator,
    diagnostics_for_runtime,
    report_stages_for_runtime,
    success_for_runtime,
)


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
) -> TS02EvaluationSnapshot:
    target_found = target in history
    return TS02EvaluationSnapshot(
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


def _evaluator(target: str) -> TS02Evaluator:
    evaluator = TS02Evaluator(
        fixture_names=FIXTURE_NAMES,
        target_fixture=target,
    )
    evaluator.reset(_snapshot(0, target=target))
    return evaluator


@pytest.mark.parametrize("search_length", (1, 2, 3))
def test_five_stage_funnel_supports_all_search_lengths(
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

    button_steps: tuple[int, ...] = ()
    for press_count in range(1, search_length + 1):
        step = search_length + press_count
        button_steps = (*button_steps, step)
        result = evaluator.observe(
            _snapshot(
                step,
                target=target,
                history=history,
                button_steps=button_steps,
                success=press_count == search_length,
            )
        )
        assert result.stages["button_pressed"].qualified
        assert result.stages["button_pressed"].qualified_step == search_length + 1
        assert result.stages["correct_press_count"].qualified is (
            press_count == search_length
        )

    assert result.success
    assert result.stages["correct_press_count"].qualified_step == (
        search_length * 2
    )


def test_extra_press_revokes_only_the_episode_level_final_verdict() -> None:
    target = FIXTURE_NAMES[0]
    evaluator = _evaluator(target)
    evaluator.observe(
        _snapshot(1, target=target, history=(target,))
    )
    succeeded = evaluator.observe(
        _snapshot(
            2,
            target=target,
            history=(target,),
            button_steps=(2,),
            success=True,
        )
    )
    assert succeeded.success

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
    assert not extra.success
    assert (
        extra.stages["correct_press_count"].reason
        == "completion_button_count_does_not_match_search_count"
    )


def test_later_search_revokes_stop_and_all_downstream_stages() -> None:
    target = FIXTURE_NAMES[0]
    evaluator = _evaluator(target)
    evaluator.observe(
        _snapshot(1, target=target, history=(target,))
    )
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
    assert not continued.stages["search_stopped_after_target"].qualified
    assert continued.stages["search_stopped_after_target"].invalidated
    assert not continued.stages["button_pressed"].qualified
    assert continued.stages["button_pressed"].invalidated
    assert not continued.stages["correct_press_count"].qualified
    assert continued.stages["correct_press_count"].invalidated

    report = continued.to_report_stages()
    assert report["stages"]["target_found"]["passed"] is True
    for name in (
        "search_stopped_after_target",
        "button_pressed",
        "correct_press_count",
    ):
        assert report["stages"][name]["passed"] is False
        assert report["stages"][name]["first_pass_step"] is None


def test_public_report_has_exactly_five_stages_and_no_extra_success() -> None:
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

    assert report["profile_id"] == "TS02/report-v1"
    assert report["order"] == list(REPORT_STAGE_ORDER)
    assert tuple(report["stages"]) == REPORT_STAGE_ORDER
    assert "success" not in report["stages"]
    assert report["stages"]["correct_press_count"] == {
        "applicable": True,
        "passed": True,
        "first_pass_step": 2,
    }


@pytest.mark.parametrize(
    ("checked_count", "target_found"),
    ((0, False), (1, False), (1, True), (2, True), (3, True)),
)
def test_snapshot_press_requirement_matches_completed_search(
    checked_count: int, target_found: bool,
) -> None:
    history = FIXTURE_NAMES[:checked_count]
    target = history[-1] if target_found else FIXTURE_NAMES[-1]
    required_count = checked_count if target_found else 0
    values = {
        "step_id": checked_count,
        "fixture_names": FIXTURE_NAMES,
        "target_fixture": target,
        "completed_fixture_history": history,
        "completed_fixture_steps": {
            fixture: index for index, fixture in enumerate(history, start=1)
        },
        "target_found": target_found,
    }
    snapshot = TS02EvaluationSnapshot(
        **values, required_button_press_count=required_count,
    )
    assert snapshot.required_button_press_count == required_count
    with pytest.raises(ValueError, match="required_button_press_count"):
        TS02EvaluationSnapshot(
            **values, required_button_press_count=required_count + 1,
        )


def test_same_step_is_idempotent_even_if_snapshot_contents_change() -> None:
    target = FIXTURE_NAMES[0]
    evaluator = TS02Evaluator(
        fixture_names=FIXTURE_NAMES,
        target_fixture=target,
    )
    evaluator.reset(_snapshot(1, target=target))
    initial = evaluator.result()

    same_step = evaluator.observe(
        _snapshot(1, target=target, history=(target,))
    )

    assert same_step is initial
    assert not same_step.stages["one_drawer_checked"].qualified


def test_backward_step_is_rejected() -> None:
    target = FIXTURE_NAMES[0]
    evaluator = _evaluator(target)
    evaluator.observe(_snapshot(2, target=target))

    with pytest.raises(ValueError, match="moved backwards"):
        evaluator.observe(_snapshot(1, target=target))


def test_snapshot_configuration_cannot_change_within_episode() -> None:
    evaluator = _evaluator(FIXTURE_NAMES[0])

    with pytest.raises(ValueError, match="configuration changed"):
        evaluator.observe(
            _snapshot(1, target=FIXTURE_NAMES[1])
        )


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
        def evaluation_snapshot(self) -> TS02EvaluationSnapshot:
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

    monkeypatch.setattr(ts02_runtime, "TS02Evaluator", _FakeEvaluator)
    monkeypatch.setattr(
        ts02_runtime,
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
    updated, initialization, metadata, targets = ts02_runtime._initialize(
        object(),
        env,
        {},
    )

    assert updated == "updated"
    assert initialization == "initialization"
    assert targets == "targets"
    assert metadata["task_evaluator"] is env._ts02_evaluator
    assert calls == {
        "reset": 1,
        "snapshot": "SNAPSHOT",
        "kwargs": {
            "fixture_names": FIXTURE_NAMES,
            "target_fixture": FIXTURE_NAMES[1],
        },
    }


def test_runtime_registers_public_report_stage_projector() -> None:
    assert ts02_runtime.TASK_RUNTIME_API.report_stage_profile_id == "TS02/report-v1"
    assert callable(ts02_runtime.TASK_RUNTIME_API.report_stages)
