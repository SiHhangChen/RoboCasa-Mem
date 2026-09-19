"""Task-level evaluation tests for WX-01."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import bench.tasks.wx01.runtime as wx01_runtime
from bench.evaluation.report_stages import validate_report_stages
from bench.tasks.wx01.evaluation import (
    REPORT_STAGE_ORDER,
    TASK_EVALUATOR_METADATA_KEY,
    WX01EvaluationSnapshot,
    WX01Evaluator,
    diagnostics_for_runtime,
    report_stages_for_runtime,
    success_for_runtime,
)


def _snapshot(
    step: int,
    *,
    wiped: bool = False,
    tool_released: bool = False,
    object_at_origin: bool = False,
    object_on_table: bool = False,
    object_released: bool = False,
    success: bool = False,
) -> WX01EvaluationSnapshot:
    return WX01EvaluationSnapshot(
        step_id=step,
        remembered_area_wiped=wiped,
        cleaning_tool_released=tool_released,
        blocking_object_at_origin=object_at_origin,
        blocking_object_on_dining_table=object_on_table,
        blocking_object_released=object_released,
        domain_success=success,
        details={"legacy_check": True},
    )


def _evaluator(*, initially_released: bool = False) -> WX01Evaluator:
    evaluator = WX01Evaluator()
    evaluator.reset(_snapshot(0, tool_released=initially_released))
    return evaluator


def _complete(evaluator: WX01Evaluator):
    evaluator.observe(_snapshot(1, wiped=True))
    evaluator.observe(_snapshot(2, wiped=True, tool_released=True))
    return evaluator.observe(
        _snapshot(
            3,
            wiped=True,
            tool_released=True,
            object_at_origin=True,
            object_on_table=True,
            # Preserve the existing WX-01 final condition: cup release is
            # diagnostic evidence but is not required for task success.
            object_released=False,
            success=True,
        )
    )


def test_wx01_evaluator_qualifies_public_funnel() -> None:
    result = _complete(_evaluator())

    assert result.success
    assert result.terminal
    assert {
        name: result.stages[name].qualified_step for name in REPORT_STAGE_ORDER
    } == {
        "wipe_completed": 1,
        "cleaning_tool_released": 2,
        "success": 3,
    }
    diagnostics = result.to_diagnostics()
    assert diagnostics["legacy_check"]
    assert diagnostics["stage_checks"] == {
        name: True for name in REPORT_STAGE_ORDER
    }

    report = result.to_report_stages()
    assert report == {
        "schema_version": "membench.report_stages/v1",
        "task_id": "WX-01",
        "profile_id": "WX-01/report-v2",
        "order": ["wipe_completed", "cleaning_tool_released", "success"],
        "stages": {
            "wipe_completed": {
                "applicable": True,
                "passed": True,
                "first_pass_step": 1,
            },
            "cleaning_tool_released": {
                "applicable": True,
                "passed": True,
                "first_pass_step": 2,
            },
            "success": {
                "applicable": True,
                "passed": True,
                "first_pass_step": 3,
            },
        },
    }
    assert validate_report_stages(
        report,
        expected_task_id="WX-01",
        expected_profile_id="WX-01/report-v2",
        expected_success=True,
    ) == report


def test_initial_tool_release_does_not_complete_cleaning_before_wipe() -> None:
    evaluator = _evaluator(initially_released=True)
    initial = evaluator.result()
    unchanged = evaluator.observe(_snapshot(1, tool_released=True))

    assert not initial.stages["wipe_completed"].qualified
    assert not initial.stages["cleaning_tool_released"].observed
    assert not initial.stages["cleaning_tool_released"].qualified
    assert not unchanged.stages["cleaning_tool_released"].qualified
    assert unchanged.stages["cleaning_tool_released"].reason == (
        "waiting_for_prerequisite:wipe_completed"
    )


def test_wipe_while_tool_is_held_only_qualifies_first_stage() -> None:
    result = _evaluator().observe(_snapshot(1, wiped=True))

    assert result.stages["wipe_completed"].qualified
    assert not result.stages["cleaning_tool_released"].observed
    assert not result.stages["cleaning_tool_released"].qualified
    assert result.stages["cleaning_tool_released"].reason == "not_observed"


def test_cleaning_completion_is_historical_after_post_wipe_release() -> None:
    evaluator = _evaluator()
    evaluator.observe(_snapshot(1, wiped=True))
    released = evaluator.observe(
        _snapshot(2, wiped=True, tool_released=True)
    )
    picked_again = evaluator.observe(
        _snapshot(3, wiped=True, tool_released=False)
    )

    assert released.stages["cleaning_tool_released"].qualified_step == 2
    assert picked_again.stages["cleaning_tool_released"].qualified
    assert picked_again.stages["cleaning_tool_released"].qualified_step == 2


def test_success_is_revoked_when_final_object_state_regresses() -> None:
    evaluator = _evaluator()
    success = _complete(evaluator)
    regressed = evaluator.observe(
        _snapshot(
            4,
            wiped=True,
            tool_released=True,
            object_at_origin=False,
            object_on_table=True,
        )
    )

    assert success.success
    assert not regressed.success
    assert not regressed.terminal
    assert regressed.stages["wipe_completed"].qualified
    assert regressed.stages["cleaning_tool_released"].qualified
    assert not regressed.stages["success"].qualified
    assert regressed.stages["success"].invalidated
    assert regressed.stages["success"].reason == "final_state_regressed"
    assert not regressed.to_report_stages()["stages"]["success"]["passed"]


def test_snapshot_rejects_inconsistent_domain_success() -> None:
    with pytest.raises(ValueError, match="domain_success requires"):
        _snapshot(1, success=True)


def test_public_report_does_not_expose_expert_action_steps() -> None:
    report = _complete(_evaluator()).to_report_stages()

    assert set(report["stages"]) == set(REPORT_STAGE_ORDER)
    for internal_action in (
        "navigate",
        "pick",
        "return_arm",
        "place_cleaning_tool",
    ):
        assert internal_action not in report["stages"]


def test_runtime_callbacks_share_initialized_evaluator() -> None:
    evaluator = _evaluator()

    class FakeEnv:
        snapshot = _snapshot(1, wiped=True)

        def evaluation_snapshot(self) -> WX01EvaluationSnapshot:
            return self.snapshot

    runtime = SimpleNamespace(
        env=FakeEnv(),
        metadata={TASK_EVALUATOR_METADATA_KEY: evaluator},
    )

    diagnostics = diagnostics_for_runtime(runtime)
    assert diagnostics["stage_checks"]["wipe_completed"]
    assert not success_for_runtime(runtime, {})
    assert report_stages_for_runtime(runtime)["order"] == list(
        REPORT_STAGE_ORDER
    )


def test_runtime_initialize_mounts_fresh_evaluator(monkeypatch) -> None:
    calls: dict[str, object] = {}

    class FakeEvaluator:
        def reset(self, snapshot: object) -> None:
            calls["snapshot"] = snapshot

    monkeypatch.setattr(wx01_runtime, "WX01Evaluator", FakeEvaluator)
    monkeypatch.setattr(
        wx01_runtime,
        "apply_initial_object_plan",
        lambda env, episode, enabled: ("objects_obs", "objects"),
    )
    monkeypatch.setattr(
        wx01_runtime,
        "apply_robot_base_initial_state",
        lambda env, obs, config, navigation, seed: ("base_obs", "base"),
    )
    monkeypatch.setattr(
        wx01_runtime,
        "resolve_episode_targets",
        lambda env, plan: "targets",
    )
    monkeypatch.setattr(
        wx01_runtime,
        "build_manipulation_assembly",
        lambda env, obs, settings, planner_script, execute: SimpleNamespace(
            controller="controller"
        ),
    )
    monkeypatch.setattr(
        wx01_runtime,
        "initialize_short_start_state",
        lambda env, controller, targets, config, grasp_pose_file: (
            "short_obs",
            "short_start",
        ),
    )
    descriptor = SimpleNamespace(
        episode_plan=SimpleNamespace(seed=7),
        config=SimpleNamespace(
            initial_state=SimpleNamespace(
                objects={"enabled": True},
                robot_base="robot_base",
            ),
            manipulation="manipulation",
        ),
        navigation="navigation",
        resources={
            "planner_script": "planner.py",
            "cleaning_tool_grasp_pose": "grasp.json",
        },
        execute=False,
    )

    class FakeEnv:
        _wx01_evaluator = None

        def evaluation_snapshot(self) -> str:
            return "SNAPSHOT"

    env = FakeEnv()
    updated, initialization, metadata, targets = wx01_runtime._initialize(
        descriptor,
        env,
        {},
    )

    assert updated == "short_obs"
    assert initialization == {
        "objects": "objects",
        "base": "base",
        "short_start": "short_start",
    }
    assert targets == "targets"
    assert metadata[TASK_EVALUATOR_METADATA_KEY] is env._wx01_evaluator
    assert calls["snapshot"] == "SNAPSHOT"


def test_runtime_registers_public_report_stage_projector() -> None:
    assert wx01_runtime.TASK_RUNTIME_API.report_stage_profile_id == (
        "WX-01/report-v2"
    )
    assert callable(wx01_runtime.TASK_RUNTIME_API.report_stages)
