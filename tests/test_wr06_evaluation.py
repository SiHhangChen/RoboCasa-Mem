"""Task-level evaluation tests for WR-06."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import bench.tasks.wr06.runtime as wr06_runtime
from bench.evaluation.report_stages import validate_report_stages
from bench.tasks.wr06.evaluation import (
    REPORT_STAGE_ORDER,
    TASK_EVALUATOR_METADATA_KEY,
    WR06EvaluationSnapshot,
    WR06Evaluator,
    diagnostics_for_runtime,
    report_stages_for_runtime,
    success_for_runtime,
)


def _snapshot(
    step: int,
    *,
    removed: bool = False,
    staged: bool = False,
    barrier: bool = False,
    first_return_level: str | None = None,
    wrong_return: bool = False,
    returned: bool = False,
    protocol_violation: bool = False,
    regressions: tuple[str, ...] = (),
    success: bool = False,
    failure_code: str | None = None,
) -> WR06EvaluationSnapshot:
    return WR06EvaluationSnapshot(
        step_id=step,
        initial_level="shelf_lower",
        initial_level_physically_verified=True,
        item_removed_from_home=removed,
        item_staged_at_plate=staged,
        memory_barrier_latched=barrier,
        first_return_level=first_return_level,
        wrong_return_level_attempted=wrong_return,
        returned_item_id="ketchup" if returned else None,
        protocol_violation=protocol_violation,
        stage_regressions=regressions,
        domain_success=success,
        failure_code=failure_code,
        details={"legacy_check": True},
    )


def _evaluator() -> WR06Evaluator:
    evaluator = WR06Evaluator()
    evaluator.reset(_snapshot(0))
    return evaluator


def _complete(evaluator: WR06Evaluator):
    evaluator.observe(_snapshot(1, removed=True))
    evaluator.observe(
        _snapshot(2, removed=True, staged=True, barrier=True)
    )
    evaluator.observe(
        _snapshot(
            3,
            removed=True,
            staged=True,
            barrier=True,
            first_return_level="shelf_lower",
        )
    )
    return evaluator.observe(
        _snapshot(
            4,
            removed=True,
            staged=True,
            barrier=True,
            first_return_level="shelf_lower",
            returned=True,
            success=True,
        )
    )


def test_wr06_evaluator_qualifies_public_funnel() -> None:
    result = _complete(_evaluator())

    assert result.success
    assert result.terminal
    assert {
        name: result.stages[name].qualified_step for name in REPORT_STAGE_ORDER
    } == {
        "item_removed": 1,
        "item_staged": 2,
        "correct_return_level": 3,
        "success": 4,
    }
    assert result.to_diagnostics()["stage_checks"] == {
        name: True for name in REPORT_STAGE_ORDER
    }

    report = result.to_report_stages()
    assert report["task_id"] == "WR-06"
    assert report["profile_id"] == "WR-06/report-v1"
    assert report["order"] == list(REPORT_STAGE_ORDER)
    assert set(report["stages"]) == set(REPORT_STAGE_ORDER)
    assert "memory_barrier_latched" not in report["order"]
    assert "memory_barrier_latched" not in report["stages"]
    assert validate_report_stages(
        report,
        expected_task_id="WR-06",
        expected_profile_id="WR-06/report-v1",
        expected_success=True,
    ) == report


def test_initial_home_state_does_not_count_as_removal_or_return() -> None:
    result = _evaluator().result()

    assert not result.success
    assert not result.stages["item_removed"].observed
    assert not result.stages["item_removed"].qualified
    assert not result.stages["correct_return_level"].observed
    assert result.stages["correct_return_level"].reason == (
        "waiting_for_prerequisite:item_staged"
    )


def test_staging_requires_removal_to_have_qualified() -> None:
    evaluator = _evaluator()
    staged_too_early = evaluator.observe(
        _snapshot(1, removed=False, staged=True)
    )

    assert staged_too_early.stages["item_staged"].observed
    assert not staged_too_early.stages["item_staged"].qualified
    assert staged_too_early.stages["item_staged"].reason == (
        "prerequisite_not_qualified:item_removed"
    )


def test_wrong_first_return_level_cannot_be_repaired_later() -> None:
    evaluator = _evaluator()
    evaluator.observe(_snapshot(1, removed=True))
    evaluator.observe(
        _snapshot(2, removed=True, staged=True, barrier=True)
    )
    wrong = evaluator.observe(
        _snapshot(
            3,
            removed=True,
            staged=True,
            barrier=True,
            first_return_level="shelf_upper",
            wrong_return=True,
            protocol_violation=True,
            failure_code="wrong_return_level",
        )
    )
    corrected_position = evaluator.observe(
        _snapshot(
            4,
            removed=True,
            staged=True,
            barrier=True,
            first_return_level="shelf_lower",
            wrong_return=True,
            protocol_violation=True,
            failure_code="wrong_return_level",
        )
    )

    assert wrong.terminal
    assert wrong.failure_code == "wrong_return_level"
    assert not wrong.stages["correct_return_level"].qualified
    assert wrong.stages["correct_return_level"].reason == (
        "wrong_first_return_level:shelf_upper"
    )
    assert not corrected_position.stages["correct_return_level"].qualified
    assert corrected_position.failure_code == "wrong_return_level"


def test_success_is_revoked_when_final_state_regresses() -> None:
    evaluator = _evaluator()
    success = _complete(evaluator)
    regressed = evaluator.observe(
        _snapshot(
            5,
            removed=True,
            staged=True,
            barrier=True,
            first_return_level="shelf_lower",
            returned=True,
            protocol_violation=True,
            regressions=("item_removed_after_success",),
            failure_code="item_removed_after_success",
        )
    )

    assert success.success
    assert not regressed.success
    assert regressed.terminal
    assert regressed.failure_code == "item_removed_after_success"
    assert regressed.stages["correct_return_level"].qualified
    assert not regressed.stages["success"].qualified
    assert regressed.stages["success"].invalidated
    assert regressed.stages["success"].reason == "final_state_regressed"
    assert not regressed.to_report_stages()["stages"]["success"]["passed"]


def test_snapshot_rejects_inconsistent_domain_success() -> None:
    with pytest.raises(ValueError, match="domain_success requires"):
        _snapshot(1, success=True)


def test_runtime_callbacks_share_initialized_evaluator() -> None:
    evaluator = _evaluator()

    class FakeEnv:
        snapshot = _snapshot(1, removed=True)

        def evaluation_snapshot(self) -> WR06EvaluationSnapshot:
            return self.snapshot

    runtime = SimpleNamespace(
        env=FakeEnv(),
        metadata={TASK_EVALUATOR_METADATA_KEY: evaluator},
    )

    diagnostics = diagnostics_for_runtime(runtime)
    assert diagnostics["stage_checks"]["item_removed"]
    assert not success_for_runtime(runtime, {})
    assert report_stages_for_runtime(runtime)["order"] == list(
        REPORT_STAGE_ORDER
    )


def test_runtime_initialize_mounts_fresh_evaluator(monkeypatch) -> None:
    calls: dict[str, object] = {}

    class FakeEvaluator:
        def reset(self, snapshot: object) -> None:
            calls["snapshot"] = snapshot

    monkeypatch.setattr(wr06_runtime, "WR06Evaluator", FakeEvaluator)
    monkeypatch.setattr(
        wr06_runtime,
        "_initialize_shelf",
        lambda descriptor, env, observation: (
            "updated",
            "initialization",
            {"existing": True},
            "targets",
        ),
    )

    class FakeEnv:
        _wr06_evaluator = None

        def evaluation_snapshot(self) -> str:
            return "SNAPSHOT"

    env = FakeEnv()
    updated, initialization, metadata, targets = wr06_runtime._initialize(
        object(),
        env,
        {},
    )

    assert updated == "updated"
    assert initialization == "initialization"
    assert targets == "targets"
    assert metadata["existing"] is True
    assert metadata[TASK_EVALUATOR_METADATA_KEY] is env._wr06_evaluator
    assert calls["snapshot"] == "SNAPSHOT"


def test_runtime_registers_public_report_stage_projector() -> None:
    assert wr06_runtime.TASK_RUNTIME_API.report_stage_profile_id == (
        "WR-06/report-v1"
    )
    assert callable(wr06_runtime.TASK_RUNTIME_API.report_stages)
