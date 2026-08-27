"""Task-level evaluation tests for TM02."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import bench.tasks.tm02.runtime as tm02_runtime
from bench.evaluation.report_stages import validate_report_stages
from bench.tasks.tm02.evaluation import (
    REPORT_STAGE_ORDER,
    TM02EvaluationSnapshot,
    TM02Evaluator,
    diagnostics_for_runtime,
    report_stages_for_runtime,
    success_for_runtime,
)


def _snapshot(
    step: int,
    *,
    a1: bool = False,
    b1: bool = False,
    a2: bool = False,
    b2: bool = False,
    button: bool = False,
    success: bool = False,
    failure_code: str | None = None,
    event_steps: dict[str, int] | None = None,
) -> TM02EvaluationSnapshot:
    return TM02EvaluationSnapshot(
        step_id=step,
        faucet_on_completed=a1,
        pot_filled_and_can_retrieved=b1,
        fill_pot_thread_completed=a2,
        retrieve_can_thread_completed=b2,
        completion_button_pressed=button,
        absorbing_failure=failure_code is not None,
        failure_code=failure_code,
        domain_success=success,
        event_steps=event_steps or {},
    )


def _evaluator() -> TM02Evaluator:
    evaluator = TM02Evaluator()
    evaluator.reset(_snapshot(0))
    return evaluator


def test_initial_state_does_not_pass_any_stage() -> None:
    result = _evaluator().result()

    assert not result.success
    assert not result.terminal
    assert all(not result.stages[name].qualified for name in REPORT_STAGE_ORDER)


def test_a1_and_combined_b1_qualify_first_two_stages() -> None:
    evaluator = _evaluator()

    a1 = evaluator.observe(
        _snapshot(3, a1=True, event_steps={"A1": 3})
    )
    assert a1.stages["faucet_on"].qualified
    assert a1.stages["faucet_on"].qualified_step == 3
    assert not a1.stages["pot_filled_and_can_retrieved"].qualified

    b1 = evaluator.observe(
        _snapshot(
            7,
            a1=True,
            b1=True,
            event_steps={"A1": 3, "B1": 7},
        )
    )
    assert b1.stages["pot_filled_and_can_retrieved"].qualified
    assert b1.stages["pot_filled_and_can_retrieved"].qualified_step == 7
    assert not b1.stages["both_threads_completed"].qualified


@pytest.mark.parametrize(
    ("a2_step", "b2_step"),
    ((11, 12), (12, 11)),
)
def test_both_thread_orders_qualify_s3_at_later_completion_step(
    a2_step: int,
    b2_step: int,
) -> None:
    evaluator = _evaluator()
    event_steps = {
        "A1": 3,
        "B1": 7,
        "A2": a2_step,
        "B2": b2_step,
    }

    result = evaluator.observe(
        _snapshot(
            12,
            a1=True,
            b1=True,
            a2=True,
            b2=True,
            event_steps=event_steps,
        )
    )

    assert result.stages["faucet_on"].qualified_step == 3
    assert result.stages["pot_filled_and_can_retrieved"].qualified_step == 7
    assert result.stages["both_threads_completed"].qualified
    assert result.stages["both_threads_completed"].qualified_step == 12
    assert not result.success


def test_success_uses_complete_domain_verdict_and_public_schema() -> None:
    evaluator = _evaluator()
    result = evaluator.observe(
        _snapshot(
            13,
            a1=True,
            b1=True,
            a2=True,
            b2=True,
            button=True,
            success=True,
            event_steps={
                "A1": 3,
                "B1": 7,
                "B2": 11,
                "A2": 12,
                "completion_button": 13,
            },
        )
    )

    assert result.success
    assert result.terminal
    report = result.to_report_stages()
    assert report["profile_id"] == "TM02/report-v1"
    assert report["order"] == list(REPORT_STAGE_ORDER)
    assert report["stages"]["both_threads_completed"]["first_pass_step"] == 12
    assert report["stages"]["success"]["first_pass_step"] == 13
    validate_report_stages(report, expected_task_id="TM02", expected_success=True)


def test_final_state_regression_revokes_only_success() -> None:
    evaluator = _evaluator()
    event_steps = {
        "A1": 3,
        "B1": 7,
        "B2": 11,
        "A2": 12,
        "completion_button": 13,
    }
    evaluator.observe(
        _snapshot(
            13,
            a1=True,
            b1=True,
            a2=True,
            b2=True,
            button=True,
            success=True,
            event_steps=event_steps,
        )
    )

    result = evaluator.observe(
        _snapshot(
            14,
            a1=True,
            b1=True,
            a2=True,
            b2=True,
            button=True,
            event_steps=event_steps,
        )
    )

    assert all(
        result.stages[name].qualified
        for name in REPORT_STAGE_ORDER[:-1]
    )
    assert not result.success
    assert result.stages["success"].invalidated
    assert result.stages["success"].reason == "final_state_regressed"
    assert result.to_report_stages()["stages"]["success"] == {
        "applicable": True,
        "passed": False,
        "first_pass_step": None,
    }


def test_absorbing_failure_blocks_success_but_preserves_completed_history() -> None:
    evaluator = _evaluator()
    result = evaluator.observe(
        _snapshot(
            13,
            a1=True,
            b1=True,
            a2=True,
            b2=True,
            button=True,
            failure_code="duplicate_completion_button_press",
            event_steps={"A1": 3, "B1": 7, "B2": 11, "A2": 12},
        )
    )

    assert result.terminal
    assert result.failure_code == "duplicate_completion_button_press"
    assert all(
        result.stages[name].qualified
        for name in REPORT_STAGE_ORDER[:-1]
    )
    assert not result.stages["success"].qualified
    assert result.stages["success"].invalidated


def test_same_step_is_idempotent_and_backward_step_is_rejected() -> None:
    evaluator = TM02Evaluator()
    evaluator.reset(_snapshot(1))
    initial = evaluator.result()

    assert evaluator.observe(_snapshot(1, a1=True)) is initial
    with pytest.raises(ValueError, match="moved backwards"):
        evaluator.observe(_snapshot(0))


def test_runtime_callbacks_share_one_evaluator() -> None:
    evaluator = _evaluator()
    snapshot = _snapshot(
        13,
        a1=True,
        b1=True,
        a2=True,
        b2=True,
        button=True,
        success=True,
        event_steps={
            "A1": 3,
            "B1": 7,
            "B2": 11,
            "A2": 12,
            "completion_button": 13,
        },
    )

    class _Env:
        def evaluation_snapshot(self) -> TM02EvaluationSnapshot:
            return snapshot

    runtime = SimpleNamespace(
        env=_Env(),
        metadata={"task_evaluator": evaluator},
    )

    assert success_for_runtime(runtime)
    assert diagnostics_for_runtime(runtime)["success"]
    assert report_stages_for_runtime(runtime)["stages"]["success"]["passed"]


def test_runtime_initialization_configures_physics_and_mounts_evaluator(
    monkeypatch,
) -> None:
    calls: dict[str, object] = {"physics": 0, "reset": 0}

    class _FakeEvaluator:
        def reset(self, snapshot: object) -> None:
            calls["reset"] = int(calls["reset"]) + 1
            calls["snapshot"] = snapshot

    monkeypatch.setattr(tm02_runtime, "TM02Evaluator", _FakeEvaluator)
    monkeypatch.setattr(
        tm02_runtime,
        "_initialize_base",
        lambda descriptor, env, observation: (
            {"updated": True},
            {"base": True},
            {"base_metadata": True},
            "targets",
        ),
    )

    class _Env:
        def configure_canonical_physics(self) -> None:
            calls["physics"] = int(calls["physics"]) + 1

        def evaluation_snapshot(self) -> str:
            return "SNAPSHOT"

    env = _Env()
    result = tm02_runtime._initialize(object(), env, {})

    assert result[:2] == ({"updated": True}, {"base": True})
    assert result[3] == "targets"
    assert result[2]["base_metadata"] is True
    assert result[2]["task_evaluator"] is env._tm02_evaluator
    assert calls == {"physics": 1, "reset": 1, "snapshot": "SNAPSHOT"}


def test_runtime_registers_evaluator_backed_callbacks() -> None:
    api = tm02_runtime.TASK_RUNTIME_API
    assert api.report_stage_profile_id == "TM02/report-v1"
    assert callable(api.report_stages)
    assert tm02_runtime._ADAPTER._diagnostics is diagnostics_for_runtime
    assert tm02_runtime._ADAPTER._success is success_for_runtime
