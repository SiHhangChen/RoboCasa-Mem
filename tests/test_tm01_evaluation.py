"""Task-level evaluation tests for TM01."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import bench.tasks.tm01.runtime as tm01_runtime
from bench.evaluation.report_stages import validate_report_stages
from bench.tasks.tm01.evaluation import (
    REPORT_STAGE_ORDER,
    TM01EvaluationSnapshot,
    TM01Evaluator,
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
) -> TM01EvaluationSnapshot:
    return TM01EvaluationSnapshot(
        step_id=step,
        burner_on_completed=a1,
        one_drawer_inspected=b1,
        boiling_thread_completed=a2,
        drawer_thread_completed=b2,
        completion_button_pressed=button,
        absorbing_failure=failure_code is not None,
        failure_code=failure_code,
        domain_success=success,
        event_steps=event_steps or {},
    )


def _evaluator() -> TM01Evaluator:
    evaluator = TM01Evaluator()
    evaluator.reset(_snapshot(0))
    return evaluator


def test_initial_state_does_not_pass_any_stage() -> None:
    result = _evaluator().result()

    assert not result.success
    assert not result.terminal
    assert all(not result.stages[name].qualified for name in REPORT_STAGE_ORDER)


def test_a1_and_complete_drawer_cycle_qualify_first_two_stages() -> None:
    evaluator = _evaluator()

    a1 = evaluator.observe(
        _snapshot(2, a1=True, event_steps={"A1": 2})
    )
    assert a1.stages["burner_on"].qualified
    assert a1.stages["burner_on"].qualified_step == 2
    assert not a1.stages["one_drawer_inspected"].qualified

    b1 = evaluator.observe(
        _snapshot(
            5,
            a1=True,
            b1=True,
            event_steps={"A1": 2, "B1": 5},
        )
    )
    assert b1.stages["one_drawer_inspected"].qualified
    assert b1.stages["one_drawer_inspected"].qualified_step == 5
    assert not b1.stages["both_threads_completed"].qualified


@pytest.mark.parametrize(
    ("a2_step", "b2_step"),
    ((8, 9), (9, 8)),
)
def test_both_thread_orders_qualify_s3_and_s4_at_semantic_steps(
    a2_step: int,
    b2_step: int,
) -> None:
    evaluator = _evaluator()
    event_steps = {
        "A1": 2,
        "B1": 5,
        "A2": a2_step,
        "B2": b2_step,
    }

    result = evaluator.observe(
        _snapshot(
            9,
            a1=True,
            b1=True,
            a2=True,
            b2=True,
            event_steps=event_steps,
        )
    )

    assert result.stages["burner_on"].qualified_step == 2
    assert result.stages["one_drawer_inspected"].qualified_step == 5
    assert result.stages["one_thread_completed"].qualified_step == 8
    assert result.stages["both_threads_completed"].qualified
    assert result.stages["both_threads_completed"].qualified_step == 9
    assert not result.success


def test_success_uses_complete_domain_verdict_and_public_schema() -> None:
    evaluator = _evaluator()
    result = evaluator.observe(
        _snapshot(
            10,
            a1=True,
            b1=True,
            a2=True,
            b2=True,
            button=True,
            success=True,
            event_steps={
                "A1": 2,
                "B1": 5,
                "A2": 9,
                "B2": 8,
                "completion_button": 10,
            },
        )
    )

    assert result.success
    assert result.terminal
    report = result.to_report_stages()
    assert report["profile_id"] == "TM01/report-v2"
    assert report["order"] == list(REPORT_STAGE_ORDER)
    assert report["stages"]["both_threads_completed"]["first_pass_step"] == 9
    assert report["stages"]["success"]["first_pass_step"] == 10
    validate_report_stages(report, expected_task_id="TM01", expected_success=True)


def test_final_state_regression_revokes_only_success() -> None:
    evaluator = _evaluator()
    event_steps = {
        "A1": 2,
        "B1": 5,
        "A2": 8,
        "B2": 9,
        "completion_button": 10,
    }
    evaluator.observe(
        _snapshot(
            10,
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
            11,
            a1=True,
            b1=True,
            a2=True,
            b2=True,
            button=True,
            success=False,
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
            10,
            a1=True,
            b1=True,
            a2=True,
            b2=True,
            button=True,
            failure_code="invalid_completion_button_press",
            event_steps={"A1": 2, "B1": 5, "A2": 8, "B2": 9},
        )
    )

    assert result.terminal
    assert result.failure_code == "invalid_completion_button_press"
    assert all(
        result.stages[name].qualified
        for name in REPORT_STAGE_ORDER[:-1]
    )
    assert not result.stages["success"].qualified
    assert result.stages["success"].invalidated


def test_same_step_is_idempotent_and_backward_step_is_rejected() -> None:
    evaluator = TM01Evaluator()
    evaluator.reset(_snapshot(1))
    initial = evaluator.result()

    assert evaluator.observe(_snapshot(1, a1=True)) is initial
    with pytest.raises(ValueError, match="moved backwards"):
        evaluator.observe(_snapshot(0))


def test_runtime_callbacks_share_one_evaluator() -> None:
    evaluator = _evaluator()
    snapshot = _snapshot(
        10,
        a1=True,
        b1=True,
        a2=True,
        b2=True,
        button=True,
        success=True,
        event_steps={
            "A1": 2,
            "B1": 5,
            "A2": 8,
            "B2": 9,
            "completion_button": 10,
        },
    )

    class _Env:
        def evaluation_snapshot(self) -> TM01EvaluationSnapshot:
            return snapshot

    runtime = SimpleNamespace(
        env=_Env(),
        metadata={"task_evaluator": evaluator},
    )

    assert success_for_runtime(runtime)
    assert diagnostics_for_runtime(runtime)["success"]
    assert report_stages_for_runtime(runtime)["stages"]["success"]["passed"]


def test_runtime_initialization_mounts_fresh_evaluator(monkeypatch) -> None:
    calls: dict[str, object] = {"reset": 0}

    class _FakeEvaluator:
        def reset(self, snapshot: object) -> None:
            calls["reset"] = int(calls["reset"]) + 1
            calls["snapshot"] = snapshot

    monkeypatch.setattr(tm01_runtime, "TM01Evaluator", _FakeEvaluator)
    monkeypatch.setattr(
        tm01_runtime,
        "_initialize_base",
        lambda descriptor, env, observation: (
            {"updated": True},
            {"base": True},
            {"base_metadata": True},
            "targets",
        ),
    )

    class _Env:
        def evaluation_snapshot(self) -> str:
            return "SNAPSHOT"

    env = _Env()
    result = tm01_runtime._initialize(object(), env, {})

    assert result[:2] == ({"updated": True}, {"base": True})
    assert result[3] == "targets"
    assert result[2]["base_metadata"] is True
    assert result[2]["task_evaluator"] is env._tm01_evaluator
    assert calls == {"reset": 1, "snapshot": "SNAPSHOT"}


def test_runtime_registers_evaluator_backed_callbacks() -> None:
    api = tm01_runtime.TASK_RUNTIME_API
    assert api.report_stage_profile_id == "TM01/report-v2"
    assert callable(api.report_stages)
    assert tm01_runtime._ADAPTER._diagnostics is diagnostics_for_runtime
    assert tm01_runtime._ADAPTER._success is success_for_runtime
