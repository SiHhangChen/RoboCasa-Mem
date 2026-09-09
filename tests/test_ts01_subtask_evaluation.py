from __future__ import annotations

from dataclasses import dataclass

from bench.tasks.ts01 import subtask_evaluation


@dataclass
class _Stage:
    qualified: bool


class _State:
    def __init__(
        self,
        *,
        sugar_goal_ready: bool,
        success: bool,
        failure_code=None,
    ):
        self.stages = {"success": _Stage(success)}
        self.failure_code = failure_code
        self.details = {"sugar_goal_ready": sugar_goal_ready}


class _Env:
    timestep = 0


class _Runtime:
    env = _Env()


def test_ts01_oracle_scheduler_switches_prompts(monkeypatch):
    states = iter(
        [
            _State(sugar_goal_ready=False, success=False),
            _State(sugar_goal_ready=True, success=False),
            _State(sugar_goal_ready=True, success=True),
        ]
    )
    monkeypatch.setattr(
        subtask_evaluation,
        "evaluate_initialized_runtime",
        lambda runtime: next(states),
    )

    runtime = _Runtime()
    scheduler = subtask_evaluation.TS01OracleSubtaskScheduler(runtime)
    assert scheduler.current_prompt == "add a sugar cube to the hot chocolate"

    runtime.env.timestep = 100
    event = scheduler.after_step(runtime, {})
    assert event.changed is True
    assert event.current_subtask == "press_button"
    assert event.prompt == "press the completion button"

    runtime.env.timestep = 200
    event = scheduler.after_step(runtime, {})
    assert event.complete is True
    assert scheduler.to_metadata()["finished"] is True


def test_ts01_oracle_scheduler_reports_over_add(monkeypatch):
    states = iter(
        [
            _State(sugar_goal_ready=False, success=False),
            _State(sugar_goal_ready=False, success=False, failure_code="over_add"),
        ]
    )
    monkeypatch.setattr(
        subtask_evaluation,
        "evaluate_initialized_runtime",
        lambda runtime: next(states),
    )

    runtime = _Runtime()
    scheduler = subtask_evaluation.TS01OracleSubtaskScheduler(runtime)
    event = scheduler.after_step(runtime, {})

    assert event.failed is True
    assert event.failure_reason == "over_add"
    assert scheduler.to_metadata()["failed"] is True
