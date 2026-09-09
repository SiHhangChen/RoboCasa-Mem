from __future__ import annotations

from dataclasses import dataclass

from bench.tasks.wa01 import subtask_evaluation


@dataclass
class _Stage:
    qualified: bool


class _State:
    def __init__(self, *, stages: dict[str, bool], failure_code=None):
        self.stages = {name: _Stage(value) for name, value in stages.items()}
        self.failure_code = failure_code
        self.details = {
            "sugar": {"sugared_coffee_id": "coffee_A"},
            "dining_placement": {
                "planned_slot_by_coffee": {"coffee_A": "left", "coffee_B": "right"}
            },
        }


class _Env:
    timestep = 0


class _Runtime:
    env = _Env()


def test_wa01_oracle_scheduler_switches_prompts(monkeypatch):
    states = iter(
        [
            _State(stages={"sugar": False, "one_ready": False, "ready": False, "success": False}),
            _State(stages={"sugar": True, "one_ready": False, "ready": False, "success": False}),
            _State(stages={"sugar": True, "one_ready": True, "ready": False, "success": False}),
            _State(stages={"sugar": True, "one_ready": True, "ready": True, "success": False}),
            _State(stages={"sugar": True, "one_ready": True, "ready": True, "success": True}),
        ]
    )
    monkeypatch.setattr(
        subtask_evaluation,
        "evaluate_initialized_runtime",
        lambda runtime: next(states),
    )

    runtime = _Runtime()
    scheduler = subtask_evaluation.WA01OracleSubtaskScheduler(runtime)
    assert scheduler.current_prompt == "add one sugar cube to a coffee cup"

    runtime.env.timestep = 100
    event = scheduler.after_step(runtime, {})
    assert event.changed is True
    assert event.current_subtask == "one_ready"
    assert event.prompt == "carry one coffee cup to the dining counter"

    runtime.env.timestep = 200
    event = scheduler.after_step(runtime, {})
    assert event.current_subtask == "ready"
    assert event.prompt == "carry the remaining coffee cup to the dining counter"

    runtime.env.timestep = 300
    event = scheduler.after_step(runtime, {})
    assert event.current_subtask == "success"
    assert event.prompt == "pick up the coffee cup in the left dining placement zone"

    runtime.env.timestep = 400
    event = scheduler.after_step(runtime, {})
    assert event.complete is True
    assert scheduler.to_metadata()["finished"] is True
