from __future__ import annotations

from dataclasses import dataclass

from bench.tasks.wa05 import subtask_evaluation


@dataclass
class _Stage:
    qualified: bool


class _State:
    def __init__(self, *, filled=False, staged=False, success=False, temperature=None, failure=None):
        self.stages = {
            "cup_filled": _Stage(filled),
            "cup_staged": _Stage(staged),
            "success": _Stage(success),
        }
        self.details = {"filled_temperature": temperature}
        self.failure_code = failure


class _Env:
    timestep = 0

    def object_position(self, cup_id):
        del cup_id
        return [0.0, 0.0, 1.0]


class _Runtime:
    env = _Env()


def test_wa05_oracle_scheduler_switches_and_resolves_hot_button(monkeypatch):
    states = iter(
        [
            _State(),
            _State(),
            _State(),
            _State(),
            _State(filled=True, temperature="hot"),
            _State(filled=True, staged=True, temperature="hot"),
            _State(filled=True, staged=True, success=True, temperature="hot"),
        ]
    )
    monkeypatch.setattr(
        subtask_evaluation,
        "evaluate_initialized_runtime",
        lambda runtime: next(states),
    )
    monkeypatch.setattr(
        subtask_evaluation,
        "_cup_is_held_and_lifted",
        lambda runtime, initial_position: True,
    )

    runtime = _Runtime()
    scheduler = subtask_evaluation.WA05OracleSubtaskScheduler(runtime)
    assert scheduler.current_prompt == "pick up the cup"

    assert not scheduler.after_step(runtime, {}).changed
    assert not scheduler.after_step(runtime, {}).changed
    event = scheduler.after_step(runtime, {})
    assert event.current_subtask == "fill_cup"
    assert event.prompt == "fill the cup with water"

    event = scheduler.after_step(runtime, {})
    assert event.current_subtask == "place_cup"
    assert event.prompt == "place the cup in the service area"

    event = scheduler.after_step(runtime, {})
    assert event.current_subtask == "press_button"
    assert event.prompt == "press the red button for hot water"

    event = scheduler.after_step(runtime, {})
    assert event.complete
    assert scheduler.to_metadata()["finished"] is True


def test_wa05_button_prompt_uses_actual_cold_fill():
    assert (
        subtask_evaluation._button_prompt(_State(temperature="cold"))
        == "press the blue button for cold water"
    )
