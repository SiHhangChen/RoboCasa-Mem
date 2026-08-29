from __future__ import annotations

from dataclasses import dataclass

from bench.evaluation.stage_rollout import (
    FAILURE,
    INVALID_PREFIX,
    SUCCESS,
    FunctionalStageTask,
    StageSpec,
    StageEvaluationResult,
    aggregate_stage_results,
    macro_stage_sr,
    evaluate_stage,
)


@dataclass
class _Model:
    calls: list[object]
    reset_calls: int = 0

    def reset_episode(self, instruction: str) -> None:
        assert instruction == "remember the target"
        self.reset_calls += 1

    def predict(self, observation: object) -> list[int]:
        self.calls.append(observation)
        return [len(self.calls)]


class _Expert:
    def predict(self, privileged_state: dict[str, bool]) -> list[int]:
        assert "stage" not in privileged_state
        return [0]


def _task(*, cross_target: bool = False) -> FunctionalStageTask:
    state = {"a": False, "b": False, "c": False}

    def reset(seed: int) -> None:
        assert seed == 7
        state.update(a=False, b=False, c=False)

    def observe() -> dict[str, bool]:
        return dict(state)

    def privileged() -> dict[str, bool]:
        return dict(state)

    def step(action: list[int]) -> dict[str, object]:
        if action == [0]:
            state["a"] = True
            if cross_target:
                state["b"] = True
        else:
            state["b"] = True
        return {"observation": observe(), "terminal": state["b"]}

    return FunctionalStageTask(
        task_id="TEST",
        instruction="remember the target",
        stage_specs=(
            StageSpec("A", lambda value: value["a"]),
            StageSpec("B", lambda value: value["b"], model_step_budget=2),
        ),
        reset_fn=reset,
        observe_fn=observe,
        privileged_state_fn=privileged,
        step_fn=step,
    )


def test_prefix_keeps_model_online_and_handoff_predicts_from_new_observation() -> None:
    model = _Model([])
    result = evaluate_stage(_task(), "B", 7, model, _Expert())

    assert result.status == SUCCESS
    assert result.expert_prefix_steps == 1
    assert result.model_control_steps == 1
    assert result.total_model_inference_calls == 2
    assert result.handoff_observation_id == 1
    assert model.reset_calls == 1
    assert model.calls[0]["a"] is False
    assert model.calls[1]["a"] is True
    assert [item["action_source"] for item in (log.to_dict() for log in result.logs)] == [
        "expert",
        "model",
    ]
    # The model's first proposal was discarded; the second proposal was used.
    assert result.logs[0].proposed_model_action == [1]
    assert result.logs[0].executed_action == [0]
    assert result.logs[1].proposed_model_action == [2]


def test_first_stage_does_not_require_an_expert_controller() -> None:
    state = {"done": False}
    task = FunctionalStageTask(
        task_id="TEST",
        instruction="remember the target",
        stage_specs=(StageSpec("A", lambda value: value["done"], model_step_budget=1),),
        reset_fn=lambda seed: state.update(done=False),
        observe_fn=lambda: dict(state),
        privileged_state_fn=lambda: dict(state),
        step_fn=lambda action: state.update(done=True) or {"observation": dict(state)},
    )
    result = evaluate_stage(task, "A", 7, _Model([]), None)
    assert result.status == SUCCESS


def test_expert_crossing_target_is_invalid_prefix() -> None:
    result = evaluate_stage(_task(cross_target=True), "B", 7, _Model([]), _Expert())
    assert result.status == INVALID_PREFIX
    assert result.success is False
    assert result.failure_reason == "expert_completed_target_during_prefix"


def test_target_budget_counts_only_model_control_steps() -> None:
    state = {"done": False}
    task = FunctionalStageTask(
        task_id="TEST",
        instruction="remember the target",
        stage_specs=(StageSpec("A", lambda value: value["done"], model_step_budget=2),),
        reset_fn=lambda seed: state.update(done=False),
        observe_fn=lambda: dict(state),
        privileged_state_fn=lambda: dict(state),
        step_fn=lambda action: {"observation": dict(state), "terminal": False},
    )
    result = evaluate_stage(task, "A", 7, _Model([]), _Expert())
    assert result.status == FAILURE
    assert result.failure_reason == "model_step_budget"
    assert result.expert_prefix_steps == 0
    assert result.model_control_steps == 2


def test_stage_statistics_exclude_invalid_prefixes() -> None:
    results = [
        StageEvaluationResult("TEST", "B", 1, SUCCESS, True, 1, 1, 1, 2, 2, 1),
        StageEvaluationResult("TEST", "B", 2, FAILURE, False, None, 1, 2, 3, 3, 1),
        StageEvaluationResult("TEST", "B", 3, INVALID_PREFIX, False, None, 0, 0, 1, 1, None),
    ]
    stats = aggregate_stage_results(results)
    assert stats.stage_sr == 0.5
    assert stats.prefix_validity == 2 / 3
    assert stats.wilson_interval() is not None
    assert macro_stage_sr(results) == 0.5


def test_chunk_proposed_before_handoff_is_discarded() -> None:
    class ChunkModel(_Model):
        def predict(self, observation: object) -> list[list[int]]:
            self.calls.append(observation)
            return [[len(self.calls)], [99]]

    model = ChunkModel([])
    result = evaluate_stage(_task(), "B", 7, model, _Expert())
    assert result.status == SUCCESS
    assert result.logs[0].proposed_model_action == [[1], [99]]
    assert result.logs[0].executed_action == [0]
    assert result.logs[1].executed_action == [2]


def test_expert_cannot_complete_multiple_prefix_stages_in_one_step() -> None:
    state = {"a": False, "b": False, "c": False}

    task = FunctionalStageTask(
        task_id="TEST",
        instruction="remember the target",
        stage_specs=(
            StageSpec("A", lambda value: value["a"]),
            StageSpec("B", lambda value: value["b"]),
            StageSpec("C", lambda value: value["c"]),
        ),
        reset_fn=lambda seed: state.update(a=False, b=False, c=False),
        observe_fn=lambda: dict(state),
        privileged_state_fn=lambda: dict(state),
        step_fn=lambda action: (
            state.update(a=True, b=True) or {"observation": dict(state)}
        ),
    )
    result = evaluate_stage(task, "C", 7, _Model([]), _Expert())
    assert result.status == INVALID_PREFIX
    assert result.failure_reason == "expert_skipped_stage_boundary"
