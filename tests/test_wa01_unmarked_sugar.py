from __future__ import annotations

import pytest

from bench.execution.contracts import OperateStage
from bench.tasks import resolve_builtin_task_plugin
from bench.tasks.contracts import EpisodeRequest, ExpertPlanInputs
from bench.tasks.wa01.execution.handlers import build_stage_handlers
from bench.tasks.wa01.domain.episode import (
    resolve_episode_plan,
    target_sugared_coffee_for_seed,
)
from bench.tasks.wa01.domain.semantics import (
    SemanticState,
    evaluate_success,
    latch_cups_together,
    record_selection,
    record_sugar_delivery,
)
from bench.tasks.wa01.episode import resolve_episode
from bench.tasks.wa01.expert.spec import build_expert_plan
from bench.tasks.wa01.prompts import resolve_prompt_spec


def test_wa01_limits_gripper_force_only_during_sugar_transfer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeController:
        env = object()

        def pick(self, target: str, **kwargs: object) -> dict[str, object]:
            assert target == "sugar_cube"
            return {"pass": True, "pick_kwargs": kwargs}

        def place(
            self,
            target: str,
            receptacle: str,
            **kwargs: object,
        ) -> dict[str, object]:
            assert target == "sugar_cube"
            assert receptacle == "coffee_B"
            return {"pass": True, "place_kwargs": kwargs}

    applied_limits: list[float] = []

    def configure(_env: object, force_limit: float) -> dict[str, float]:
        applied_limits.append(float(force_limit))
        return {"force_limit": float(force_limit)}

    monkeypatch.setattr(
        "bench.tasks.wa01.execution.handlers._configure_right_gripper_force_limit",
        configure,
    )
    controller = FakeController()
    handlers = build_stage_handlers(
        controller,
        sugar_gripper_force_limit=5.0,
        default_gripper_force_limit=20.0,
    )

    pick = handlers["pick_sugar_limited_force"]("sugar_cube")
    assert pick["pass"]
    assert applied_limits == [5.0]

    place = handlers["place_sugar_restore_force"](
        "sugar_cube",
        receptacle="coffee_B",
    )
    assert place["pass"]
    assert applied_limits == [5.0, 20.0]


def test_wa01_prompt_allows_either_unmarked_coffee() -> None:
    prompt = resolve_prompt_spec(variant="memory")

    assert "either coffee" in prompt.initial_prompt
    assert "either coffee" in prompt.policy_prompt
    assert "mark" not in prompt.initial_prompt.casefold()
    assert "mark" not in prompt.policy_prompt.casefold()


def test_wa01_episode_has_no_preassigned_sugar_target() -> None:
    plan = resolve_episode_plan(seed=17)
    episode = resolve_episode(EpisodeRequest(task_id="WA-01", seed=17))

    assert plan.sugar_target_mode == "unmarked"
    assert plan.target_sugared_coffee_id is None
    assert episode.facts["sugar_target_mode"] == "unmarked"
    assert episode.facts["target_sugared_coffee_id"] is None


def test_wa01_expert_sugar_choice_is_independent_of_delivery_order() -> None:
    observed_positions: set[str] = set()
    for seed in range(16):
        episode = resolve_episode(EpisodeRequest(task_id="WA-01", seed=seed))
        descriptor = resolve_builtin_task_plugin("WA-01").load_runtime_api().build_descriptor(
            episode,
            False,
        )
        expert = build_expert_plan(ExpertPlanInputs.from_descriptor(descriptor))
        sugared = str(expert.metadata["sugared_coffee_id"])
        delivery_order = tuple(str(value) for value in episode.facts["delivery_order"])

        assert sugared == target_sugared_coffee_for_seed(seed)
        observed_positions.add("first" if sugared == delivery_order[0] else "second")

    assert observed_positions == {"first", "second"}


def test_wa01_returns_arm_after_second_delivery_before_final_pick() -> None:
    episode = resolve_episode(EpisodeRequest(task_id="WA-01", seed=0))
    descriptor = resolve_builtin_task_plugin("WA-01").load_runtime_api().build_descriptor(
        episode,
        False,
    )
    expert = build_expert_plan(ExpertPlanInputs.from_descriptor(descriptor))
    second = str(episode.facts["delivery_order"][1])
    return_name = f"return_after_place_second_{second}_dining_to_regular_pose"
    return_index = next(
        index for index, stage in enumerate(expert.stages) if stage.name == return_name
    )
    return_stage = expert.stages[return_index]

    assert isinstance(return_stage, OperateStage)
    assert return_stage.operation == "return_arm"
    assert return_stage.target == second
    assert return_stage.metadata["gripper_action"] == -1.0
    assert expert.stages[return_index + 1].name == "verify_cups_ready_for_selection"
    assert expert.stages[return_index + 2].name == "pick_sugared_coffee_topdown"


def test_wa01_accepts_either_delivery_and_requires_selecting_that_coffee() -> None:
    for coffee_id, other_id in (
        ("coffee_A", "coffee_B"),
        ("coffee_B", "coffee_A"),
    ):
        state = SemanticState()
        assert record_sugar_delivery(state, coffee_id)
        assert state.sugared_coffee_id == coffee_id
        assert state.unsugared_coffee_id == other_id

        latch_cups_together(state, currently_together=True)
        record_selection(state, other_id)
        assert not evaluate_success(state).success

        record_selection(state, coffee_id)
        report = evaluate_success(state)
        assert report.success
        assert report.checks() == {
            "sugar_consumed": True,
            "cups_together_before_selection": True,
            "correct_sugared_selection": True,
        }


def test_wa01_marked_compatibility_restores_old_prompt_and_target() -> None:
    seed = 17
    episode = resolve_episode(
        EpisodeRequest(
            task_id="WA-01",
            seed=seed,
            options={"sugar_target_mode": "marked"},
        )
    )
    descriptor = resolve_builtin_task_plugin("WA-01").load_runtime_api().build_descriptor(
        episode,
        False,
    )

    target = target_sugared_coffee_for_seed(seed)
    assert episode.facts["sugar_target_mode"] == "marked"
    assert episode.facts["target_sugared_coffee_id"] == target
    assert "temporarily marked" in episode.public["prompt"]
    assert descriptor.environment.env_kwargs == {
        "sugar_target_mode": "marked",
        "target_sugared_coffee_id": target,
    }


def test_wa01_marked_compatibility_requires_the_marked_coffee() -> None:
    incorrect = SemanticState(
        sugar_target_mode="marked",
        target_sugared_coffee_id="coffee_A",
    )
    record_sugar_delivery(incorrect, "coffee_B")
    latch_cups_together(incorrect, currently_together=True)
    record_selection(incorrect, "coffee_B")

    incorrect_report = evaluate_success(incorrect)
    assert not incorrect_report.success
    assert incorrect_report.checks()["sugar_assignment_correct"] is False

    correct = SemanticState(
        sugar_target_mode="marked",
        target_sugared_coffee_id="coffee_A",
    )
    record_sugar_delivery(correct, "coffee_A")
    latch_cups_together(correct, currently_together=True)
    record_selection(correct, "coffee_A")

    correct_report = evaluate_success(correct)
    assert correct_report.success
    assert correct_report.checks()["sugar_assignment_correct"] is True
