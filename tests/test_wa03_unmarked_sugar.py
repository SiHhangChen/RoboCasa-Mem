from __future__ import annotations

from bench.tasks import resolve_builtin_task_plugin
from bench.tasks.contracts import EpisodeRequest, ExpertPlanInputs
from bench.tasks.wa03.domain.episode import (
    resolve_episode_plan,
    target_sugared_coffee_for_seed,
)
from bench.tasks.wa03.domain.semantics import (
    SemanticState,
    evaluate_success,
    latch_all_cups_ready,
    record_selection,
    record_sugar_delivery,
)
from bench.tasks.wa03.episode import resolve_episode
from bench.tasks.wa03.expert.spec import build_expert_plan
from bench.tasks.wa03.prompts import resolve_prompt_spec


def test_wa03_prompt_allows_any_unmarked_coffee() -> None:
    prompt = resolve_prompt_spec(variant="memory")

    assert "any one coffee" in prompt.initial_prompt
    assert "any one coffee" in prompt.policy_prompt
    assert "mark" not in prompt.initial_prompt.casefold()
    assert "mark" not in prompt.policy_prompt.casefold()


def test_wa03_episode_has_no_preassigned_sugar_target() -> None:
    plan = resolve_episode_plan(seed=17)
    episode = resolve_episode(EpisodeRequest(task_id="WA-03", seed=17))

    assert plan.sugar_target_mode == "unmarked"
    assert plan.target_sugared_coffee_id is None
    assert episode.facts["sugar_target_mode"] == "unmarked"
    assert episode.facts["target_sugared_coffee_id"] is None
    assert "mark" not in episode.public["prompt"].casefold()


def test_wa03_expert_deterministically_sugars_first_delivery() -> None:
    episode = resolve_episode(EpisodeRequest(task_id="WA-03", seed=17))
    api = resolve_builtin_task_plugin("WA-03").load_runtime_api()
    descriptor = api.build_descriptor(episode, False)
    expert = build_expert_plan(ExpertPlanInputs.from_descriptor(descriptor))

    assert expert.metadata["sugar_target_mode"] == "unmarked"
    assert expert.metadata["target_sugared_coffee_id"] is None
    assert expert.metadata["sugared_coffee_id"] == episode.facts["delivery_order"][0]


def test_wa03_accepts_any_delivery_and_requires_selecting_that_coffee() -> None:
    for coffee_id in ("coffee_A", "coffee_B", "coffee_C"):
        state = SemanticState()
        assert record_sugar_delivery(state, coffee_id)
        assert state.sugared_coffee_id == coffee_id
        assert set(state.unsugared_coffee_ids) == {
            value for value in ("coffee_A", "coffee_B", "coffee_C") if value != coffee_id
        }

        other_id = next(value for value in state.unsugared_coffee_ids)
        latch_all_cups_ready(state, currently_ready=True)
        record_selection(state, other_id)
        assert not evaluate_success(state).success

        record_selection(state, coffee_id)
        report = evaluate_success(state)
        assert report.success
        assert report.checks() == {
            "sugar_consumed": True,
            "all_cups_ready_before_selection": True,
            "correct_sugared_selection": True,
        }


def test_wa03_marked_compatibility_restores_old_prompt_and_target() -> None:
    seed = 17
    episode = resolve_episode(
        EpisodeRequest(
            task_id="WA-03",
            seed=seed,
            options={"sugar_target_mode": "marked"},
        )
    )
    descriptor = resolve_builtin_task_plugin("WA-03").load_runtime_api().build_descriptor(
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


def test_wa03_marked_compatibility_requires_the_marked_coffee() -> None:
    incorrect = SemanticState(
        sugar_target_mode="marked",
        target_sugared_coffee_id="coffee_A",
    )
    record_sugar_delivery(incorrect, "coffee_B")
    latch_all_cups_ready(incorrect, currently_ready=True)
    record_selection(incorrect, "coffee_B")

    incorrect_report = evaluate_success(incorrect)
    assert not incorrect_report.success
    assert incorrect_report.checks()["sugar_assignment_correct"] is False

    correct = SemanticState(
        sugar_target_mode="marked",
        target_sugared_coffee_id="coffee_A",
    )
    record_sugar_delivery(correct, "coffee_A")
    latch_all_cups_ready(correct, currently_ready=True)
    record_selection(correct, "coffee_A")

    correct_report = evaluate_success(correct)
    assert correct_report.success
    assert correct_report.checks()["sugar_assignment_correct"] is True
