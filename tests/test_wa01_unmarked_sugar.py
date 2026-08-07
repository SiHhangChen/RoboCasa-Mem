from __future__ import annotations

from bench.tasks.contracts import EpisodeRequest
from bench.tasks.wa01.domain.episode import resolve_episode_plan
from bench.tasks.wa01.domain.semantics import (
    SemanticState,
    evaluate_success,
    latch_cups_together,
    record_selection,
    record_sugar_delivery,
)
from bench.tasks.wa01.episode import resolve_episode
from bench.tasks.wa01.prompts import resolve_prompt_spec


def test_wa01_prompt_allows_either_unmarked_coffee() -> None:
    prompt = resolve_prompt_spec(variant="memory")

    assert "either coffee" in prompt.initial_prompt
    assert "either coffee" in prompt.policy_prompt
    assert "mark" not in prompt.initial_prompt.casefold()
    assert "mark" not in prompt.policy_prompt.casefold()


def test_wa01_episode_has_no_preassigned_sugar_target() -> None:
    plan = resolve_episode_plan(seed=17)
    episode = resolve_episode(EpisodeRequest(task_id="WA-01", seed=17))

    assert not hasattr(plan, "target_sugared_coffee_id")
    assert "target_sugared_coffee_id" not in episode.facts


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
