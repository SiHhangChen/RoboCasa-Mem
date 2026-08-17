"""Canonical TS04 expert plan built from resolved ExpertPlanInputs."""

from __future__ import annotations

from bench.execution.plan_builder import finish_plan, navigate, operate
from bench.tasks.contracts import ExpertPlanInputs


def build_expert_plan(inputs: ExpertPlanInputs):
    episode = inputs.episode
    if episode.task_id != "TS04":
        raise ValueError(f"TS04 cannot build a plan for {episode.task_id!r}")
    full_order = tuple(str(value) for value in episode.facts["demo_fixture_order"])
    target_fixture = str(episode.facts["target_fixture"])
    if target_fixture not in full_order:
        raise ValueError("TS04 target fixture is absent from the search order")
    search_count = full_order.index(target_fixture) + 1
    fixtures = full_order[:search_count]
    actions = inputs.action_config
    stages = []
    for fixture in fixtures:
        stages.extend([
            navigate(f"{fixture}_station", name=f"navigate_{fixture}"),
            operate("open_fixture", fixture, name=f"open_{fixture}", metadata=actions.drawer_open_metadata),
            operate("inspect_fixture", fixture, name=f"inspect_{fixture}", metadata=actions.inspect_metadata),
            operate("close_fixture", fixture, name=f"close_{fixture}", metadata=actions.drawer_close_metadata),
        ])
    stages.extend([
        navigate("completion_button_station", name="navigate_completion_button"),
        operate("press_completion_button", "completion_button", name="press_completion_button", metadata=actions.completion_button_metadata),
    ])
    return finish_plan(
        episode,
        stages,
        metadata={
            "full_fixture_order": list(full_order),
            "fixture_order": list(fixtures),
            "target_fixture": target_fixture,
            "search_count": search_count,
            "required_button_presses": 1,
        },
    )


__all__ = ["build_expert_plan"]
