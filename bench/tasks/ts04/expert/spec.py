"""Canonical TS04 expert plan built from resolved ExpertPlanInputs."""

from __future__ import annotations

from bench.execution.plan_builder import finish_plan, navigate, operate
from bench.tasks.contracts import ExpertPlanInputs


def build_expert_plan(inputs: ExpertPlanInputs):
    episode = inputs.episode
    if episode.task_id != "TS04":
        raise ValueError(f"TS04 cannot build a plan for {episode.task_id!r}")
    fixtures = tuple(str(value) for value in episode.facts["demo_fixture_order"])
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
    return finish_plan(episode, stages, metadata={"fixture_order": list(fixtures)})


__all__ = ["build_expert_plan"]
