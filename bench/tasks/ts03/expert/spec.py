"""Canonical TS03 expert plan built from resolved ExpertPlanInputs."""

from __future__ import annotations

from bench.execution.plan_builder import finish_plan, navigate, operate
from bench.tasks.contracts import ExpertPlanInputs


def build_expert_plan(inputs: ExpertPlanInputs):
    episode = inputs.episode
    if episode.task_id != "TS03":
        raise ValueError(f"TS03 cannot build a plan for {episode.task_id!r}")
    facts = dict(episode.facts)
    actions = inputs.action_config
    stages = []
    for index in range(int(facts["target_sugar_count"])):
        sugar_names = tuple(str(value) for value in facts.get("target_sugar_names", ()))
        sugar = sugar_names[index] if index < len(sugar_names) else f"sugar{index}"
        metadata = dict(actions.sugar_delivery_metadata_by_index[index + 1])
        stages.extend([
            navigate("sugar_station", purpose="pick", name=f"navigate_{sugar}"),
            operate("pick_sugar_limited_force", sugar, name=f"pick_{sugar}", metadata=actions.sugar_pick_metadata),
            operate(
                "return_arm",
                sugar,
                name=f"retract_{sugar}_after_pick",
                metadata={
                    **dict(actions.return_arm_metadata),
                    "stage": f"retract_{sugar}_after_pick",
                    "gripper_action": 1.0,
                },
            ),
            navigate(
                "saucepan_station",
                purpose="place",
                holding=sugar,
                name=f"navigate_{sugar}_to_saucepan",
            ),
            operate(
                "place_sugar_in_live_saucepan",
                sugar,
                name=f"place_{sugar}_saucepan",
                metadata=metadata,
            ),
            operate(
                "return_arm",
                sugar,
                name=f"return_after_{sugar}_place",
                metadata={
                    **dict(actions.return_arm_metadata),
                    "stage": f"return_after_{sugar}_place",
                    "gripper_action": -1.0,
                },
            ),
        ])
    stages.extend([
        navigate(
            "completion_button_station",
            purpose="finish",
            name="navigate_completion_button",
        ),
        operate("press_completion_button", "completion_button", name="press_completion_button", metadata=actions.completion_button_metadata),
    ])
    return finish_plan(episode, stages, metadata={"target_sugar_count": facts["target_sugar_count"], "total_sugar_count": facts["total_sugar_count"]})


__all__ = ["build_expert_plan"]
