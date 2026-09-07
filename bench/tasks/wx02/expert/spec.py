"""Canonical WX-02 expert plan built from resolved ExpertPlanInputs."""

from __future__ import annotations

from bench.execution.plan_builder import finish_plan, operate
from bench.tasks.contracts import ExpertPlanInputs
from bench.tasks.wx02.domain.contracts import PRIMARY_CUP, SECONDARY_CUP


def build_expert_plan(inputs: ExpertPlanInputs):
    episode = inputs.episode
    if episode.task_id != "WX-02":
        raise ValueError(f"WX-02 cannot build a plan for {episode.task_id!r}")
    actions = inputs.action_config

    def move_stage(object_name: str, label: str):
        return operate(
            "move_object",
            object_name,
            name=f"move_{label}_cup_to_reveal_stain",
            metadata={
                "pick_metadata": dict(actions.object_pick_metadata),
                "place_metadata": dict(
                    actions.object_temporary_place_metadata
                ),
            },
        )

    def wipe_stage(object_name: str, label: str):
        return operate(
            "pick_tool_and_wipe",
            "cleaning_tool",
            name=f"pick_tool_and_wipe_{label}_stain",
            metadata={
                "pick_metadata": dict(actions.cleaning_tool_pick_metadata),
                "wipe_metadata": dict(
                    actions.wipe_metadata_by_object[object_name]
                ),
            },
        )

    def place_tool_stage(label: str):
        return operate(
            "place_at_accessible",
            "cleaning_tool",
            name=f"place_cleaning_tool_after_{label}_wipe",
            metadata={
                **dict(actions.post_wipe_tool_place_metadata),
                "target_resolver": "post_wipe_tool_pose",
            },
        )

    def pick_cup_stage(object_name: str, label: str):
        return operate(
            "pick",
            object_name,
            name=f"pick_displaced_{label}_cup",
            metadata=actions.object_pick_metadata,
        )

    def return_cup_stage(object_name: str, label: str):
        return operate(
            "place_at_accessible",
            object_name,
            name=f"return_{label}_cup_to_remembered_origin",
            metadata={
                **dict(actions.object_origin_place_metadata),
                "target_resolver": "original_object_pose",
            },
        )

    stages = [
        operate(
            "return_arm",
            name="stand_in_front_of_two_cups",
            metadata=actions.observe_metadata,
        ),
        operate(
            "fetch_cleaning_tool",
            "cleaning_tool",
            name="navigate_to_and_pick_cleaning_tool",
            metadata=actions.cleaning_tool_pick_metadata,
        ),
        operate(
            "return_and_stage_cleaning_tool",
            "cleaning_tool",
            name="return_to_cups_and_stage_cleaning_tool",
            metadata=actions.staged_tool_place_metadata,
        ),
        move_stage(PRIMARY_CUP, "primary"),
        wipe_stage(PRIMARY_CUP, "primary"),
        place_tool_stage("primary"),
        pick_cup_stage(PRIMARY_CUP, "primary"),
        return_cup_stage(PRIMARY_CUP, "primary"),
        move_stage(SECONDARY_CUP, "secondary"),
        wipe_stage(SECONDARY_CUP, "secondary"),
        place_tool_stage("secondary"),
        pick_cup_stage(SECONDARY_CUP, "secondary"),
        return_cup_stage(SECONDARY_CUP, "secondary"),
        operate(
            "return_cleaning_tool_to_region",
            "cleaning_tool",
            name="return_cleaning_tool_to_yellow_region",
            metadata={
                "pick_metadata": dict(actions.cleaning_tool_pick_metadata),
                "place_metadata": dict(
                    actions.cleaning_tool_return_place_metadata
                ),
            },
        ),
    ]
    return finish_plan(
        episode,
        stages,
        metadata={
            "wipe_offsets_xy_by_object": dict(
                episode.facts["wipe_offsets_xy_by_object"]
            ),
            "temporary_target_metadata": dict(
                episode.facts["temporary_target_metadata"]
            ),
        },
    )


__all__ = ["build_expert_plan"]
