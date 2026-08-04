"""Canonical WX-01 expert plan built from resolved ExpertPlanInputs."""

from __future__ import annotations

from bench.execution.plan_builder import finish_plan, navigate, operate
from bench.tasks.contracts import ExpertPlanInputs


def build_expert_plan(inputs: ExpertPlanInputs):
    episode = inputs.episode
    if episode.task_id != "WX-01":
        raise ValueError(f"WX-01 cannot build a plan for {episode.task_id!r}")
    facts = dict(episode.facts)
    actions = inputs.action_config
    wipe = {
        **dict(actions.wipe_metadata),
        "target_pose_attr": "original_blocking_object_pose",
        "wipe_offsets_xy": list(facts["wipe_offsets_xy"]),
        "wipe_sampling": dict(facts["wipe_metadata"]),
    }
    stages = [
        navigate("dining_table", holding="cleaning_tool", name="navigate_to_wipe_area_holding_sponge"),
        operate("wipe", "cleaning_tool", name="wipe_remembered_blocking_origin", metadata=wipe),
        operate("place_at_accessible", "cleaning_tool", name="place_cleaning_tool_after_wipe", metadata={**dict(actions.cleaning_tool_place_metadata), "target_name": "post_wipe_dining_sponge", "target_resolver": "post_wipe_tool_pose"}),
        operate("return_arm", "cleaning_tool", name="return_after_post_wipe_sponge_place", metadata={**dict(actions.return_arm_metadata), "stage": "return_after_post_wipe_sponge_place"}),
        operate("pick", "blocking_object", name="pick_displaced_blocking_object", metadata=actions.blocking_pick_metadata),
        operate("place_at_accessible", "blocking_object", name="return_blocking_object_to_remembered_origin", metadata={**dict(actions.blocking_origin_place_metadata), "target_name": "blocking_origin", "target_resolver": "original_blocking_pose"}),
    ]
    return finish_plan(episode, stages, metadata={"wipe_offsets_xy": list(facts["wipe_offsets_xy"]), "temporary_target_metadata": dict(facts["temporary_target_metadata"])})


__all__ = ["build_expert_plan"]
