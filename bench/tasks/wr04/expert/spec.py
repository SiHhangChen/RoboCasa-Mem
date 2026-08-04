"""Canonical WR-04 expert plan built from resolved ExpertPlanInputs."""

from __future__ import annotations

from bench.execution.plan_builder import finish_plan, navigate, operate
from bench.tasks.contracts import ExpertPlanInputs


def build_expert_plan(inputs: ExpertPlanInputs):
    episode = inputs.episode
    if episode.task_id != "WR-04":
        raise ValueError(f"WR-04 cannot build a plan for {episode.task_id!r}")
    level = str(episode.facts["demo_original_level"])
    actions = inputs.action_config
    shelf_pick = dict(actions.shelf_pick_metadata)
    sink_pick = dict(actions.sink_pick_metadata)
    sink_place = {**dict(actions.sink_place_metadata), "target_name": "ketchup_sink_left", "target_resolver": "sink_stage_pose"}
    home_place = {**dict(actions.home_place_metadata), "return_level": level, "target_name": "ketchup_return_shelf_region", "target_resolver": "home_return_pose"}
    stages = [
        navigate("shelf_lower", purpose="shelf_observe", name="observe_shelf_levels", metadata={"observe": "lower_and_upper_shelf_levels", "use_manual_path": False}),
        navigate(level, purpose="shelf_pick", name="navigate_to_original_level", metadata={"use_manual_path": False}),
        operate("pick", "ketchup", name="pick_item_from_original_level", metadata=shelf_pick),
        operate("return_arm", "ketchup", name="stow_item_after_shelf_pick", metadata={**dict(actions.return_arm_holding), "stage": "stow_item_after_shelf_pick"}),
        operate("set_torso_height", "ketchup", name="normalize_transport_posture", metadata=actions.normalize_torso_metadata),
        navigate("sink_place_station", purpose="sink_stage", holding="ketchup", name="navigate_item_to_sink_stage", metadata={"use_manual_path": False}),
        operate("place_at_accessible", "ketchup", name="place_item_at_fixed_sink_left", metadata=sink_place),
        operate("return_arm", "ketchup", name="return_arm_after_sink_stage", metadata={**dict(actions.return_arm_empty), "stage": "return_after_sink_stage"}),
        operate("latch_original_level_hidden", "ketchup", name="latch_original_level_memory_barrier"),
        navigate("sink_place_station", purpose="sink_pick", name="navigate_to_staged_item", metadata={"use_manual_path": False}),
        operate("pick", "ketchup", name="pick_item_from_sink_stage", metadata=sink_pick),
        operate("return_arm", "ketchup", name="stow_item_for_return", metadata={**dict(actions.return_arm_holding), "stage": "stow_item_for_return"}),
        navigate(level, purpose="home_return", holding="ketchup", name="navigate_to_remembered_level", metadata={**dict(actions.return_navigation), "use_manual_path": False}),
        operate("return_item_to_level", "ketchup", name="return_item_to_original_level", metadata=home_place),
        operate("return_arm", "ketchup", name="return_arm_after_home_place", metadata={**dict(actions.return_arm_empty), "stage": "return_after_home_place"}),
    ]
    return finish_plan(episode, stages, metadata={"demo_original_level": level})


__all__ = ["build_expert_plan"]
