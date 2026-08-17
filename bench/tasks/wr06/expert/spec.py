"""Canonical WR-06 expert plan built from resolved ExpertPlanInputs."""

from __future__ import annotations

from bench.execution.plan_builder import finish_plan, navigate, operate
from bench.tasks.contracts import ExpertPlanInputs
from bench.tasks.wr06.domain.episode import validate_plate_stage_region


def _plate_stage_station_for_region(region: str) -> str:
    station_by_region = {
        "plate_left": "plate_left_station",
        # The calibrated right station was validated with mustard; a ketchup
        # shelf grasp preserves a different EEF-to-bottle rotation and needs
        # the mirrored left-slot base geometry (WR-05-validated).
        "plate_right": "ketchup_plate_right_station",
    }
    try:
        return station_by_region[validate_plate_stage_region(region)]
    except KeyError as exc:
        raise KeyError(f"unknown WR-06 plate stage region {region!r}") from exc


def _plate_pick_station_for_region(region: str) -> str:
    station_by_region = {
        "plate_left": "plate_left_pick_station",
        "plate_right": "plate_right_pick_station",
    }
    try:
        return station_by_region[validate_plate_stage_region(region)]
    except KeyError as exc:
        raise KeyError(f"unknown WR-06 plate region {region!r}") from exc


def build_expert_plan(inputs: ExpertPlanInputs):
    episode = inputs.episode
    if episode.task_id != "WR-06":
        raise ValueError(f"WR-06 cannot build a plan for {episode.task_id!r}")
    level = str(episode.facts["demo_original_level"])
    plate_side = str(episode.facts["plate_stage_region"])
    actions = inputs.action_config
    shelf_pick = dict(actions.shelf_pick_metadata)
    plate_pick = dict(actions.plate_pick_metadata)
    plate_place = {**dict(actions.plate_place_metadata), "target_name": f"ketchup_{plate_side}", "target_resolver": "plate_stage_pose"}
    home_place = {**dict(actions.home_place_metadata), "return_level": level, "target_name": "ketchup_return_shelf_region", "target_resolver": "home_return_pose"}
    stages = [
        navigate("shelf_lower", purpose="shelf_observe", name="observe_shelf_levels", metadata={"observe": "lower_and_upper_shelf_levels", "use_manual_path": False}),
        navigate(level, purpose="shelf_pick", name="navigate_to_original_level", metadata={"use_manual_path": False}),
        operate("pick", "ketchup", name="pick_item_from_original_level", metadata=shelf_pick),
        operate("return_arm", "ketchup", name="stow_item_after_shelf_pick", metadata={**dict(actions.return_arm_holding), "stage": "stow_item_after_shelf_pick"}),
        operate("set_torso_height", "ketchup", name="normalize_transport_posture", metadata=actions.normalize_torso_metadata),
        navigate(_plate_stage_station_for_region(plate_side), purpose="plate_stage", holding="ketchup", name="navigate_item_to_plate_stage", metadata={"use_manual_path": False}),
        operate("place_at_accessible", "ketchup", name=f"place_item_at_{plate_side}", metadata=plate_place),
        operate("return_arm", "ketchup", name="return_arm_after_plate_stage", metadata={**dict(actions.return_arm_empty), "stage": "return_after_plate_stage"}),
        operate("latch_original_level_hidden", "ketchup", name="latch_original_level_memory_barrier"),
        navigate(_plate_pick_station_for_region(plate_side), purpose="plate_pick", name="navigate_to_staged_item", metadata={"use_manual_path": False}),
        operate("pick", "ketchup", name="pick_item_from_plate_stage", metadata=plate_pick),
        operate("return_arm", "ketchup", name="stow_item_for_return", metadata={**dict(actions.return_arm_holding), "stage": "stow_item_for_return"}),
        navigate(level, purpose="home_return", holding="ketchup", name="navigate_to_remembered_level", metadata={**dict(actions.return_navigation), "use_manual_path": False}),
        operate("return_item_to_level", "ketchup", name="return_item_to_original_level", metadata=home_place),
        operate("return_arm", "ketchup", name="return_arm_after_home_place", metadata={**dict(actions.return_arm_empty), "stage": "return_after_home_place"}),
    ]
    return finish_plan(episode, stages, metadata={"demo_original_level": level, "plate_stage_region": plate_side})


__all__ = ["build_expert_plan"]
