"""Canonical WR-03 expert plan built from resolved ExpertPlanInputs."""

from __future__ import annotations

from bench.execution.plan_builder import finish_plan, navigate, operate
from bench.tasks.contracts import ExpertPlanInputs


def build_expert_plan(inputs: ExpertPlanInputs):
    episode = inputs.episode
    if episode.task_id != "WR-03":
        raise ValueError(f"WR-03 cannot build a plan for {episode.task_id!r}")
    level = str(episode.facts["demo_storage_level"])
    actions = inputs.action_config

    def return_arm(target: str, name: str, gripper: float):
        return operate("return_arm", target, name=name, metadata={**dict(actions.return_arm_metadata), "stage": name, "gripper_action": gripper})

    stages = [
        navigate("item_station", purpose="pick", name="navigate_to_visible_can"),
        operate("pick_colored_item", "red_can", name="pick_can_for_storage", metadata=actions.storage_pick_metadata),
        return_arm("red_can", "return_with_can_for_transport", 0.0),
        navigate(f"{level}_storage_station", purpose="store", holding="red_can", name=f"navigate_can_to_{level}_cabinet", metadata={"gripper_action": 0.0}),
        operate("place_in_storage", "red_can", name=f"place_can_in_{level}_cabinet", metadata={**dict(actions.storage_place_metadata), "level": level}),
        operate("verify_storage", "red_can", name=f"verify_can_stored_in_{level}_cabinet", metadata={"level": level}),
        return_arm("red_can", "return_after_can_stored", -1.0),
    ]
    for cabinet_level in ("upper", "lower"):
        stages.extend([
            navigate(f"{cabinet_level}_door_station", purpose="close", name=f"navigate_to_{cabinet_level}_door_for_closing"),
            operate("close_storage_door", cabinet_level, name=f"close_{cabinet_level}_cabinet_after_storage", metadata=actions.door_metadata),
            return_arm(cabinet_level, f"return_after_closing_{cabinet_level}_cabinet", -1.0),
        ])
    stages.extend([
        operate("latch_storage_hidden", "red_can", name="latch_hidden_can_location"),
        navigate(f"{level}_door_station", purpose="retrieve_door", name=f"navigate_to_{level}_door_for_retrieval"),
        operate("open_retrieval_door", level, name="open_remembered_cabinet", metadata=actions.door_metadata),
        operate("verify_retrieval_door", level, name="verify_remembered_cabinet_open"),
        return_arm(level, "return_after_opening_remembered_cabinet", -1.0),
        navigate(f"{level}_storage_station", purpose="retrieve", name=f"navigate_to_{level}_storage_station_for_retrieval"),
        operate("pick_colored_item", "red_can", name="retrieve_same_can_from_cabinet", metadata=actions.retrieval_pick_metadata),
    ])
    return finish_plan(episode, stages, metadata={"demo_storage_level": level})


__all__ = ["build_expert_plan"]
