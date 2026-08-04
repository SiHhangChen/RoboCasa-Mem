"""Task-local WR-01 expert plan built from immutable episode facts."""

from __future__ import annotations

from bench.execution.contracts import ExpertPlan
from bench.execution.plan_builder import finish_plan, navigate, operate
from bench.tasks.contracts import ExpertPlanInputs


def build_expert_plan(inputs: ExpertPlanInputs) -> ExpertPlan:
    episode = inputs.episode
    if episode.task_id != "WR-01":
        raise ValueError(f"WR-01 cannot build a plan for {episode.task_id!r}")
    facts = dict(episode.facts)
    actions = inputs.action_config
    item_by_level = dict(facts["item_by_level"])
    target = str(facts["target_item_id"])
    target_level = str(facts["target_level"])
    stages: list[object] = [navigate("item_station", purpose="pick", name="navigate_to_visible_items")]

    def ret(item: str, name: str, gripper: float) -> object:
        return operate("return_arm", item, name=name,
                       metadata={**dict(actions.return_arm_metadata), "stage": name, "gripper_action": gripper})

    for index, level in enumerate(("lower", "upper")):
        item = str(item_by_level[level])
        if index:
            stages.append(navigate("item_station", purpose="pick", name=f"navigate_to_{item}_for_{level}_storage"))
        stages.extend([
            operate("pick_colored_item", item, name=f"pick_{item}_for_{level}_storage", metadata=actions.storage_pick_metadata),
            ret(item, f"return_with_{item}_for_{level}_transport", 0.0),
            navigate(f"{level}_storage_station", purpose="store", holding=item, name=f"navigate_{item}_to_{level}_cabinet"),
            operate("place_in_storage", item, name=f"place_{item}_in_{level}_cabinet", metadata={**dict(actions.storage_place_metadata), "level": level}),
            operate("verify_storage", item, name=f"verify_{item}_stored_in_{level}_cabinet", metadata={"level": level}),
            ret(item, f"return_after_{item}_stored_in_{level}_cabinet", -1.0),
        ])
    for level in ("upper", "lower"):
        stages.extend([
            navigate(f"{level}_door_station", purpose="close", name=f"navigate_to_{level}_door_for_closing"),
            operate("close_storage_door", level, name=f"close_{level}_cabinet_after_storage", metadata=actions.door_metadata),
            ret(level, f"return_after_closing_{level}_cabinet", -1.0),
        ])
    stages.extend([
        operate("latch_storage_hidden", "item_relation", name="latch_hidden_item_to_cabinet_relation"),
        navigate(f"{target_level}_door_station", purpose="retrieve_door", name=f"navigate_to_{target_level}_door_for_retrieval"),
        operate("open_retrieval_door", target_level, name="open_remembered_target_cabinet", metadata=actions.door_metadata),
        operate("verify_retrieval_door", target_level, name="verify_remembered_target_cabinet_open"),
        ret(target_level, "return_after_opening_remembered_cabinet", -1.0),
        navigate(f"{target_level}_storage_station", purpose="retrieve", name=f"navigate_to_{target_level}_storage_station_for_retrieval"),
        operate("pick_colored_item", target, name="pick_requested_colored_item", metadata=actions.retrieval_pick_metadata),
    ])
    return finish_plan(episode, stages, metadata={"item_by_level": item_by_level, "target_item_id": target, "target_level": target_level})


__all__ = ["build_expert_plan"]
