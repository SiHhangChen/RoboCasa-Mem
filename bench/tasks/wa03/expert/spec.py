"""Task-local WA-03 expert plan.

The plan is assembled from the immutable episode facts and task-owned
operation metadata.  No simulator context or task-specific workflow is consulted.
"""

from __future__ import annotations

from bench.execution.contracts import ExpertPlan
from bench.execution.plan_builder import finish_plan, navigate, operate
from bench.tasks.contracts import ExpertPlanInputs
from bench.tasks.wa03.domain.geometry import dining_region_config_summary


def build_expert_plan(inputs: ExpertPlanInputs) -> ExpertPlan:
    episode = inputs.episode
    if episode.task_id != "WA-03":
        raise ValueError(f"WA-03 cannot build a plan for {episode.task_id!r}")
    facts = dict(episode.facts)
    actions = inputs.action_config
    target = str(facts["target_sugared_coffee_id"])
    delivery_order = tuple(str(value) for value in facts["delivery_order"])
    if len(delivery_order) != 3:
        raise ValueError("WA-03 expert plan requires exactly three coffee deliveries")
    slots = dict(facts["final_slot_assignment"])
    sugar = str(getattr(inputs.scene, "sugar_name", "sugar_cube"))

    def return_arm(target_name: str, stage_name: str, gripper: float) -> object:
        return operate("return_arm", target_name, name=stage_name,
                        metadata={**dict(actions.return_arm_metadata), "stage": stage_name,
                                  "gripper_action": gripper})

    def deliver(coffee: str, ordinal: str, *, index: int, is_last: bool) -> list[object]:
        result: list[object] = []
        if index > 0:
            result.append(navigate(coffee, purpose="pick", name=f"navigate_pick_{ordinal}_delivery_{coffee}",
                                   skip_if_reachable=True, fallback_if_unreachable=True))
        result.extend([
            operate("verify_pick_reachability", coffee, name=f"verify_{ordinal}_pick_reachability_{coffee}"),
            operate("pick", coffee, name=f"pick_{ordinal}_delivery_{coffee}",
                    metadata=actions.transport_pick_metadata_by_mug[coffee]),
        ])
        if index > 0:
            result.append(operate("lift_held", coffee, name=f"raise_{ordinal}_delivery_{coffee}_for_transport",
                                  metadata=actions.transport_lift_metadata_by_mug[coffee]))
        result.extend([
            return_arm(coffee, f"return_{ordinal}_delivery_{coffee}_to_regular_pose_for_transport", 1.0),
            operate("verify_transport_hold", coffee, name=f"verify_{ordinal}_{coffee}_held_for_transport"),
            navigate("dining_delivery", purpose="place", holding=coffee,
                     name=f"navigate_deliver_{ordinal}_{coffee}_dining"),
            operate("place_dynamic", coffee, name=f"place_{ordinal}_{coffee}_dining",
                    metadata=actions.dining_place_metadata_by_mug[coffee]),
            operate("verify_dining_placement", coffee, name=f"verify_{ordinal}_{coffee}_on_dining",
                    metadata={"dining_target_slot": slots[coffee]}),
        ])
        if not is_last:
            result.append(return_arm(coffee, f"return_after_place_{ordinal}_{coffee}_dining_to_regular_pose", -1.0))
        else:
            result.append(operate(
                "verify_all_cups_ready",
                "coffee_set",
                name="verify_all_cups_ready_for_selection",
            ))
        return result

    ordinals = ("first", "second", "third")
    delivery_stages = [
        stage
        for index, coffee in enumerate(delivery_order)
        for stage in deliver(
            coffee,
            ordinals[index],
            index=index,
            is_last=index == len(delivery_order) - 1,
        )
    ]

    stages: list[object] = [
        navigate(sugar, purpose="pick", name="navigate_pick_sugar", skip_if_reachable=True,
                 fallback_if_unreachable=True),
        operate("pick", sugar, name="pick_sugar", metadata=actions.sugar_pick_metadata),
        return_arm(sugar, "return_picked_sugar_to_regular_pose", 1.0),
        navigate(target, purpose="place", holding=sugar, name="navigate_place_sugar",
                 skip_if_reachable=True, fallback_if_unreachable=True,
                 metadata=actions.sugar_delivery_metadata),
        operate("place", sugar, name="place_sugar",
                metadata={**dict(actions.sugar_delivery_metadata), "receptacle": target}),
        operate("record_sugar_delivery", sugar, name="record_sugar_delivery"),
        return_arm(sugar, "return_after_place_sugar_to_regular_pose", -1.0),
        *delivery_stages,
        operate("pick", target, name="pick_sugared_coffee_topdown",
                metadata=actions.final_selection_pick_metadata),
        operate("verify_transport_hold", target, name="verify_sugared_coffee_held_after_topdown_pick"),
        operate("select", target, name="select_sugared_coffee"),
    ]
    return finish_plan(episode, stages, metadata={
        "target_sugared_coffee_id": target,
        "delivery_order": list(delivery_order),
        "final_slot_assignment": slots,
        "dining_region_config": dining_region_config_summary(actions.dining_region_config),
        "dining_regions_by_slot": dict(actions.dining_regions_by_slot),
        "sampled_dining_targets_by_slot": dict(actions.sampled_dining_targets_by_slot),
    })


__all__ = ["build_expert_plan"]
