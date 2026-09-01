"""Task-local WA-01 expert plan.

The plan is assembled from the immutable episode facts and task-owned
operation metadata.  No simulator context or task-specific workflow is consulted.
"""

from __future__ import annotations

from bench.execution.contracts import ExpertPlan
from bench.execution.plan_builder import finish_plan, navigate, operate
from bench.tasks.contracts import ExpertPlanInputs
from bench.tasks.wa01.domain.episode import target_sugared_coffee_for_seed
from bench.tasks.wa01.domain.geometry import dining_region_config_summary


def build_expert_plan(inputs: ExpertPlanInputs) -> ExpertPlan:
    episode = inputs.episode
    if episode.task_id != "WA-01":
        raise ValueError(f"WA-01 cannot build a plan for {episode.task_id!r}")
    facts = dict(episode.facts)
    actions = inputs.action_config
    first, second = (str(v) for v in facts["delivery_order"])
    sugar_target_mode = str(facts.get("sugar_target_mode", "unmarked"))
    if sugar_target_mode == "marked":
        sugared = str(facts["target_sugared_coffee_id"])
    else:
        # Select the sugared cup independently from delivery order. The
        # seed-owned choice keeps demonstrations reproducible while allowing
        # both sugared-first and sugared-second episodes.
        sugared = target_sugared_coffee_for_seed(episode.seed)
    slots = dict(facts["final_slot_assignment"])
    sugar = str(getattr(inputs.scene, "sugar_name", "sugar_cube"))

    def return_arm(target_name: str, stage_name: str, gripper: float) -> object:
        return operate("return_arm", target_name, name=stage_name,
                        metadata={**dict(actions.return_arm_metadata), "stage": stage_name,
                                  "gripper_action": gripper})

    def deliver(coffee: str, ordinal: str) -> list[object]:
        result: list[object] = []
        if ordinal == "second":
            result.append(navigate(coffee, purpose="pick", name=f"navigate_pick_second_delivery_{coffee}",
                                   skip_if_reachable=True, fallback_if_unreachable=True))
        result.extend([
            operate("verify_pick_reachability", coffee, name=f"verify_{ordinal}_pick_reachability_{coffee}"),
            operate("pick", coffee, name=f"pick_{ordinal}_delivery_{coffee}",
                    metadata=actions.transport_pick_metadata_by_mug[coffee]),
        ])
        if ordinal == "second":
            result.append(operate("lift_held", coffee, name=f"raise_second_delivery_{coffee}_for_transport",
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
        result.append(return_arm(
            coffee,
            f"return_after_place_{ordinal}_{coffee}_dining_to_regular_pose",
            -1.0,
        ))
        if ordinal == "second":
            result.append(operate("verify_cups_ready", "coffee_pair", name="verify_cups_ready_for_selection"))
        return result

    stages: list[object] = [
        navigate(sugar, purpose="pick", name="navigate_pick_sugar", skip_if_reachable=True,
                 fallback_if_unreachable=True),
        operate("pick_sugar_limited_force", sugar, name="pick_sugar",
                metadata=actions.sugar_pick_metadata),
        return_arm(sugar, "return_picked_sugar_to_regular_pose", 1.0),
        navigate(sugared, purpose="place", holding=sugar, name="navigate_place_sugar",
                 skip_if_reachable=True, fallback_if_unreachable=True,
                 metadata=actions.sugar_delivery_metadata),
        operate("place_sugar_restore_force", sugar, name="place_sugar",
                metadata={**dict(actions.sugar_delivery_metadata), "receptacle": sugared}),
        operate("record_sugar_delivery", sugar, name="record_sugar_delivery",
                metadata={"coffee_id": sugared}),
        return_arm(sugar, "return_after_place_sugar_to_regular_pose", -1.0),
        *deliver(first, "first"),
        *deliver(second, "second"),
        operate("pick", sugared, name="pick_sugared_coffee_topdown",
                metadata=actions.final_selection_pick_metadata),
        operate("verify_transport_hold", sugared, name="verify_sugared_coffee_held_after_topdown_pick"),
        operate("select", sugared, name="select_sugared_coffee"),
    ]
    return finish_plan(episode, stages, metadata={
        "sugar_target_mode": sugar_target_mode,
        "target_sugared_coffee_id": facts.get("target_sugared_coffee_id"),
        "sugared_coffee_id": sugared,
        "delivery_order": [first, second],
        "final_slot_assignment": slots,
        "dining_region_config": dining_region_config_summary(actions.dining_region_config),
        "dining_regions_by_slot": dict(actions.dining_regions_by_slot),
        "sampled_dining_targets_by_slot": dict(actions.sampled_dining_targets_by_slot),
    })


__all__ = ["build_expert_plan"]
