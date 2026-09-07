"""Canonical WR-08 expert plan for two color-matched carton transports."""

from __future__ import annotations

from bench.execution.plan_builder import finish_plan, navigate, operate
from bench.tasks.contracts import ExpertPlanInputs
from bench.tasks.wr08.domain.contracts import CARTON_ASSET_BY_ITEM, ITEM_IDS, TASK_REVISION


def build_expert_plan(inputs: ExpertPlanInputs):
    episode = inputs.episode
    if episode.task_id != "WR-08":
        raise ValueError(f"WR-08 cannot build a plan for {episode.task_id!r}")
    facts = dict(episode.facts)
    source_slots = dict(facts["source_slot_by_item"])
    source_colors = dict(facts["source_color_by_item"])
    target_by_color = dict(facts["target_slot_by_color"])
    order = tuple(facts["transport_order"])
    actions = inputs.action_config
    stages: list[object] = [
        navigate(
            "source_plates",
            purpose="source_observe",
            name="observe_source_sugar_plates",
            metadata={"observe": "three_colored_source_sugar_plates"},
        )
    ]
    for item in order:
        source_slot = str(source_slots[item])
        source_color = str(source_colors[item])
        target_slot = str(target_by_color[source_color])
        stages.extend(
            [
                navigate(
                    source_slot,
                    purpose="source_pick",
                    name=f"navigate_{item}_source_plate",
                    metadata={"xy_tolerance": 0.08, "yaw_tolerance": 0.12},
                ),
                operate(
                    "pick",
                    item,
                    name=f"pick_{item}_from_source_plate",
                    metadata=actions.source_pick_metadata[item],
                ),
                operate(
                    "return_arm",
                    item,
                    name=f"stow_{item}_after_source_pick",
                    metadata={**dict(actions.return_arm_holding), "stage": f"stow_{item}_after_source_pick"},
                ),
                operate(
                    "latch_source_color_memory",
                    item,
                    name=f"latch_{item}_source_color",
                    metadata={"source_color": source_color, "source_slot": source_slot},
                ),
                navigate(
                    "target_plates",
                    purpose="target_place",
                    holding=item,
                    name=f"navigate_{item}_to_matching_target_plate",
                    metadata={"holding_navigation_mode": "rotate_then_translate", "xy_tolerance": 0.08, "yaw_tolerance": 0.12, "target_slot": target_slot, "source_color": source_color},
                ),
                operate(
                    "place_at_accessible",
                    item,
                    name=f"place_{item}_at_{target_slot}",
                    metadata={
                        **dict(actions.target_place_metadata[target_slot]),
                        "target_name": f"{item}_{target_slot}",
                        "target_resolver": "target_plate_pose",
                        "target_slot": target_slot,
                        "source_color": source_color,
                    },
                ),
                operate(
                    "verify_target_plate",
                    item,
                    name=f"verify_{item}_matching_target_plate",
                    metadata={"target_slot": target_slot, "source_color": source_color},
                ),
                operate(
                    "return_arm",
                    item,
                    name=f"return_arm_after_{item}_target_place",
                    metadata={**dict(actions.return_arm_empty), "stage": f"return_after_{item}_target_place"},
                ),
            ]
        )
    return finish_plan(
        episode,
        stages,
        metadata={
            "task_revision": TASK_REVISION,
            "carton_asset_by_item": dict(CARTON_ASSET_BY_ITEM),
            "source_plate_color_by_slot": dict(facts["source_plate_color_by_slot"]),
            "target_plate_color_by_slot": dict(facts["target_plate_color_by_slot"]),
            "source_slot_by_item": source_slots,
            "source_color_by_item": source_colors,
            "target_slot_by_color": target_by_color,
            "transport_order": list(order),
        },
    )


__all__ = ["build_expert_plan"]
