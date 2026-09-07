"""Canonical WR-01 v3 expert plan with reset-time cabinet storage."""

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
    target = str(facts["target_item_id"])
    target_level = str(facts["target_level"])
    close_order = tuple(str(v) for v in facts["close_order"])

    stages: list[object] = [
        navigate(
            "cabinet_front_station",
            purpose="pre_close_staging",
            name="navigate_to_pre_close_neutral",
            metadata={"use_manual_path": False, "xy_tolerance": 0.08, "yaw_tolerance": 0.12},
        ),
    ]
    for index, level in enumerate(close_order, start=1):
        stages.extend([
            navigate(f"{level}_door_station", purpose="close", name=f"navigate_to_door_{index}", metadata={"door_sequence_index": index}),
            operate("close_storage_door", level, name=f"close_door_{index}", metadata={**dict(actions.door_metadata), "door_sequence_index": index}),
            operate("return_arm", level, name=f"return_after_door_{index}", metadata={**dict(actions.return_arm_metadata), "gripper_action": -1.0}),
        ])
    stages.extend([
        navigate(
            "cabinet_front_station",
            purpose="memory_handoff",
            name="navigate_to_neutral_reset",
            metadata={"use_manual_path": False, "xy_tolerance": 0.08, "yaw_tolerance": 0.12},
        ),
        operate("canonical_memory_reset", "memory_handoff", name="canonical_memory_reset"),
        operate("latch_memory_handoff", "memory_handoff", name="latch_memory_handoff"),
        navigate(f"{target_level}_door_station", purpose="retrieve_door", name="navigate_to_remembered_cabinet"),
        operate("open_retrieval_door", target_level, name="open_remembered_target_cabinet", metadata=actions.door_metadata),
        operate("verify_retrieval_door", target_level, name="verify_remembered_target_cabinet"),
        operate("return_arm", target_level, name="return_after_opening_target_cabinet", metadata={**dict(actions.return_arm_metadata), "gripper_action": -1.0}),
        navigate(
            f"{target_level}_storage_station",
            purpose="retrieve",
            name="navigate_to_target_can",
            metadata={
                "xy_tolerance": 0.02,
                "yaw_tolerance": 0.02,
                "force_terminal_settle": True,
            },
        ),
        operate("pick_colored_item", target, name="pick_requested_colored_can", metadata=actions.retrieval_pick_metadata),
        operate("return_arm", target, name="return_with_requested_can", metadata={**dict(actions.return_arm_metadata), "gripper_action": 0.0}),
        navigate(
            "item_station",
            purpose="place",
            holding=target,
            name="navigate_to_initial_table",
            metadata={
                "xy_tolerance": 0.08,
                "yaw_tolerance": 0.12,
                "terminal_tolerance_only": True,
                "use_sampled_terminal_pose": True,
            },
        ),
        operate("place_on_table", target, name="place_requested_can_on_table", metadata=actions.storage_place_metadata),
        operate("verify_final_placement", target, name="verify_final_table_placement"),
    ])
    return finish_plan(episode, stages, metadata={
        "task_revision": "WR01-v3",
        "target_item_id": target,
        "target_level": target_level,
        "close_order": list(close_order),
    })


__all__ = ["build_expert_plan"]
