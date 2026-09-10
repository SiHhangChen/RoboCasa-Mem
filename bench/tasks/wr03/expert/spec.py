"""Canonical WR-03 v3 expert plan with reset-time cabinet storage."""

from __future__ import annotations

from bench.execution.plan_builder import finish_plan, navigate, operate
from bench.tasks.contracts import ExpertPlanInputs


def build_expert_plan(inputs: ExpertPlanInputs):
    episode = inputs.episode
    if episode.task_id != "WR-03":
        raise ValueError(f"WR-03 cannot build a plan for {episode.task_id!r}")
    facts = dict(episode.facts)
    actions = inputs.action_config
    level = str(facts["stored_level"])
    close_order = tuple(str(v) for v in facts["close_order"])
    stages: list[object] = [
        navigate(
            "cabinet_front_station",
            purpose="pre_close_staging",
            name="navigate_to_pre_close_neutral",
            metadata={"use_manual_path": False, "xy_tolerance": 0.08, "yaw_tolerance": 0.12},
        ),
    ]
    for index, close_level in enumerate(close_order, start=1):
        stages.extend([
            navigate(f"{close_level}_door_station", purpose="close", name=f"navigate_to_door_{index}", metadata={"door_sequence_index": index}),
            operate("close_storage_door", close_level, name=f"close_door_{index}", metadata={**dict(actions.door_metadata), "door_sequence_index": index}),
            operate("return_arm", close_level, name=f"return_after_door_{index}", metadata={**dict(actions.return_arm_metadata), "gripper_action": -1.0}),
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
        navigate(f"{level}_door_station", purpose="retrieve_door", name="navigate_to_remembered_cabinet"),
        operate("open_retrieval_door", level, name="open_remembered_cabinet", metadata=actions.door_metadata),
        operate("verify_retrieval_door", level, name="verify_remembered_cabinet"),
        operate("return_arm", level, name="return_after_opening_cabinet", metadata={**dict(actions.return_arm_metadata), "gripper_action": -1.0}),
        navigate(
            f"{level}_storage_station",
            purpose="retrieve",
            name="navigate_to_can",
            metadata={
                "xy_tolerance": 0.02,
                "yaw_tolerance": 0.02,
                "force_terminal_settle": True,
            },
        ),
        operate("pick_colored_item", "red_can", name="retrieve_can", metadata=actions.retrieval_pick_metadata),
        operate("return_arm", "red_can", name="return_with_can", metadata={**dict(actions.return_arm_metadata), "gripper_action": 0.0}),
        navigate(
            "item_station",
            purpose="place",
            holding="red_can",
            name="navigate_to_initial_table",
            metadata={
                "xy_tolerance": 0.08,
                "yaw_tolerance": 0.12,
                "terminal_tolerance_only": True,
                "use_sampled_terminal_pose": True,
            },
        ),
        operate("place_on_table", "red_can", name="place_can_on_table", metadata=actions.storage_place_metadata),
        operate("verify_final_placement", "red_can", name="verify_final_table_placement"),
    ])
    return finish_plan(episode, stages, metadata={
        "task_revision": "WR03-v3",
        "stored_level": level,
        "close_order": list(close_order),
    })


__all__ = ["build_expert_plan"]
