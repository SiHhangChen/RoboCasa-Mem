"""Calibrated, action-only physical item insertion for TM02."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from bench.tasks.tm02.execution.calibration import (
    resolve_anchor_pose_world,
    target_in_current_base,
    without_obs,
)


def adopt_picked_item(controller: Any, item_id: str) -> dict[str, Any]:
    """Publish a physical grasp to the planner without changing simulation state."""

    if (
        controller._virtual_held_object != str(item_id)
        or controller._virtual_object_from_eef is None
    ):
        return {
            "pass": False,
            "reason": "validated_pick_did_not_publish_held_transform",
            "object_name": str(item_id),
        }
    return {
        "pass": True,
        "object_name": str(item_id),
        "mode": "physical_contact_hold",
        "collisions_disabled": False,
        "simulation_state_modified": False,
    }


def place_item_with_calibration(
    controller: Any,
    item_id: str,
    *,
    level: str,
    preinsert_file: str | Path,
    release_file: str | Path,
    insertion_plan_attempts: int,
    release_settle_steps: int,
    held_gripper_action: float = 0.0,
) -> dict[str, Any]:
    """Move through the validated cabinet entry corridor, release, and retreat."""

    execute = bool(controller.runtime_config.execute)
    if execute and controller._virtual_held_object != str(item_id):
        return {
            "primitive": "place_in_storage",
            "object_name": item_id,
            "level": level,
            "pass": False,
            "reason": "physical_pick_not_validated",
        }
    preinsert_world, preinsert_info = resolve_anchor_pose_world(controller.env, preinsert_file)
    release_world, release_info = resolve_anchor_pose_world(controller.env, release_file)
    if str(level) == "upper":
        lower = np.asarray(controller.env.cabinet_for_level("lower").pos, dtype=float)
        upper = np.asarray(controller.env.cabinet_for_level("upper").pos, dtype=float)
        offset = upper - lower
        preinsert_world[:3, 3] += offset
        release_world[:3, 3] += offset
        preinsert_info["cabinet_level_offset_world"] = offset.tolist()
        release_info["cabinet_level_offset_world"] = offset.tolist()
    targets = {
        "preinsert": target_in_current_base(controller, preinsert_world),
        "release": target_in_current_base(controller, release_world),
    }
    stages: list[dict[str, Any]] = []
    release_stage: dict[str, Any] | None = None
    for name in ("preinsert", "release"):
        stage = None
        for attempt in range(1, max(1, int(insertion_plan_attempts)) + 1):
            stage = controller.run_planned_stage(
                targets[name],
                stage=f"{level}_item_{name}",
                gripper_action=float(held_gripper_action),
                holding_object=item_id,
                execute=execute,
                record_trajectory=(name == "release"),
            )
            stage["planning_attempt"] = attempt
            if bool(stage.get("pass", False)) or stage.get("failure_phase") != "plan":
                break
        assert stage is not None
        stages.append(stage)
        if not bool(stage.get("pass", False)):
            break
        if name == "release":
            release_stage = stage
    released = release_stage is not None and bool(release_stage.get("pass", False))
    release_command: dict[str, Any] = {"skipped": True}
    release_settle: dict[str, Any] = {"skipped": True}
    physics_handoff: dict[str, Any] = {"pass": False, "skipped": True}
    retreat: dict[str, Any] = {"pass": False, "skipped": True}
    if released:
        release_command = without_obs(
            controller.run_gripper_command(-1.0, controller.runtime_config.release_steps)
        )
        controller._clear_virtual_held_object(item_id)
        physics_handoff = {
            "pass": True,
            "skipped": False,
            "object_name": str(item_id),
            "mode": "action_only_physics",
            "simulation_state_modified": False,
        }
        release_settle = without_obs(
            controller.run_gripper_command(-1.0, max(0, int(release_settle_steps)))
        )
        trajectory = release_stage.get("executed_trajectory") or release_stage.get(
            "planned_trajectory"
        )
        if trajectory:
            retreat = controller.execute_arm_trajectory(
                list(reversed(trajectory)),
                stage=f"{level}_item_retreat",
                gripper_action=-1.0,
            )
        else:
            retreat = controller.run_planned_stage(
                targets["preinsert"],
                stage=f"{level}_item_retreat",
                gripper_action=-1.0,
                execute=execute,
            )
    checks = {
        "preinsert": bool(stages and stages[0].get("pass", False)),
        "release_pose": bool(released),
        "release_command": bool(released and not release_command.get("skipped", False)),
        "physics_handoff": bool(physics_handoff.get("pass", False)),
        "release_settle": bool(released and not release_settle.get("skipped", False)),
        "retreat": bool(retreat.get("pass", False)),
    }
    if not execute:
        checks["release_command"] = True
        checks["physics_handoff"] = True
        checks["release_settle"] = True
    return {
        "primitive": "place_in_storage",
        "object_name": item_id,
        "level": str(level),
        "calibration": {"preinsert": preinsert_info, "release": release_info},
        "stages": without_obs(stages),
        "release_gripper": release_command,
        "physics_handoff": physics_handoff,
        "release_settle": release_settle,
        "retreat": without_obs(retreat),
        "checks": checks,
        "pass": bool(all(checks.values())),
        "reason": None if all(checks.values()) else "calibrated_storage_insertion_failed",
    }


__all__ = ["adopt_picked_item", "place_item_with_calibration"]
