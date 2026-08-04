"""Calibrated item insertion and held-object support for WR01."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from bench.tasks.wr01.execution.calibration import (
    resolve_anchor_pose_world,
    target_in_current_base,
    without_obs,
)


def install_held_item_follow_assist(env: Any, controller: Any) -> None:
    """Keep a validated item attachment stable during arm and mobile-base motion."""

    controller._wr01_hold_enabled = False
    controller._wr01_collision_state = {}
    original_step = env.step

    def step_with_wr01_attachment(action: Any):
        resolved_action = _neutralize_held_close_command(env, action, controller)
        result = original_step(resolved_action)
        held = controller._virtual_held_object
        transform = controller._virtual_object_from_eef
        if bool(controller._wr01_hold_enabled) and held is not None and transform is not None:
            pose_world = controller.stage_executor.current_eef_site_world() @ transform
            _set_object_pose_matrix(env, held, pose_world)
        if bool(controller._wr01_hold_enabled) and held is not None:
            obs = env._get_observations(force_update=True)
            if isinstance(result, tuple) and len(result) == 5:
                return obs, result[1], result[2], result[3], result[4]
            return obs, result[1], result[2], result[3]
        return result

    env.step = step_with_wr01_attachment


def _neutralize_held_close_command(
    env: Any,
    action: Any,
    controller: Any,
) -> Any:
    """Preserve the validated jaw target instead of repeatedly squeezing in transit."""

    if not bool(controller._wr01_hold_enabled):
        return action
    indexes = getattr(
        env.robots[0].composite_controller,
        "_action_split_indexes",
        {},
    )
    bounds = indexes.get("right_gripper")
    if bounds is None:
        return action
    start, end = (int(bounds[0]), int(bounds[1]))
    values = np.asarray(action, dtype=float)
    if values.ndim != 1 or start < 0 or end > values.size or start >= end:
        return action
    if not np.any(values[start:end] > 0.0):
        return action
    neutral = values.copy()
    neutral[start:end] = np.minimum(neutral[start:end], 0.0)
    return neutral


def adopt_picked_item(
    controller: Any,
    item_id: str,
    *,
    disable_collisions: bool,
) -> dict[str, Any]:
    """Enable the transport assist only after the shared physical pick passed."""

    if (
        controller._virtual_held_object != str(item_id)
        or controller._virtual_object_from_eef is None
    ):
        return {
            "pass": False,
            "reason": "validated_pick_did_not_publish_held_transform",
            "object_name": str(item_id),
        }
    if disable_collisions:
        _disable_object_collisions(controller, item_id)
    controller._wr01_hold_enabled = True
    return {
        "pass": True,
        "object_name": str(item_id),
        "mode": "validated_pick_attachment",
        "collisions_disabled": bool(disable_collisions),
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
    if execute and not bool(controller._wr01_hold_enabled):
        return {
            "primitive": "place_in_storage",
            "object_name": item_id,
            "level": level,
            "pass": False,
            "reason": "item_hold_assist_not_active",
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
            if bool(stage.get("pass", False)):
                break
            if stage.get("failure_phase") != "plan":
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
        # Keep the validated transport attachment active only while the jaws open.
        # Collisions remain disabled during this short clearance window so the
        # formerly closed fingers cannot eject the item when contacts are restored.
        release_command = without_obs(
            controller.run_gripper_command(-1.0, controller.runtime_config.release_steps)
        )
        restored_geom_count = _restore_object_collisions(controller, item_id)
        velocity_reset = _zero_object_free_joint_velocity(controller, item_id)
        controller._wr01_hold_enabled = False
        controller._clear_virtual_held_object(item_id)
        physics_handoff = {
            "pass": True,
            "skipped": False,
            "object_name": str(item_id),
            "restored_collision_geom_count": int(restored_geom_count),
            "velocity_reset": velocity_reset,
        }
        # Let gravity and cabinet contacts settle the released item before the
        # arm moves away. This preserves the shared release -> settle -> retreat
        # lifecycle and prevents the visible post-release suspension.
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


def _disable_object_collisions(controller: Any, object_name: str) -> None:
    state = controller._wr01_collision_state.setdefault(str(object_name), [])
    if state:
        return
    obj = controller.env.objects[str(object_name)]
    for geom_name in list(obj.contact_geoms or []):
        geom_id = int(controller.env.sim.model.geom_name2id(geom_name))
        state.append(
            (
                geom_id,
                int(controller.env.sim.model.geom_contype[geom_id]),
                int(controller.env.sim.model.geom_conaffinity[geom_id]),
            )
        )
        controller.env.sim.model.geom_contype[geom_id] = 0
        controller.env.sim.model.geom_conaffinity[geom_id] = 0


def _restore_object_collisions(controller: Any, object_name: str) -> int:
    state = list(
        controller._wr01_collision_state.get(str(object_name), [])
    )
    for geom_id, contype, conaffinity in state:
        controller.env.sim.model.geom_contype[geom_id] = int(contype)
        controller.env.sim.model.geom_conaffinity[geom_id] = int(conaffinity)
    controller._wr01_collision_state[str(object_name)] = []
    controller.env.sim.forward()
    return len(state)


def _zero_object_free_joint_velocity(
    controller: Any,
    object_name: str,
) -> dict[str, Any]:
    """Reset only the released object's free-joint velocity at physics handoff."""

    obj = controller.env.objects[str(object_name)]
    joint_name = str(obj.joints[0])
    qvel = np.asarray(
        controller.env.sim.data.get_joint_qvel(joint_name),
        dtype=float,
    ).reshape(-1)
    if qvel.size == 0 or not np.all(np.isfinite(qvel)):
        raise ValueError(f"invalid free-joint velocity for {object_name!r}")
    initial_norm = float(np.linalg.norm(qvel))
    controller.env.sim.data.set_joint_qvel(joint_name, np.zeros_like(qvel))
    controller.env.sim.forward()
    return {
        "joint_name": joint_name,
        "dof_count": int(qvel.size),
        "initial_norm": initial_norm,
        "final_norm": 0.0,
    }


def _set_object_pose_matrix(env: Any, object_name: str, pose_world: np.ndarray) -> None:
    matrix = np.asarray(pose_world, dtype=float).reshape(4, 4)
    # MuJoCo free-joint qpos stores quaternions as wxyz.
    from bench.execution.planning import pose_from_matrix

    pose = pose_from_matrix(matrix)
    env.set_object_pose(object_name, pose["p"], pose["q_xyzw"])


__all__ = [
    "adopt_picked_item",
    "install_held_item_follow_assist",
    "place_item_with_calibration",
]
