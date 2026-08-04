"""Validated physical slide-door primitive for WR01."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from bench.tasks.wr03.execution.calibration import (
    matrix_with_pose,
    resolve_anchor_pose_world,
    target_in_current_base,
    without_obs,
)


def move_slide_door(
    controller: Any,
    *,
    level: str,
    pose_file: str | Path,
    opening: bool,
    pregrasp_distance: float,
    retreat_distance: float,
    open_distances: tuple[float, ...],
    close_deltas: tuple[float, ...],
    gripper_open_steps: int,
    gripper_close_steps: int,
    gripper_release_steps: int,
    motion_settle_steps: int,
    final_motion_settle_steps: int,
    retreat_settle_steps: int,
    opened_threshold: float,
    closed_threshold: float,
) -> dict[str, Any]:
    """Open or close one cabinet through planned handle contact."""

    env = controller.env
    fixture = env.cabinet_for_level(level)
    joint_name = _single_joint_name(fixture)
    initial = _joint_metrics(env, fixture, joint_name)
    grasp_world, grasp_info = resolve_anchor_pose_world(env, pose_file)
    axis_world, axis_info = _slide_axis_world(env, joint_name)
    poses = _motion_poses(
        grasp_world=grasp_world,
        axis_world=axis_world,
        opening=opening,
        pregrasp_distance=float(pregrasp_distance),
        open_distances=open_distances,
        close_deltas=close_deltas,
        current_distance=float(initial["raw_qpos"] - axis_info["range"][0]),
    )
    targets = {
        name: target_in_current_base(controller, matrix)
        for name, matrix in poses.items()
    }
    execute = bool(controller.runtime_config.execute)
    stages: list[dict[str, Any]] = []
    commands: dict[str, Any] = {}
    commands["open_before_approach"] = without_obs(
        controller.run_gripper_command(-1.0, int(gripper_open_steps))
    )
    for name in ("pregrasp", "grasp"):
        stage = controller.run_planned_stage(
            targets[name],
            stage=f"{level}_door_{'open' if opening else 'close'}_{name}",
            gripper_action=-1.0,
            execute=execute,
            settle_steps=int(controller.runtime_config.settle_steps),
        )
        stages.append(stage)
        if not bool(stage.get("pass", False)):
            break
    approached = len(stages) == 2 and all(bool(stage.get("pass", False)) for stage in stages)
    motion_names = [
        name
        for name in poses
        if name.startswith("open_") or name.startswith("close_")
    ]
    if approached:
        commands["close_on_handle"] = without_obs(
            controller.run_gripper_command(1.0, int(gripper_close_steps))
        )
        for index, name in enumerate(motion_names):
            is_final_motion = index == len(motion_names) - 1
            stage = controller.run_planned_stage(
                targets[name],
                stage=f"{level}_door_{name}",
                gripper_action=1.0,
                execute=execute,
                settle_steps=int(
                    final_motion_settle_steps
                    if is_final_motion
                    else motion_settle_steps
                ),
            )
            stages.append(stage)
            if not bool(stage.get("pass", False)):
                break
    final = _joint_metrics(env, fixture, joint_name)
    if approached:
        commands["release_handle"] = without_obs(
            controller.run_gripper_command(-1.0, int(gripper_release_steps))
        )
        retreat_world = np.asarray(
            controller.stage_executor.current_eef_site_world(), dtype=float
        ).copy()
        retreat_axis = np.asarray(grasp_world[:3, 2], dtype=float)
        retreat_axis /= max(float(np.linalg.norm(retreat_axis)), 1e-12)
        retreat_world[:3, 3] -= retreat_axis * float(retreat_distance)
        retreat = controller.run_planned_stage(
            target_in_current_base(controller, retreat_world),
            stage=f"{level}_door_{'open' if opening else 'close'}_retreat",
            gripper_action=-1.0,
            execute=execute,
            settle_steps=int(retreat_settle_steps),
        )
        stages.append(retreat)
    stages_pass = bool(stages) and all(bool(stage.get("pass", False)) for stage in stages)
    state_pass = (
        float(final["open_fraction"]) >= float(opened_threshold)
        if opening
        else float(final["open_fraction"]) <= float(closed_threshold)
    ) if execute else True
    checks = {"planned_and_executed": stages_pass, "door_state": state_pass}
    return {
        "primitive": "open_storage_door" if opening else "close_storage_door",
        "control_mode": "physical_handle",
        "level": str(level),
        "initial_metrics": initial,
        "final_metrics": final,
        "axis": axis_info,
        "grasp": grasp_info,
        "motion_targets": list(motion_names),
        "settle_policy": {
            "approach": int(controller.runtime_config.settle_steps),
            "motion": int(motion_settle_steps),
            "final_motion": int(final_motion_settle_steps),
            "retreat": int(retreat_settle_steps),
        },
        "commands": commands,
        "stages": without_obs(stages),
        "checks": checks,
        "pass": bool(all(checks.values())),
        "reason": None if all(checks.values()) else "physical_door_motion_failed",
    }


def _single_joint_name(fixture: Any) -> str:
    names = list(getattr(fixture, "door_joint_names", ()) or ())
    if len(names) != 1:
        raise RuntimeError(f"WR-01 slide cabinet needs one joint, got {names}")
    return str(names[0])


def _joint_metrics(env: Any, fixture: Any, joint_name: str) -> dict[str, Any]:
    raw = float(env.sim.data.qpos[env.sim.model.get_joint_qpos_addr(joint_name)])
    normalized = float(fixture.get_joint_state(env, [joint_name])[joint_name])
    return {"joint_name": joint_name, "raw_qpos": raw, "open_fraction": normalized}


def _slide_axis_world(env: Any, joint_name: str) -> tuple[np.ndarray, dict[str, Any]]:
    model = env.sim.model
    joint_id = int(model.joint_name2id(joint_name))
    body_id = int(model.jnt_bodyid[joint_id])
    local = np.asarray(model.jnt_axis[joint_id], dtype=float)
    rotation = np.asarray(env.sim.data.body_xmat[body_id], dtype=float).reshape(3, 3)
    world = rotation @ local
    world /= max(float(np.linalg.norm(world)), 1e-12)
    joint_range = np.asarray(model.jnt_range[joint_id], dtype=float)
    return world, {"local": local.tolist(), "world": world.tolist(), "range": joint_range.tolist()}


def _motion_poses(
    *,
    grasp_world: np.ndarray,
    axis_world: np.ndarray,
    opening: bool,
    pregrasp_distance: float,
    open_distances: tuple[float, ...],
    close_deltas: tuple[float, ...],
    current_distance: float,
) -> dict[str, np.ndarray]:
    approach = np.asarray(grasp_world[:3, 2], dtype=float)
    approach /= max(float(np.linalg.norm(approach)), 1e-12)
    position = np.asarray(grasp_world[:3, 3], dtype=float)
    rotation = np.asarray(grasp_world[:3, :3], dtype=float)
    poses = {
        "pregrasp": matrix_with_pose(position - approach * pregrasp_distance, rotation),
        "grasp": np.asarray(grasp_world, dtype=float),
    }
    distances = open_distances if opening else close_deltas
    for index, distance in enumerate(distances, start=1):
        delta = float(distance) if opening else min(float(distance), max(0.0, current_distance))
        direction = 1.0 if opening else -1.0
        poses[f"{'open' if opening else 'close'}_{index}"] = matrix_with_pose(
            position + direction * axis_world * delta,
            rotation,
        )
    return poses


__all__ = ["move_slide_door"]
