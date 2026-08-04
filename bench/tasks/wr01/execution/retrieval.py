"""Validated cabinet-front retrieval corridor for stored WR-01 items."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from bench.execution.planning.pick_place_targets import object_pose_matrix
from bench.tasks.wr01.execution.calibration import (
    matrix_with_pose,
    resolve_anchor_pose_world,
    target_in_current_base,
    without_obs,
)
from bench.tasks.wr01.execution.storage import adopt_picked_item


def retrieve_item_from_cabinet(
    controller: Any,
    item_id: str,
    *,
    lower_prepose_file: str | Path,
    front_clearance_distance: float,
    front_pregrasp_distance: float,
    front_clearance_z_offset: float,
    grasp_z_offset: float,
    grasp_lateral_offset: float,
    grasp_inward_offset: float,
    close_gripper_steps: int,
    post_close_hold_steps: int,
    held_gripper_action: float,
    disable_collisions_during_transport: bool,
) -> dict[str, Any]:
    """Approach the live stored item collinearly from the cabinet front and extract it."""

    env = controller.env
    level = str(env.level_by_item[str(item_id)])
    initial_object = object_pose_matrix(env, item_id)
    inward, lateral = _cabinet_front_axes(env, level)
    up = np.asarray([0.0, 0.0, 1.0], dtype=float)
    rotation = np.column_stack((lateral, up, inward))
    grasp_position = (
        initial_object[:3, 3]
        + lateral * float(grasp_lateral_offset)
        + inward * float(grasp_inward_offset)
        + up * float(grasp_z_offset)
    )
    grasp = matrix_with_pose(grasp_position, rotation)
    pregrasp = matrix_with_pose(
        grasp_position - inward * float(front_pregrasp_distance),
        rotation,
    )
    clearance = matrix_with_pose(
        pregrasp[:3, 3]
        - inward * float(front_clearance_distance)
        + up * float(front_clearance_z_offset),
        rotation,
    )
    poses: dict[str, np.ndarray] = {}
    calibration: dict[str, Any] = {}
    if level == "lower":
        lower_prepose, lower_prepose_info = resolve_anchor_pose_world(
            env,
            lower_prepose_file,
        )
        poses["lower_retrieval_prepose"] = lower_prepose
        calibration["lower_retrieval_prepose"] = lower_prepose_info
    poses.update(
        {
            "front_clearance": clearance,
            "front_pregrasp": pregrasp,
            "front_grasp": grasp,
        }
    )
    targets = {
        name: target_in_current_base(controller, pose)
        for name, pose in poses.items()
    }
    stages: list[dict[str, Any]] = []
    for name in poses:
        stage = controller.run_planned_stage(
            targets[name],
            stage=f"retrieval_stored_{name}",
            gripper_action=-1.0,
            execute=True,
            record_trajectory=True,
        )
        stages.append(stage)
        if not bool(stage.get("pass", False)):
            break

    approached = len(stages) == len(poses) and all(
        bool(stage.get("pass", False)) for stage in stages
    )
    required_prepose_passed = bool(
        level != "lower"
        or (
            stages
            and stages[0].get("stage")
            == "retrieval_stored_lower_retrieval_prepose"
            and stages[0].get("pass", False)
        )
    )
    close_command: dict[str, Any] = {"skipped": True}
    hold_command: dict[str, Any] = {"skipped": True}
    validation: dict[str, Any] = {}
    adoption: dict[str, Any] = {"pass": False, "reason": "approach_failed"}
    retreat: dict[str, Any] = {"pass": False, "skipped": True}
    if approached:
        close_command = without_obs(
            controller.run_gripper_command(1.0, int(close_gripper_steps))
        )
        hold_command = without_obs(
            controller.run_gripper_command(
                float(held_gripper_action),
                int(post_close_hold_steps),
            )
        )
        validation = controller._pick_grasp_validation(
            str(item_id),
            initial_object_pos=initial_object[:3, 3],
        )
        grasp_confirmed = bool(
            validation.get("contact", False)
            or validation.get("grasped_threshold_0.6", False)
        )
        if grasp_confirmed:
            eef_world = controller.stage_executor.current_eef_site_world()
            controller._virtual_held_object = str(item_id)
            controller._virtual_object_from_eef = (
                np.linalg.inv(eef_world) @ object_pose_matrix(env, item_id)
            )
            adoption = adopt_picked_item(
                controller,
                str(item_id),
                disable_collisions=bool(disable_collisions_during_transport),
            )
            if bool(adoption.get("pass", False)):
                retreat = _reverse_approach(
                    controller,
                    stages,
                    held_gripper_action=float(held_gripper_action),
                )

    controller.refresh_obs()
    env.update_state()
    extracted = bool(
        approached
        and adoption.get("pass", False)
        and retreat.get("pass", False)
        and not env.is_item_in_level(str(item_id), level)
    )
    retrieval_recorded = bool(
        extracted and env.record_retrieval_pick(str(item_id))
    )
    checks = {
        "required_prepose": required_prepose_passed,
        "approach": approached,
        "close_command": approached and not close_command.get("skipped", False),
        "validated_grasp": bool(adoption.get("pass", False)),
        "retreat": bool(retreat.get("pass", False)),
        "item_extracted": extracted,
        "retrieval_event_recorded": retrieval_recorded,
    }
    return {
        "primitive": "retrieve_item_from_cabinet",
        "object_name": str(item_id),
        "level": level,
        "approach": {
            "policy": "cabinet_front_collinear_retrieval",
            "pose_order": list(poses),
            "required_lower_prepose": level == "lower",
            "inward_world": inward.tolist(),
            "lateral_world": lateral.tolist(),
            "front_clearance_distance": float(front_clearance_distance),
            "front_pregrasp_distance": float(front_pregrasp_distance),
            "front_clearance_z_offset": float(front_clearance_z_offset),
            "grasp_z_offset": float(grasp_z_offset),
            "grasp_lateral_offset": float(grasp_lateral_offset),
            "grasp_inward_offset": float(grasp_inward_offset),
            "targets_world": {
                name: pose.tolist() for name, pose in poses.items()
            },
        },
        "calibration": calibration,
        "stages": without_obs(stages),
        "close_gripper": close_command,
        "post_close_hold": hold_command,
        "grasp_validation": validation,
        "transport_adoption": adoption,
        "retreat": without_obs(retreat),
        "checks": checks,
        "pass": bool(all(checks.values())),
        "reason": None if all(checks.values()) else "cabinet_front_retrieval_failed",
    }


def _cabinet_front_axes(env: Any, level: str) -> tuple[np.ndarray, np.ndarray]:
    fixture = env.cabinet_for_level(level)
    preinsert_name = f"{fixture.name}_item_preinsert_site"
    release_name = f"{fixture.name}_item_release_site"
    model = env.sim.model
    data = env.sim.data
    preinsert = np.asarray(
        data.site_xpos[int(model.site_name2id(preinsert_name))], dtype=float
    )
    release = np.asarray(
        data.site_xpos[int(model.site_name2id(release_name))], dtype=float
    )
    inward = release - preinsert
    inward[2] = 0.0
    inward /= max(float(np.linalg.norm(inward)), 1e-12)
    up = np.asarray([0.0, 0.0, 1.0], dtype=float)
    lateral = np.cross(up, inward)
    lateral /= max(float(np.linalg.norm(lateral)), 1e-12)
    return inward, lateral


def _reverse_approach(
    controller: Any,
    stages: list[dict[str, Any]],
    *,
    held_gripper_action: float,
) -> dict[str, Any]:
    trajectory: list[list[float]] = []
    for stage in reversed(stages):
        waypoints = stage.get("executed_trajectory") or stage.get(
            "planned_trajectory"
        )
        if waypoints:
            trajectory.extend(list(reversed(waypoints)))
    if trajectory:
        result = controller.execute_arm_trajectory(
            trajectory,
            stage="retrieval_retreat",
            gripper_action=float(held_gripper_action),
        )
        result["strategy"] = "reverse_cabinet_front_approach"
        result["trajectory_len"] = len(trajectory)
        return result
    return {
        "stage": "retrieval_retreat",
        "strategy": "reverse_cabinet_front_approach",
        "trajectory_len": 0,
        "pass": False,
        "reason": "approach_trajectory_unavailable",
    }


__all__ = ["retrieve_item_from_cabinet"]
