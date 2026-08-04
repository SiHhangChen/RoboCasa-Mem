"""Mobile-base execution helpers.

This is the bench-owned Phase 6 replacement for the previous base-navigation
helpers. It intentionally keeps env stepping here, outside the pure planning
algorithms.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from bench.execution.planning.nav_algo import normalize_angle
from bench.execution.navigation.velocity_profile import BaseVelocityLimits, planner_for_env


def qpos_for_controller(env: Any, controller: Any) -> np.ndarray:
    return np.array(env.sim.data.qpos[controller.qpos_index], dtype=float)


def action_dict_for_arm_and_gripper(
    env: Any,
    *,
    arm_qpos: np.ndarray,
    gripper_action: float | None = None,
) -> dict[str, np.ndarray]:
    robot = env.robots[0]
    action_dict: dict[str, np.ndarray] = {}
    for part_name, (start, end) in robot.composite_controller._action_split_indexes.items():
        action_dict[part_name] = np.zeros(end - start, dtype=float)
    action_dict["right"] = np.array(arm_qpos, dtype=float)
    if gripper_action is not None and "right_gripper" in action_dict:
        action_dict["right_gripper"] = np.full_like(
            action_dict["right_gripper"],
            float(gripper_action),
            dtype=float,
        )
    return action_dict


def base_yaw(obs: dict[str, Any]) -> float:
    x, y, z, w = np.array(obs["robot0_base_quat"], dtype=float)
    return float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


def world_vec_to_base(obs: dict[str, Any], vec_world: np.ndarray) -> np.ndarray:
    yaw = base_yaw(obs)
    c = float(np.cos(yaw))
    s = float(np.sin(yaw))
    rot_base_to_world = np.array(
        [
            [c, -s, 0.0],
            [s, c, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    return rot_base_to_world.T @ np.array(vec_world, dtype=float)


def planar_world_vec_to_local_yaw(vec_world_xy: np.ndarray, yaw: float) -> np.ndarray:
    vec_world_xy = np.array(vec_world_xy, dtype=float)[:2]
    c = float(np.cos(yaw))
    s = float(np.sin(yaw))
    rot_local_to_world = np.array([[c, -s], [s, c]], dtype=float)
    return rot_local_to_world.T @ vec_world_xy


def planar_local_yaw_vec_to_world(vec_local_xy: np.ndarray, yaw: float) -> np.ndarray:
    vec_local_xy = np.array(vec_local_xy, dtype=float)[:2]
    c = float(np.cos(yaw))
    s = float(np.sin(yaw))
    rot_local_to_world = np.array([[c, -s], [s, c]], dtype=float)
    return rot_local_to_world @ vec_local_xy


def _base_translation_anchor_yaw(env: Any) -> float:
    import robosuite.utils.transform_utils as T

    anchor_ori = np.array(
        getattr(env, "init_robot_base_ori_anchor", [0.0, 0.0, 0.0]),
        dtype=float,
    )
    return float(T.mat2euler(T.euler2mat(anchor_ori))[2])


def _controller_pose_yaws(env: Any) -> tuple[float, float]:
    import robosuite.utils.transform_utils as T

    controller = env.robots[0].composite_controller.part_controllers.get("base")
    if controller is None:
        return 0.0, 0.0
    init_ori = getattr(controller, "init_ori", None)
    init_yaw = 0.0 if init_ori is None else float(T.mat2euler(init_ori)[2])
    _, current_ori = controller.get_base_pose()
    current_yaw = float(T.mat2euler(current_ori)[2])
    return init_yaw, current_yaw


def controller_translation_yaw(env: Any) -> float:
    """World yaw of raw base action x/y after the mobile-base controller transform."""

    init_yaw, current_yaw = _controller_pose_yaws(env)
    return normalize_angle(_base_translation_anchor_yaw(env) + current_yaw - init_yaw)


def world_xy_to_controller_action(env: Any, vec_world_xy: np.ndarray) -> np.ndarray:
    vec_world_xy = np.array(vec_world_xy, dtype=float).reshape(-1)[:2]
    return planar_world_vec_to_local_yaw(
        vec_world_xy,
        controller_translation_yaw(env),
    )


def _clip_action_norm(values: np.ndarray, *, max_norm: float = 1.0) -> np.ndarray:
    values = np.array(values, dtype=float)
    norm = float(np.linalg.norm(values))
    if norm > float(max_norm) > 0.0:
        return values / norm * float(max_norm)
    return values


def _single_axis_action(
    value: float,
    *,
    gain: float,
    max_mag: float,
    min_mag: float = 0.0,
) -> float:
    raw = float(np.clip(float(value) * float(gain), -float(max_mag), float(max_mag)))
    if float(min_mag) > 0.0 and abs(float(value)) > 1e-9 and 0.0 < abs(raw) < float(min_mag):
        raw = float(np.sign(raw) * float(min_mag))
    return raw


def _set_terminal_axis_action(
    env: Any,
    base_action: np.ndarray,
    *,
    axis: int,
    value: float,
    target_yaw: float,
    action_frame: str,
) -> None:
    local_action = np.zeros(2, dtype=float)
    local_action[int(axis)] = float(value)
    if action_frame == "world":
        command_xy = world_xy_to_controller_action(
            env,
            planar_local_yaw_vec_to_world(local_action, float(target_yaw)),
        )
    elif action_frame == "base":
        command_xy = local_action
    else:
        raise ValueError(f"unsupported base action frame {action_frame!r}")
    if base_action.shape[0] >= 2:
        base_action[:2] = command_xy
    elif base_action.shape[0] >= 1:
        base_action[0] = float(command_xy[0])


def base_path_monitor_summary(monitor: dict[str, Any] | None) -> dict[str, Any] | None:
    if monitor is None:
        return None
    if "base_geoms" in monitor or "obstacle_geoms" in monitor:
        from bench.execution.planning.collision_monitor import base_path_monitor_summary as _summary

        return _summary(monitor)
    return {
        "enabled": bool(monitor.get("enabled", False)),
        "abort_on_collision": bool(monitor.get("abort_on_collision", True)),
        "fixture_keywords": monitor.get("fixture_keywords"),
        "exclude_keywords": monitor.get("exclude_keywords"),
        "holding_object": monitor.get("holding_object"),
    }


def check_base_path_collision(env: Any, monitor: dict[str, Any] | None) -> dict[str, Any]:
    if monitor is None or not monitor.get("enabled", False):
        return {"enabled": False, "collision": False}
    check_fn = monitor.get("check") or monitor.get("check_fn")
    if callable(check_fn):
        result = check_fn(env, monitor)
        result["enabled"] = True
        result.setdefault("collision", False)
        return result
    if "base_geoms" in monitor or "obstacle_geoms" in monitor:
        from bench.execution.planning.collision_monitor import check_base_path_collision as _check

        return _check(env, monitor)
    return {
        "enabled": True,
        "collision": False,
        "reason": "monitor_check_unavailable",
    }


def step_base_and_hold_arm(
    env: Any,
    *,
    arm_qpos: np.ndarray,
    base_action: np.ndarray,
    base_mode: float = 1.0,
    gripper_action: float | None = None,
    torso_action: float | np.ndarray | None = None,
    post_step_callback: Callable[[dict[str, Any]], dict[str, Any] | None] | None = None,
) -> dict[str, Any]:
    robot = env.robots[0]
    action_dict = action_dict_for_arm_and_gripper(
        env,
        arm_qpos=np.array(arm_qpos, dtype=float),
        gripper_action=gripper_action,
    )
    if "base" in action_dict:
        action_dict["base"] = np.array(base_action, dtype=float)
    if torso_action is not None:
        if "torso" not in action_dict:
            raise RuntimeError("robot action does not expose a torso controller")
        torso_values = np.array(torso_action, dtype=float).reshape(-1)
        if torso_values.size == 1:
            torso_values = np.full_like(
                action_dict["torso"],
                float(torso_values[0]),
                dtype=float,
            )
        if torso_values.shape != action_dict["torso"].shape:
            raise ValueError(
                f"torso action shape {torso_values.shape} does not match "
                f"controller shape {action_dict['torso'].shape}"
            )
        action_dict["torso"] = torso_values
    action_dict["base_mode"] = float(base_mode)
    action = robot.create_action_vector(action_dict)
    if not np.all(np.isfinite(action)):
        raise RuntimeError("base action contains NaN or Inf")
    obs, _, _, _ = env.step(action)
    if post_step_callback is not None:
        updated_obs = post_step_callback(obs)
        if updated_obs is not None:
            obs = updated_obs
    return obs


def sync_arm_controller_to_current_qpos(env: Any) -> dict[str, Any]:
    robot = env.robots[0]
    right = robot.composite_controller.part_controllers.get("right")
    if right is None:
        return {"synced": False, "reason": "right_controller_missing"}

    current_qpos = qpos_for_controller(env, right)
    if hasattr(right, "update_initial_joints"):
        right.update_initial_joints(current_qpos)
    if hasattr(right, "reset_goal"):
        right.reset_goal()
    if hasattr(robot.composite_controller, "reset"):
        robot.composite_controller.reset()
    return {
        "synced": True,
        "right_qpos": current_qpos.astype(float).tolist(),
    }


def set_base_pose_kinematic(
    env: Any,
    *,
    target_xy: np.ndarray,
    target_yaw: float | None,
) -> dict[str, Any]:
    import robosuite.utils.transform_utils as T
    from robocasa.utils.env_utils import set_robot_to_position

    target_xy = np.array(target_xy, dtype=float)
    target_yaw = 0.0 if target_yaw is None else float(target_yaw)
    anchor_ori = np.array(
        getattr(env, "init_robot_base_ori_anchor", [0.0, 0.0, 0.0]),
        dtype=float,
    )
    anchor_yaw = float(T.mat2euler(T.euler2mat(anchor_ori))[2])

    command_xy = target_xy.astype(float).copy()
    for _ in range(5):
        set_robot_to_position(env, np.array([command_xy[0], command_xy[1], 0.0], dtype=float))
        env.sim.data.qpos[env.sim.model.get_joint_qpos_addr("mobilebase0_joint_mobile_yaw")] = (
            normalize_angle(target_yaw - anchor_yaw)
        )
        env.sim.data.qvel[
            env.sim.model.get_joint_qvel_addr("mobilebase0_joint_mobile_forward")
        ] = 0.0
        env.sim.data.qvel[
            env.sim.model.get_joint_qvel_addr("mobilebase0_joint_mobile_side")
        ] = 0.0
        env.sim.data.qvel[
            env.sim.model.get_joint_qvel_addr("mobilebase0_joint_mobile_yaw")
        ] = 0.0
        env.sim.forward()
        obs = env._get_observations(force_update=True)
        actual_xy = np.array(obs["robot0_base_pos"][:2], dtype=float)
        residual = target_xy - actual_xy
        if float(np.linalg.norm(residual)) < 1e-4:
            break
        command_xy = command_xy + residual

    sync_arm_controller_to_current_qpos(env)
    return env._get_observations(force_update=True)


def _held_object_motion(
    env: Any,
    holding_object_name: str | None,
    initial_held_pos: np.ndarray | None,
    holding_drop_tolerance: float,
) -> dict[str, Any]:
    if holding_object_name is None:
        return {"enabled": False}
    final_held_pos = None
    try:
        final_held_pos = np.array(
            env.sim.data.body_xpos[env.obj_body_id[holding_object_name]],
            dtype=float,
        )
    except Exception:
        final_held_pos = None
    if initial_held_pos is not None and final_held_pos is not None:
        held_delta = final_held_pos - initial_held_pos
        return {
            "enabled": True,
            "object_name": holding_object_name,
            "initial_pos": initial_held_pos.astype(float).tolist(),
            "final_pos": final_held_pos.astype(float).tolist(),
            "delta": held_delta.astype(float).tolist(),
            "drop_tolerance": float(holding_drop_tolerance),
            "drop_ok": float(held_delta[2]) >= -float(holding_drop_tolerance),
        }
    return {
        "enabled": True,
        "object_name": holding_object_name,
        "reason": "object_position_unavailable",
        "drop_ok": False,
    }


def _initial_held_object_pos(env: Any, holding_object_name: str | None) -> np.ndarray | None:
    if holding_object_name is None:
        return None
    try:
        return np.array(env.sim.data.body_xpos[env.obj_body_id[holding_object_name]], dtype=float)
    except Exception:
        return None


def move_base_to_pose(
    env: Any,
    obs: dict[str, Any],
    *,
    target_xy: np.ndarray,
    target_yaw: float | None,
    arm_qpos: np.ndarray,
    gripper_action: float | None,
    tolerance: float,
    yaw_tolerance: float,
    max_steps: int,
    xy_gain: float = 4.0,
    yaw_gain: float = 2.0,
    action_frame: str = "world",
    translation_action_sign: float = 1.0,
    path_monitor: dict[str, Any] | None = None,
    holding_object_name: str | None = None,
    holding_drop_tolerance: float = 0.04,
    kinematic: bool = False,
    velocity_limits: BaseVelocityLimits | None = None,
    post_step_callback: Callable[[dict[str, Any]], dict[str, Any] | None] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    target_xy = np.array(target_xy, dtype=float)
    initial_held_pos = _initial_held_object_pos(env, holding_object_name)
    planner = planner_for_env(env, velocity_limits)
    if planner is not None:
        planner.begin()
    steps = 0
    checked_collision_steps = 0
    first_collision = None
    final_collision = {"enabled": False, "collision": False}
    if path_monitor is not None and path_monitor.get("enabled", False):
        checked_collision_steps += 1
        final_collision = check_base_path_collision(env, path_monitor)
        if final_collision.get("collision", False):
            first_collision = dict(final_collision)
            first_collision["phase"] = "initial"
            first_collision["checked_collision_steps"] = int(checked_collision_steps)
    if kinematic:
        obs = set_base_pose_kinematic(env, target_xy=target_xy, target_yaw=target_yaw)
    else:
        for _ in range(max_steps):
            base_xy = np.array(obs["robot0_base_pos"][:2], dtype=float)
            err_xy = target_xy - base_xy
            dist = float(np.linalg.norm(err_xy))
            yaw_err = (
                0.0
                if target_yaw is None
                else normalize_angle(float(target_yaw) - base_yaw(obs))
            )
            split = env.robots[0].composite_controller._action_split_indexes
            if "base" not in split:
                raise RuntimeError("base action part not found")
            start, end = split["base"]
            base_action = np.zeros(end - start, dtype=float)
            xy_done = dist <= tolerance
            yaw_done = target_yaw is None or abs(yaw_err) <= yaw_tolerance
            if xy_done and yaw_done:
                break
            if first_collision is not None and bool(
                (path_monitor or {}).get("abort_on_collision", True)
            ):
                break
            if path_monitor is not None and path_monitor.get("enabled", False):
                checked_collision_steps += 1
                final_collision = check_base_path_collision(env, path_monitor)
                if final_collision.get("collision", False):
                    first_collision = dict(final_collision)
                    first_collision["phase"] = "before_step"
                    first_collision["checked_collision_steps"] = int(checked_collision_steps)
                    if bool(path_monitor.get("abort_on_collision", True)):
                        break

            # 当距离很近且主要是旋转任务时，只执行旋转，避免漂移
            rotation_only_threshold = tolerance * 1.5
            is_rotation_only = (
                dist <= rotation_only_threshold
                and not yaw_done
                and abs(yaw_err) > yaw_tolerance * 0.5
            )

            if not xy_done and not is_rotation_only:
                if action_frame == "base":
                    translate_error = world_vec_to_base(
                        obs,
                        np.array([err_xy[0], err_xy[1], 0.0], dtype=float),
                    )[:2]
                elif action_frame == "world":
                    translate_error = world_xy_to_controller_action(env, err_xy)
                else:
                    raise ValueError(f"unsupported base action frame {action_frame!r}")
                if planner is not None:
                    translate_action = planner.xy_action(
                        translate_error, tolerance=tolerance, phase="translate"
                    )
                else:
                    translate_action = _clip_action_norm(translate_error * xy_gain)
                base_action[:2] = translate_action * float(translation_action_sign)

            if not yaw_done and base_action.shape[0] >= 3:
                if planner is not None:
                    base_action[2] = planner.yaw_action(
                        yaw_err, tolerance=yaw_tolerance, phase="yaw"
                    )
                else:
                    base_action[2] = np.clip(yaw_err * yaw_gain, -1.0, 1.0)

            obs = step_base_and_hold_arm(
                env,
                arm_qpos=arm_qpos,
                base_action=base_action,
                base_mode=1.0,
                gripper_action=gripper_action,
                post_step_callback=post_step_callback,
            )
            steps += 1
            if path_monitor is not None and path_monitor.get("enabled", False):
                checked_collision_steps += 1
                final_collision = check_base_path_collision(env, path_monitor)
                if final_collision.get("collision", False):
                    first_collision = dict(final_collision)
                    first_collision["phase"] = "after_step"
                    first_collision["checked_collision_steps"] = int(checked_collision_steps)
                    if bool(path_monitor.get("abort_on_collision", True)):
                        break

    final_xy = np.array(obs["robot0_base_pos"][:2], dtype=float)
    final_dist = float(np.linalg.norm(target_xy - final_xy))
    final_yaw_error = (
        0.0
        if target_yaw is None
        else normalize_angle(float(target_yaw) - base_yaw(obs))
    )
    held_motion = (
        {"enabled": holding_object_name is not None, "drop_ok": True}
        if kinematic
        else _held_object_motion(
            env,
            holding_object_name,
            initial_held_pos,
            holding_drop_tolerance,
        )
    )
    path_collision_free = first_collision is None and not bool(
        final_collision.get("collision", False)
    )
    holding_motion_ok = not held_motion.get("enabled", False) or bool(
        held_motion.get("drop_ok", False)
    )
    checks = {
        "xy": final_dist <= tolerance,
        "yaw": final_dist <= tolerance
        and (target_yaw is None or abs(final_yaw_error) <= yaw_tolerance),
        "path_collision_free": path_collision_free,
        "holding_motion": holding_motion_ok,
    }
    summary = {
        "target_xy": target_xy.astype(float).tolist(),
        "target_yaw": None if target_yaw is None else float(target_yaw),
        "final_xy": final_xy.astype(float).tolist(),
        "final_yaw": base_yaw(obs),
        "final_dist": final_dist,
        "final_yaw_error": final_yaw_error,
        "steps": int(steps),
        "kinematic": bool(kinematic),
        "action_frame": action_frame,
        "translation_action_sign": float(translation_action_sign),
        "velocity_profile": None if planner is None else planner.limits.summary(),
        "path_monitor": base_path_monitor_summary(path_monitor),
        "collision_path": {
            **final_collision,
            "first_collision": first_collision,
            "checked_collision_steps": int(checked_collision_steps),
        },
        "holding_motion": held_motion,
        "checks": checks,
    }
    summary["pass"] = bool(all(checks.values()))
    return obs, summary


def move_base_along_waypoints(
    env: Any,
    obs: dict[str, Any],
    *,
    waypoints_xy: list[list[float]] | np.ndarray,
    terminal_target_xy: list[float] | np.ndarray | None = None,
    target_yaw: float | None,
    arm_qpos: np.ndarray,
    gripper_action: float | None,
    tolerance: float,
    yaw_tolerance: float,
    max_steps: int,
    waypoint_tolerance: float = 0.10,
    action_frame: str = "base",
    lookahead_distance: float = 0.25,
    heading_tolerance: float = 0.35,
    terminal_yaw_distance: float = 0.25,
    terminal_heading_tolerance: float | None = None,
    xy_gain: float = 4.0,
    yaw_gain: float = 2.0,
    translation_action_sign: float = 1.0,
    stuck_window: int = 50,
    stuck_progress: float = 0.02,
    abort_on_stuck: bool = True,
    trace_stride: int = 10,
    terminal_translation_gain_scale: float = 0.5,
    terminal_translation_min_action: float = 0.35,
    terminal_translation_max_action: float = 0.65,
    single_axis_tracking: bool = True,
    terminal_dock_enabled: bool = True,
    terminal_dock_standoff: float = 0.35,
    path_monitor: dict[str, Any] | None = None,
    holding_object_name: str | None = None,
    holding_drop_tolerance: float = 0.04,
    kinematic: bool = False,
    velocity_limits: BaseVelocityLimits | None = None,
    post_step_callback: Callable[[dict[str, Any]], dict[str, Any] | None] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    waypoints = [np.array(item, dtype=float)[:2] for item in waypoints_xy]
    waypoints = [item for item in waypoints if item.size >= 2 and np.all(np.isfinite(item[:2]))]
    if not waypoints:
        raise ValueError("waypoints_xy must contain at least one finite xy waypoint")
    path_goal_xy = waypoints[-1]
    final_xy = (
        np.array(terminal_target_xy, dtype=float)[:2]
        if terminal_target_xy is not None
        else path_goal_xy
    )
    if kinematic:
        return move_base_to_pose(
            env,
            obs,
            target_xy=final_xy,
            target_yaw=target_yaw,
            arm_qpos=arm_qpos,
            gripper_action=gripper_action,
            tolerance=tolerance,
            yaw_tolerance=yaw_tolerance,
            max_steps=max_steps,
            action_frame=action_frame,
            path_monitor=path_monitor,
            holding_object_name=holding_object_name,
            holding_drop_tolerance=holding_drop_tolerance,
            kinematic=True,
        )

    planner = planner_for_env(env, velocity_limits)
    if planner is not None:
        planner.begin()
    initial_xy = np.array(obs["robot0_base_pos"][:2], dtype=float)
    initial_yaw = base_yaw(obs)
    initial_dist = float(np.linalg.norm(final_xy - initial_xy))
    min_dist = initial_dist
    initial_held_pos = _initial_held_object_pos(env, holding_object_name)
    steps = 0
    waypoint_index = 0
    checked_collision_steps = 0
    first_collision = None
    final_collision = {"enabled": False, "collision": False}
    if path_monitor is not None and path_monitor.get("enabled", False):
        checked_collision_steps += 1
        final_collision = check_base_path_collision(env, path_monitor)
        if final_collision.get("collision", False):
            first_collision = dict(final_collision)
            first_collision["phase"] = "initial"
            first_collision["checked_collision_steps"] = int(checked_collision_steps)

    progress_trace: list[dict[str, Any]] = []
    track_dist_history: list[float] = []
    stop_reason = "max_steps"
    last_window_progress = None
    last_phase = "init"
    last_heading_error = 0.0
    last_target_xy = final_xy
    last_lookahead_index = len(waypoints) - 1
    trace_stride = max(1, int(trace_stride))
    terminal_heading_tolerance = (
        float(yaw_tolerance)
        if terminal_heading_tolerance is None
        else float(terminal_heading_tolerance)
    )
    terminal_translation_gain = float(xy_gain) * float(terminal_translation_gain_scale)
    terminal_min_action = float(terminal_translation_min_action)
    terminal_max_action = float(terminal_translation_max_action)
    terminal_mode_active = False
    dock_final_stage_active = False
    dock_predock_xy = None
    if target_yaw is not None and terminal_target_xy is not None:
        dock_predock_xy = path_goal_xy.copy()
    elif (
        target_yaw is not None
        and bool(terminal_dock_enabled)
        and float(terminal_dock_standoff) > 1e-6
    ):
        dock_forward = np.array([np.cos(float(target_yaw)), np.sin(float(target_yaw))], dtype=float)
        dock_predock_xy = final_xy - dock_forward * float(terminal_dock_standoff)

    def _terminal_axis_value(
        err_value: float, *, phase_name: str, axis_tolerance: float
    ) -> float:
        if planner is not None:
            return planner.dock_axis_action(
                err_value, tolerance=axis_tolerance, phase=phase_name
            )
        return _single_axis_action(
            err_value,
            gain=terminal_translation_gain,
            max_mag=terminal_max_action,
            min_mag=terminal_min_action,
        )

    def _yaw_align_value(err_value: float, *, phase_name: str, axis_tolerance: float) -> float:
        if planner is not None:
            return planner.yaw_action(
                err_value, tolerance=axis_tolerance, phase=phase_name
            )
        return float(np.clip(err_value * yaw_gain, -1.0, 1.0))

    for _ in range(int(max_steps)):
        base_xy = np.array(obs["robot0_base_pos"][:2], dtype=float)
        yaw = base_yaw(obs)
        final_err_xy = final_xy - base_xy
        final_dist = float(np.linalg.norm(final_err_xy))
        final_yaw_error = 0.0 if target_yaw is None else normalize_angle(float(target_yaw) - yaw)
        min_dist = min(min_dist, final_dist)
        xy_done = final_dist <= float(tolerance)
        yaw_done = target_yaw is None or abs(final_yaw_error) <= float(yaw_tolerance)
        if xy_done and yaw_done:
            stop_reason = "target_reached"
            break
        if first_collision is not None and bool(
            (path_monitor or {}).get("abort_on_collision", True)
        ):
            stop_reason = "path_collision"
            break
        if path_monitor is not None and path_monitor.get("enabled", False):
            checked_collision_steps += 1
            final_collision = check_base_path_collision(env, path_monitor)
            if final_collision.get("collision", False):
                first_collision = dict(final_collision)
                first_collision["phase"] = "before_step"
                first_collision["checked_collision_steps"] = int(checked_collision_steps)
                if bool(path_monitor.get("abort_on_collision", True)):
                    stop_reason = "path_collision"
                    break
        split = env.robots[0].composite_controller._action_split_indexes
        if "base" not in split:
            raise RuntimeError("base action part not found")
        start, end = split["base"]
        base_action = np.zeros(end - start, dtype=float)
        target_xy = final_xy
        lookahead_index = len(waypoints) - 1
        heading_error = 0.0
        terminal_triggered = final_dist <= float(terminal_yaw_distance)
        if dock_predock_xy is not None:
            predock_dist = float(np.linalg.norm(dock_predock_xy - base_xy))
            terminal_triggered = (
                predock_dist <= max(float(waypoint_tolerance), 0.08)
                or final_dist <= float(tolerance)
            )

        if xy_done:
            phase = "align_final_yaw"
            if base_action.shape[0] >= 3:
                base_action[2] = _yaw_align_value(
                    final_yaw_error,
                    phase_name=phase,
                    axis_tolerance=float(yaw_tolerance),
                )
        elif target_yaw is not None and (terminal_mode_active or terminal_triggered):
            terminal_mode_active = True
            terminal_axis_tolerance = max(0.015, 0.5 * float(tolerance))
            predock_done_distance = max(terminal_axis_tolerance, float(tolerance))
            predock_dist = (
                float(np.linalg.norm(dock_predock_xy - base_xy))
                if dock_predock_xy is not None
                else 0.0
            )
            if abs(final_yaw_error) > float(terminal_heading_tolerance):
                phase = "align_terminal_yaw"
                target_xy = dock_predock_xy if dock_predock_xy is not None else final_xy
                if base_action.shape[0] >= 3:
                    base_action[2] = _yaw_align_value(
                        final_yaw_error,
                        phase_name=phase,
                        axis_tolerance=float(terminal_heading_tolerance),
                    )
            elif (
                dock_predock_xy is not None
                and not dock_final_stage_active
                and predock_dist > predock_done_distance
            ):
                target_xy = dock_predock_xy
                err_target = planar_world_vec_to_local_yaw(
                    dock_predock_xy - base_xy,
                    float(target_yaw),
                )
                if abs(err_target[1]) > terminal_axis_tolerance and base_action.shape[0] >= 2:
                    phase = "dock_correct_lateral_y"
                    _set_terminal_axis_action(
                        env,
                        base_action,
                        axis=1,
                        value=_terminal_axis_value(
                            err_target[1] * float(translation_action_sign),
                            phase_name=phase,
                            axis_tolerance=terminal_axis_tolerance,
                        ),
                        target_yaw=float(target_yaw),
                        action_frame=action_frame,
                    )
                elif abs(err_target[0]) > terminal_axis_tolerance and base_action.shape[0] >= 1:
                    phase = "dock_move_to_standoff_x"
                    _set_terminal_axis_action(
                        env,
                        base_action,
                        axis=0,
                        value=_terminal_axis_value(
                            err_target[0] * float(translation_action_sign),
                            phase_name=phase,
                            axis_tolerance=terminal_axis_tolerance,
                        ),
                        target_yaw=float(target_yaw),
                        action_frame=action_frame,
                    )
                else:
                    phase = "dock_settle_at_standoff"
            else:
                dock_final_stage_active = True
                target_xy = final_xy
                err_target = planar_world_vec_to_local_yaw(final_err_xy, float(target_yaw))
                if abs(err_target[1]) > terminal_axis_tolerance and base_action.shape[0] >= 2:
                    phase = "dock_final_lateral_y"
                    _set_terminal_axis_action(
                        env,
                        base_action,
                        axis=1,
                        value=_terminal_axis_value(
                            err_target[1] * float(translation_action_sign),
                            phase_name=phase,
                            axis_tolerance=terminal_axis_tolerance,
                        ),
                        target_yaw=float(target_yaw),
                        action_frame=action_frame,
                    )
                elif abs(err_target[0]) > terminal_axis_tolerance and base_action.shape[0] >= 1:
                    phase = "dock_approach_x"
                    _set_terminal_axis_action(
                        env,
                        base_action,
                        axis=0,
                        value=_terminal_axis_value(
                            err_target[0] * float(translation_action_sign),
                            phase_name=phase,
                            axis_tolerance=terminal_axis_tolerance,
                        ),
                        target_yaw=float(target_yaw),
                        action_frame=action_frame,
                    )
                else:
                    phase = "align_final_yaw"
                    if base_action.shape[0] >= 3:
                        base_action[2] = _yaw_align_value(
                            final_yaw_error,
                            phase_name=phase,
                            axis_tolerance=float(yaw_tolerance),
                        )
        else:
            if waypoint_index < len(waypoints) - 1:
                remaining = waypoints[waypoint_index:]
                remaining_dists = [float(np.linalg.norm(item - base_xy)) for item in remaining]
                nearest_offset = int(np.argmin(remaining_dists))
                nearest_index = waypoint_index + nearest_offset
                if (
                    nearest_index > waypoint_index
                    and remaining_dists[nearest_offset]
                    <= max(float(waypoint_tolerance), float(lookahead_distance))
                ):
                    waypoint_index = nearest_index
            while (
                waypoint_index < len(waypoints) - 1
                and float(np.linalg.norm(waypoints[waypoint_index] - base_xy))
                <= float(waypoint_tolerance)
            ):
                waypoint_index += 1
            for index in range(waypoint_index, len(waypoints)):
                candidate_xy = waypoints[index]
                if (
                    float(np.linalg.norm(candidate_xy - base_xy)) >= float(lookahead_distance)
                    or index == len(waypoints) - 1
                ):
                    target_xy = candidate_xy
                    lookahead_index = index
                    break
            # Pure-pursuit lookahead is forward progress: once a later waypoint
            # becomes the active target, never select an already-passed point on
            # a subsequent step. Without this, sparse paths can oscillate between
            # the first two points after moving more than one lookahead radius.
            waypoint_index = max(waypoint_index, lookahead_index)

            target_delta_world = target_xy - base_xy
            desired_heading = float(np.arctan2(target_delta_world[1], target_delta_world[0]))
            heading_error = normalize_angle(desired_heading - yaw)
            if abs(heading_error) > float(heading_tolerance):
                phase = "align_to_path"
                if base_action.shape[0] >= 3:
                    base_action[2] = _yaw_align_value(
                        heading_error,
                        phase_name=phase,
                        axis_tolerance=float(heading_tolerance),
                    )
            else:
                phase = "track_waypoint"
                if action_frame == "base":
                    translate_error = world_vec_to_base(
                        obs,
                        np.array([target_delta_world[0], target_delta_world[1], 0.0], dtype=float),
                    )[:2]
                elif action_frame == "world":
                    translate_error = world_xy_to_controller_action(env, target_delta_world)
                else:
                    raise ValueError(f"unsupported base action frame {action_frame!r}")
                if planner is not None:
                    # 用 max(到lookahead点距离, 到终点距离) 规划速度，
                    # 避免 lookahead 点近导致中段无谓减速。
                    track_dist = float(np.linalg.norm(translate_error))
                    profile_dist = max(track_dist, final_dist)
                    profile_error = (
                        translate_error * (profile_dist / track_dist)
                        if track_dist > 1e-9
                        else translate_error
                    )
                    translate_action = (
                        planner.xy_action(
                            profile_error,
                            tolerance=float(tolerance),
                            phase="track_waypoint",
                            single_axis=bool(single_axis_tracking),
                        )
                        * float(translation_action_sign)
                    )
                else:
                    translate_action = _clip_action_norm(
                        translate_error * xy_gain * float(translation_action_sign),
                        max_norm=1.0,
                    )
                if bool(single_axis_tracking):
                    axis = int(np.argmax(np.abs(translate_action[:2])))
                    base_action[:2] = 0.0
                    base_action[axis] = float(translate_action[axis])
                    phase = "track_waypoint_x" if axis == 0 else "track_waypoint_y"
                else:
                    base_action[:2] = translate_action

        pre_xy = base_xy
        track_progress_xy = target_xy.copy()
        obs = step_base_and_hold_arm(
            env,
            arm_qpos=arm_qpos,
            base_action=base_action,
            base_mode=1.0,
            gripper_action=gripper_action,
            post_step_callback=post_step_callback,
        )
        steps += 1
        post_xy = np.array(obs["robot0_base_pos"][:2], dtype=float)
        post_yaw = base_yaw(obs)
        post_final_dist = float(np.linalg.norm(final_xy - post_xy))
        post_track_dist = float(np.linalg.norm(track_progress_xy - post_xy))
        min_dist = min(min_dist, post_final_dist)
        if path_monitor is not None and path_monitor.get("enabled", False):
            checked_collision_steps += 1
            final_collision = check_base_path_collision(env, path_monitor)
            if final_collision.get("collision", False):
                first_collision = dict(final_collision)
                first_collision["phase"] = "after_step"
                first_collision["checked_collision_steps"] = int(checked_collision_steps)
                if bool(path_monitor.get("abort_on_collision", True)):
                    stop_reason = "path_collision"
                    break
        if phase.startswith("track_waypoint"):
            track_dist_history.append(post_track_dist)
            if len(track_dist_history) > max(1, int(stuck_window)):
                previous_dist = track_dist_history[-max(1, int(stuck_window))]
                best_recent = min(track_dist_history[-max(1, int(stuck_window)) :])
                last_window_progress = float(previous_dist - best_recent)
                if (
                    post_final_dist > float(tolerance)
                    and last_window_progress < float(stuck_progress)
                ):
                    last_phase = phase
                    last_heading_error = heading_error
                    last_target_xy = target_xy
                    last_lookahead_index = lookahead_index
                    if bool(abort_on_stuck):
                        stop_reason = "stuck_no_progress"
                        break
        else:
            track_dist_history.clear()
        if steps <= 5 or steps % trace_stride == 0:
            progress_trace.append(
                {
                    "step": int(steps),
                    "phase": phase,
                    "xy": post_xy.astype(float).tolist(),
                    "yaw": float(post_yaw),
                    "final_dist": post_final_dist,
                    "track_dist": post_track_dist,
                    "target_xy": target_xy.astype(float).tolist(),
                    "waypoint_index": int(waypoint_index),
                    "lookahead_index": int(lookahead_index),
                    "heading_error": float(heading_error),
                    "final_yaw_error": 0.0
                    if target_yaw is None
                    else normalize_angle(float(target_yaw) - post_yaw),
                    "base_action": base_action.astype(float).tolist(),
                    "delta_xy_world": (post_xy - pre_xy).astype(float).tolist(),
                    "dock_predock_xy": None
                    if dock_predock_xy is None
                    else dock_predock_xy.astype(float).tolist(),
                    "dock_final_stage_active": bool(dock_final_stage_active),
                    **({} if planner is None else planner.last_speeds()),
                }
            )
        last_phase = phase
        last_heading_error = heading_error
        last_target_xy = target_xy
        last_lookahead_index = lookahead_index

    final_base_xy = np.array(obs["robot0_base_pos"][:2], dtype=float)
    final_dist = float(np.linalg.norm(final_xy - final_base_xy))
    final_yaw_error = (
        0.0
        if target_yaw is None
        else normalize_angle(float(target_yaw) - base_yaw(obs))
    )
    held_motion = _held_object_motion(
        env,
        holding_object_name,
        initial_held_pos,
        holding_drop_tolerance,
    )
    path_collision_free = first_collision is None and not bool(
        final_collision.get("collision", False)
    )
    holding_motion_ok = not held_motion.get("enabled", False) or bool(
        held_motion.get("drop_ok", False)
    )
    checks = {
        "xy": final_dist <= tolerance,
        "yaw": target_yaw is None or abs(final_yaw_error) <= yaw_tolerance,
        "path_collision_free": path_collision_free,
        "holding_motion": holding_motion_ok,
    }
    summary = {
        "target_xy": final_xy.astype(float).tolist(),
        "path_goal_xy": path_goal_xy.astype(float).tolist(),
        "terminal_target_xy": None
        if terminal_target_xy is None
        else np.array(terminal_target_xy, dtype=float)[:2].astype(float).tolist(),
        "target_yaw": None if target_yaw is None else float(target_yaw),
        "initial_xy": initial_xy.astype(float).tolist(),
        "initial_yaw": float(initial_yaw),
        "initial_dist": float(initial_dist),
        "final_xy": final_base_xy.astype(float).tolist(),
        "final_yaw": base_yaw(obs),
        "final_dist": final_dist,
        "final_yaw_error": final_yaw_error,
        "min_dist": float(min_dist),
        "dist_reduction": float(initial_dist - final_dist),
        "steps": int(steps),
        "action_frame": action_frame,
        "translation_action_sign": float(translation_action_sign),
        "waypoint_follower": True,
        "follower": "staged_lookahead",
        "velocity_profile": None if planner is None else planner.limits.summary(),
        "lookahead_distance": float(lookahead_distance),
        "heading_tolerance": float(heading_tolerance),
        "terminal_yaw_distance": float(terminal_yaw_distance),
        "terminal_heading_tolerance": float(terminal_heading_tolerance),
        "terminal_translation_gain_scale": float(terminal_translation_gain_scale),
        "terminal_translation_min_action": float(terminal_min_action),
        "terminal_translation_max_action": float(terminal_max_action),
        "single_axis_tracking": bool(single_axis_tracking),
        "abort_on_stuck": bool(abort_on_stuck),
        "terminal_dock_enabled": bool(terminal_dock_enabled),
        "terminal_dock_standoff": float(terminal_dock_standoff),
        "dock_predock_xy": None
        if dock_predock_xy is None
        else dock_predock_xy.astype(float).tolist(),
        "waypoint_tolerance": float(waypoint_tolerance),
        "waypoint_count": int(len(waypoints)),
        "waypoints_xy": [item.astype(float).tolist() for item in waypoints],
        "last_phase": last_phase,
        "last_heading_error": float(last_heading_error),
        "last_target_xy": last_target_xy.astype(float).tolist(),
        "last_lookahead_index": int(last_lookahead_index),
        "stop_reason": stop_reason,
        "last_window_progress": last_window_progress,
        "terminal_mode_active": bool(terminal_mode_active),
        "dock_final_stage_active": bool(dock_final_stage_active),
        "progress_trace": progress_trace,
        "path_monitor": base_path_monitor_summary(path_monitor),
        "collision_path": {
            **final_collision,
            "first_collision": first_collision,
            "checked_collision_steps": int(checked_collision_steps),
        },
        "holding_motion": held_motion,
        "checks": checks,
    }
    summary["pass"] = bool(all(checks.values()))
    return obs, summary


__all__ = [
    "action_dict_for_arm_and_gripper",
    "base_path_monitor_summary",
    "base_yaw",
    "check_base_path_collision",
    "controller_translation_yaw",
    "move_base_along_waypoints",
    "move_base_to_pose",
    "qpos_for_controller",
    "set_base_pose_kinematic",
    "step_base_and_hold_arm",
    "sync_arm_controller_to_current_qpos",
    "world_xy_to_controller_action",
    "world_vec_to_base",
]
