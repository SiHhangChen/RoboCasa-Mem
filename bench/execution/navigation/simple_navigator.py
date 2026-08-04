"""极简导航控制器 - 转向和平移完全解耦

设计原则：
1. 一次只做一个动作（转向 OR 平移）
2. 先转向到目标方向，再平移到目标位置
3. 不使用A*或任何复杂规划
4. 固定点位之间的导航通过预定义waypoints完成
"""

from __future__ import annotations

from typing import Any

import numpy as np

from bench.execution.navigation.velocity_profile import (
    AxialVelocityProfile,
    BaseVelocityLimits,
    PlanarVelocityProfile,
    resolve_control_dt,
)
from bench.execution.planning.nav_algo import normalize_angle


class SimpleNavigator:
    """极简导航器 - 转向和平移严格分离"""

    def __init__(self, env: Any, velocity_limits: BaseVelocityLimits | None = None):
        self.env = env
        self.robot = env.robots[0]
        self.velocity_limits = (
            velocity_limits if velocity_limits is not None and velocity_limits.enabled else None
        )
        self._control_dt = resolve_control_dt(env)

    def navigate_to_pose(
        self,
        obs: dict[str, Any],
        target_xy: tuple[float, float] | np.ndarray,
        target_yaw: float,
        *,
        arm_qpos: np.ndarray,
        gripper_action: float = -1.0,
        holding_object: str | None = None,
        xy_tolerance: float = 0.06,
        yaw_tolerance: float = 0.10,
        max_steps_rotate: int = 300,
        max_steps_translate: int = 500,
        action_frame: str = "world",
        holding_drop_tolerance: float = 0.04,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """导航到目标位姿 - 严格按照 旋转→平移→旋转 的顺序

        步骤：
        1. 旋转：转向目标点方向
        2. 平移：直线移动到目标点
        3. 旋转：调整到最终目标朝向

        Args:
            obs: 当前观测
            target_xy: 目标xy位置
            target_yaw: 目标yaw角度
            arm_qpos: 手臂关节位置
            gripper_action: 夹爪动作
            holding_object: 持有的物体名称（用于检测掉落）
            xy_tolerance: xy位置容忍度（米）
            yaw_tolerance: yaw角度容忍度（弧度）
            max_steps_rotate: 旋转最大步数
            max_steps_translate: 平移最大步数

        Returns:
            (final_obs, summary_dict)
        """
        target_xy = np.array(target_xy, dtype=float)[:2]
        target_yaw = float(target_yaw)

        start_xy = np.array(obs["robot0_base_pos"][:2], dtype=float)
        start_yaw = self._get_yaw(obs)

        phases = []

        # === 阶段1: 旋转到目标点方向 ===
        direction_to_target = target_xy - start_xy
        distance_to_target = float(np.linalg.norm(direction_to_target))

        if distance_to_target > xy_tolerance:
            # 需要移动，先转向目标点
            approach_yaw = float(np.arctan2(direction_to_target[1], direction_to_target[0]))

            obs, phase1 = self._rotate_to_yaw(
                obs,
                target_yaw=approach_yaw,
                arm_qpos=arm_qpos,
                gripper_action=gripper_action,
                holding_object=holding_object,
                yaw_tolerance=yaw_tolerance,
                max_steps=max_steps_rotate,
                holding_drop_tolerance=holding_drop_tolerance,
            )
            phase1["phase"] = "rotate_to_approach_direction"
            phases.append(phase1)

            if not phase1["pass"]:
                return obs, {
                    "primitive": "simple_navigate_to_pose",
                    "pass": False,
                    "reason": "rotate_to_approach_failed",
                    "start_xy": start_xy.tolist(),
                    "start_yaw": start_yaw,
                    "target_xy": target_xy.tolist(),
                    "target_yaw": target_yaw,
                    "phases": phases,
                }

            # === 阶段2: 直线平移到目标点 ===
            obs, phase2 = self._translate_to_xy(
                obs,
                target_xy=target_xy,
                arm_qpos=arm_qpos,
                gripper_action=gripper_action,
                holding_object=holding_object,
                xy_tolerance=xy_tolerance,
                max_steps=max_steps_translate,
                action_frame=action_frame,
                holding_drop_tolerance=holding_drop_tolerance,
            )
            phase2["phase"] = "translate_to_target"
            phases.append(phase2)

            if not phase2["pass"]:
                return obs, {
                    "primitive": "simple_navigate_to_pose",
                    "pass": False,
                    "reason": "translate_to_target_failed",
                    "start_xy": start_xy.tolist(),
                    "start_yaw": start_yaw,
                    "target_xy": target_xy.tolist(),
                    "target_yaw": target_yaw,
                    "phases": phases,
                }

        # === 阶段3: 旋转到最终目标朝向 ===
        obs, phase3 = self._rotate_to_yaw(
            obs,
            target_yaw=target_yaw,
            arm_qpos=arm_qpos,
            gripper_action=gripper_action,
            holding_object=holding_object,
            yaw_tolerance=yaw_tolerance,
            max_steps=max_steps_rotate,
            holding_drop_tolerance=holding_drop_tolerance,
        )
        phase3["phase"] = "rotate_to_final_yaw"
        phases.append(phase3)

        final_xy = np.array(obs["robot0_base_pos"][:2], dtype=float)
        final_yaw = self._get_yaw(obs)
        final_xy_error = float(np.linalg.norm(target_xy - final_xy))
        final_yaw_error = abs(normalize_angle(target_yaw - final_yaw))

        success = (
            final_xy_error <= xy_tolerance
            and final_yaw_error <= yaw_tolerance
            and all(p["pass"] for p in phases)
        )

        return obs, {
            "primitive": "simple_navigate_to_pose",
            "pass": success,
            "start_xy": start_xy.tolist(),
            "start_yaw": start_yaw,
            "target_xy": target_xy.tolist(),
            "target_yaw": target_yaw,
            "final_xy": final_xy.tolist(),
            "final_yaw": final_yaw,
            "final_xy_error": final_xy_error,
            "final_yaw_error": final_yaw_error,
            "action_frame": action_frame,
            "holding_drop_tolerance": float(holding_drop_tolerance),
            "phases": phases,
            "total_steps": sum(p["steps"] for p in phases),
        }

    def navigate_right_turn_corridor(
        self,
        obs: dict[str, Any],
        target_xy: tuple[float, float] | np.ndarray,
        target_yaw: float,
        *,
        arm_qpos: np.ndarray,
        gripper_action: float = -1.0,
        holding_object: str | None = None,
        xy_tolerance: float = 0.06,
        yaw_tolerance: float = 0.10,
        max_steps_rotate: int = 300,
        max_steps_translate: int = 900,
        action_frame: str = "world",
        right_turn_deg: float = 90.0,
        holding_drop_tolerance: float = 0.04,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Navigate through a right-turn corridor with decoupled primitives."""
        target_xy = np.array(target_xy, dtype=float)[:2]
        target_yaw = float(target_yaw)
        start_xy = np.array(obs["robot0_base_pos"][:2], dtype=float)
        start_yaw = self._get_yaw(obs)
        first_yaw = normalize_angle(start_yaw - np.deg2rad(float(right_turn_deg)))
        approach_yaw = normalize_angle(target_yaw)
        corner_xy, corner_summary = _right_turn_corridor_corner(
            start_xy=start_xy,
            first_yaw=first_yaw,
            target_xy=target_xy,
            approach_yaw=approach_yaw,
        )

        phases = []
        current_obs = obs
        phase_specs = [
            (
                "right_turn_from_start",
                "rotate",
                first_yaw,
                max_steps_rotate,
            ),
            (
                "straight_corridor",
                "translate",
                corner_xy,
                max_steps_translate,
            ),
            (
                "right_turn_to_approach",
                "rotate",
                approach_yaw,
                max_steps_rotate,
            ),
            (
                "straight_to_target",
                "translate",
                target_xy,
                max_steps_translate,
            ),
            (
                "align_final_yaw",
                "rotate",
                target_yaw,
                max_steps_rotate,
            ),
        ]
        for phase_name, phase_kind, phase_target, max_steps in phase_specs:
            if phase_kind == "rotate":
                current_obs, phase = self._rotate_to_yaw(
                    current_obs,
                    target_yaw=float(phase_target),
                    arm_qpos=arm_qpos,
                    gripper_action=gripper_action,
                    holding_object=holding_object,
                    yaw_tolerance=yaw_tolerance,
                    max_steps=int(max_steps),
                    holding_drop_tolerance=holding_drop_tolerance,
                )
            else:
                current_obs, phase = self._translate_to_xy(
                    current_obs,
                    target_xy=np.array(phase_target, dtype=float)[:2],
                    arm_qpos=arm_qpos,
                    gripper_action=gripper_action,
                    holding_object=holding_object,
                    xy_tolerance=xy_tolerance,
                    max_steps=int(max_steps),
                    action_frame=action_frame,
                    holding_drop_tolerance=holding_drop_tolerance,
                )
            phase["phase"] = phase_name
            phases.append(phase)
            if not bool(phase.get("pass", False)):
                break

        final_xy = np.array(current_obs["robot0_base_pos"][:2], dtype=float)
        final_yaw = self._get_yaw(current_obs)
        final_xy_error = float(np.linalg.norm(target_xy - final_xy))
        final_yaw_error = abs(normalize_angle(target_yaw - final_yaw))
        success = (
            final_xy_error <= xy_tolerance
            and final_yaw_error <= yaw_tolerance
            and len(phases) == len(phase_specs)
            and all(bool(p.get("pass", False)) for p in phases)
        )
        return current_obs, {
            "primitive": "simple_right_turn_corridor",
            "pass": bool(success),
            "start_xy": start_xy.astype(float).tolist(),
            "start_yaw": float(start_yaw),
            "target_xy": target_xy.astype(float).tolist(),
            "target_yaw": float(target_yaw),
            "final_xy": final_xy.astype(float).tolist(),
            "final_yaw": float(final_yaw),
            "final_xy_error": final_xy_error,
            "final_yaw_error": final_yaw_error,
            "first_yaw": float(first_yaw),
            "corner_xy": corner_xy.astype(float).tolist(),
            "corner": corner_summary,
            "action_frame": action_frame,
            "right_turn_deg": float(right_turn_deg),
            "holding_drop_tolerance": float(holding_drop_tolerance),
            "phases": phases,
            "total_steps": sum(int(p.get("steps", 0)) for p in phases),
        }

    def _rotate_to_yaw(
        self,
        obs: dict[str, Any],
        target_yaw: float,
        arm_qpos: np.ndarray,
        gripper_action: float,
        holding_object: str | None,
        yaw_tolerance: float,
        max_steps: int,
        holding_drop_tolerance: float,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """纯旋转 - 完全不执行xy平移"""
        from bench.execution.navigation.base_motion import base_yaw, step_base_and_hold_arm

        initial_xy = np.array(obs["robot0_base_pos"][:2], dtype=float)
        initial_yaw = float(base_yaw(obs))
        initial_held_pos = self._get_held_pos(holding_object)

        steps = 0
        max_drift = 0.0
        yaw_gain = 2.0
        yaw_profile = (
            AxialVelocityProfile(self.velocity_limits.yaw_params(), self._control_dt)
            if self.velocity_limits is not None
            else None
        )

        for _ in range(max_steps):
            current_yaw = base_yaw(obs)
            yaw_error = normalize_angle(target_yaw - current_yaw)

            if abs(yaw_error) <= yaw_tolerance:
                break

            # 构造动作：只有yaw，xy为0
            split = self.robot.composite_controller._action_split_indexes
            start, end = split["base"]
            base_action = np.zeros(end - start, dtype=float)

            # 只控制yaw
            if base_action.shape[0] >= 3:
                if yaw_profile is not None:
                    base_action[2] = yaw_profile.step_action_signed(
                        yaw_error, tolerance=yaw_tolerance
                    )
                else:
                    base_action[2] = np.clip(yaw_error * yaw_gain, -1.0, 1.0)

            obs = step_base_and_hold_arm(
                self.env,
                arm_qpos=arm_qpos,
                base_action=base_action,
                base_mode=1.0,
                gripper_action=gripper_action,
            )
            steps += 1

            # 监控xy漂移
            current_xy = np.array(obs["robot0_base_pos"][:2], dtype=float)
            drift = float(np.linalg.norm(current_xy - initial_xy))
            max_drift = max(max_drift, drift)

        final_xy = np.array(obs["robot0_base_pos"][:2], dtype=float)
        final_yaw = float(base_yaw(obs))
        final_yaw_error = abs(normalize_angle(target_yaw - final_yaw))
        final_drift = float(np.linalg.norm(final_xy - initial_xy))

        held_check = self._check_holding(
            holding_object,
            initial_held_pos,
            drop_tolerance=holding_drop_tolerance,
        )

        return obs, {
            "action": "rotate",
            "pass": final_yaw_error <= yaw_tolerance and held_check["pass"],
            "initial_xy": initial_xy.tolist(),
            "initial_yaw": initial_yaw,
            "target_yaw": target_yaw,
            "final_xy": final_xy.tolist(),
            "final_yaw": final_yaw,
            "final_yaw_error": final_yaw_error,
            "drift": final_drift,
            "max_drift": max_drift,
            "steps": steps,
            "holding_check": held_check,
        }

    def _translate_to_xy(
        self,
        obs: dict[str, Any],
        target_xy: np.ndarray,
        arm_qpos: np.ndarray,
        gripper_action: float,
        holding_object: str | None,
        xy_tolerance: float,
        max_steps: int,
        action_frame: str,
        holding_drop_tolerance: float,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """纯平移 - 完全不执行yaw旋转"""
        from bench.execution.navigation.base_motion import (
            base_yaw,
            step_base_and_hold_arm,
            world_xy_to_controller_action,
            world_vec_to_base,
        )

        target_xy = np.array(target_xy, dtype=float)[:2]
        initial_xy = np.array(obs["robot0_base_pos"][:2], dtype=float)
        initial_yaw = float(base_yaw(obs))
        initial_held_pos = self._get_held_pos(holding_object)

        steps = 0
        min_dist = float("inf")
        xy_gain = 3.0
        max_yaw_drift = 0.0
        xy_profile = (
            PlanarVelocityProfile(self.velocity_limits.xy_params(), self._control_dt)
            if self.velocity_limits is not None
            else None
        )

        for _ in range(max_steps):
            current_xy = np.array(obs["robot0_base_pos"][:2], dtype=float)
            current_yaw = base_yaw(obs)

            err_xy = target_xy - current_xy
            dist = float(np.linalg.norm(err_xy))
            min_dist = min(min_dist, dist)

            if dist <= xy_tolerance:
                break

            # 构造动作：只有xy，yaw为0
            split = self.robot.composite_controller._action_split_indexes
            start, end = split["base"]
            base_action = np.zeros(end - start, dtype=float)

            if action_frame == "base":
                translate_error = world_vec_to_base(
                    obs,
                    np.array([err_xy[0], err_xy[1], 0.0], dtype=float),
                )[:2]
            elif action_frame == "world":
                translate_error = world_xy_to_controller_action(self.env, err_xy)
            else:
                raise ValueError(f"unsupported base action frame {action_frame!r}")

            # 只控制xy
            if xy_profile is not None:
                translate_action = xy_profile.step_action(
                    translate_error, tolerance=xy_tolerance
                )
            else:
                translate_action = _clip_action_norm(translate_error * xy_gain)
            base_action[:2] = translate_action

            # yaw保持为0 - 不主动控制yaw
            # 这允许base在移动时自然漂移，但不会主动调整

            obs = step_base_and_hold_arm(
                self.env,
                arm_qpos=arm_qpos,
                base_action=base_action,
                base_mode=1.0,
                gripper_action=gripper_action,
            )
            steps += 1

            # 监控yaw漂移
            yaw_drift = abs(normalize_angle(current_yaw - initial_yaw))
            max_yaw_drift = max(max_yaw_drift, yaw_drift)

        final_xy = np.array(obs["robot0_base_pos"][:2], dtype=float)
        final_yaw = float(base_yaw(obs))
        final_dist = float(np.linalg.norm(target_xy - final_xy))
        final_yaw_drift = abs(normalize_angle(final_yaw - initial_yaw))

        held_check = self._check_holding(
            holding_object,
            initial_held_pos,
            drop_tolerance=holding_drop_tolerance,
        )

        return obs, {
            "action": "translate",
            "pass": final_dist <= xy_tolerance and held_check["pass"],
            "initial_xy": initial_xy.tolist(),
            "initial_yaw": initial_yaw,
            "target_xy": target_xy.tolist(),
            "final_xy": final_xy.tolist(),
            "final_yaw": final_yaw,
            "final_dist": final_dist,
            "min_dist": min_dist,
            "yaw_drift": final_yaw_drift,
            "max_yaw_drift": max_yaw_drift,
            "action_frame": action_frame,
            "steps": steps,
            "holding_check": held_check,
        }

    def _get_yaw(self, obs: dict[str, Any]) -> float:
        """获取当前yaw角度"""
        from bench.execution.navigation.base_motion import base_yaw
        return float(base_yaw(obs))

    def _get_held_pos(self, object_name: str | None) -> np.ndarray | None:
        """获取持有物体的初始位置"""
        if object_name is None:
            return None
        try:
            obj_id = getattr(self.env, "obj_body_id", {}).get(object_name)
            if obj_id is None:
                obj_id = self.env.sim.model.body_name2id(object_name)
            return np.array(self.env.sim.data.body_xpos[obj_id], dtype=float)
        except Exception:
            return None

    def _check_holding(
        self,
        object_name: str | None,
        initial_pos: np.ndarray | None,
        drop_tolerance: float = 0.04,
    ) -> dict[str, Any]:
        """检查持有物体是否掉落"""
        if object_name is None or initial_pos is None:
            return {"enabled": False, "pass": True}

        try:
            obj_id = getattr(self.env, "obj_body_id", {}).get(object_name)
            if obj_id is None:
                obj_id = self.env.sim.model.body_name2id(object_name)
            final_pos = np.array(self.env.sim.data.body_xpos[obj_id], dtype=float)
            delta = final_pos - initial_pos
            z_delta = float(delta[2])
            drop_ok = z_delta >= -float(drop_tolerance)
            return {
                "enabled": True,
                "object_name": object_name,
                "initial_pos": initial_pos.astype(float).tolist(),
                "final_pos": final_pos.astype(float).tolist(),
                "delta": delta.astype(float).tolist(),
                "xy_delta_norm": float(np.linalg.norm(delta[:2])),
                "z_delta": z_delta,
                "drop_ok": bool(drop_ok),
                "pass": bool(drop_ok),
                "drop_tolerance": drop_tolerance,
            }
        except Exception as e:
            return {
                "enabled": True,
                "object_name": object_name,
                "pass": False,
                "error": str(e),
            }


# === 便捷函数接口 ===


def simple_navigate_to_pose(
    env: Any,
    obs: dict[str, Any],
    target_xy: tuple[float, float] | np.ndarray,
    target_yaw: float,
    arm_qpos: np.ndarray,
    gripper_action: float = -1.0,
    **kwargs: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """便捷函数：极简导航到目标位姿

    完全解耦的转向和平移控制。
    """
    navigator = SimpleNavigator(env)
    return navigator.navigate_to_pose(
        obs,
        target_xy,
        target_yaw,
        arm_qpos=arm_qpos,
        gripper_action=gripper_action,
        **kwargs,
    )


def _clip_action_norm(values: np.ndarray, *, max_norm: float = 1.0) -> np.ndarray:
    values = np.array(values, dtype=float)
    norm = float(np.linalg.norm(values))
    if norm > float(max_norm) > 0.0:
        return values / norm * float(max_norm)
    return values


def _right_turn_corridor_corner(
    *,
    start_xy: np.ndarray,
    first_yaw: float,
    target_xy: np.ndarray,
    approach_yaw: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    start_xy = np.array(start_xy, dtype=float)[:2]
    target_xy = np.array(target_xy, dtype=float)[:2]
    first_dir = np.array([np.cos(first_yaw), np.sin(first_yaw)], dtype=float)
    approach_dir = np.array([np.cos(approach_yaw), np.sin(approach_yaw)], dtype=float)
    system = np.column_stack([first_dir, approach_dir])
    delta = target_xy - start_xy
    try:
        distance_first, distance_approach = np.linalg.solve(system, delta)
    except np.linalg.LinAlgError:
        distance_first = np.nan
        distance_approach = np.nan

    if (
        np.isfinite(distance_first)
        and np.isfinite(distance_approach)
        and distance_first > 0.0
        and distance_approach > 0.0
    ):
        corner_xy = start_xy + first_dir * float(distance_first)
        strategy = "line_intersection"
    else:
        corner_xy = np.array([start_xy[0], target_xy[1]], dtype=float)
        strategy = "axis_aligned_fallback"

    return corner_xy, {
        "strategy": strategy,
        "first_dir": first_dir.astype(float).tolist(),
        "approach_dir": approach_dir.astype(float).tolist(),
        "distance_first": None if not np.isfinite(distance_first) else float(distance_first),
        "distance_approach": (
            None if not np.isfinite(distance_approach) else float(distance_approach)
        ),
    }
