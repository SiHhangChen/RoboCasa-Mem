"""Arm and gripper execution drivers for manipulation primitives.

This is the bench-owned replacement for the previous joint-position executor. It
contains env.step code but does not import task implementations.
"""

from __future__ import annotations

from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np


PostStepCallback = Callable[[dict[str, Any]], dict[str, Any] | None]
TrajectoryStopCheck = Callable[[], dict[str, Any] | bool]


class ArmDriver(Protocol):
    def execute_trajectory(
        self,
        trajectory: list[list[float]],
        *,
        max_joint_step: float,
        settle_steps: int,
        gripper_action: float = 0.0,
        collision_monitor: dict[str, Any] | None = None,
        post_step_callback: PostStepCallback | None = None,
        trajectory_stop_check: TrajectoryStopCheck | None = None,
    ) -> dict[str, Any]:
        ...

    def run_gripper(
        self,
        *,
        arm_qpos: np.ndarray,
        gripper_action: float,
        steps: int,
        post_step_callback: PostStepCallback | None = None,
    ) -> dict[str, Any]:
        ...


@contextmanager
def _installed_post_step_observation_hook(
    env: Any,
    callback: PostStepCallback | None,
):
    if callback is None:
        yield
        return
    previous_hook = getattr(env, "_bench_post_step_observation_hook", None)
    previous_consumed = getattr(
        env,
        "_bench_post_step_observation_hook_consumed",
        False,
    )
    env._bench_post_step_observation_hook = callback
    env._bench_post_step_observation_hook_consumed = False
    try:
        yield
    finally:
        if previous_hook is None:
            try:
                delattr(env, "_bench_post_step_observation_hook")
            except AttributeError:
                pass
        else:
            env._bench_post_step_observation_hook = previous_hook
        env._bench_post_step_observation_hook_consumed = previous_consumed


def _apply_post_step_callback(
    env: Any,
    obs: dict[str, Any],
    callback: PostStepCallback | None,
) -> dict[str, Any]:
    if callback is None:
        return obs
    if bool(getattr(env, "_bench_post_step_observation_hook_consumed", False)):
        env._bench_post_step_observation_hook_consumed = False
        return obs
    updated = callback(obs)
    return obs if updated is None else updated


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


def step_arm_and_gripper(
    env: Any,
    *,
    arm_qpos: np.ndarray,
    gripper_action: float | None = None,
) -> dict[str, Any]:
    robot = env.robots[0]
    action_dict = action_dict_for_arm_and_gripper(
        env,
        arm_qpos=np.array(arm_qpos, dtype=float),
        gripper_action=gripper_action,
    )
    action = robot.create_action_vector(action_dict)
    if not np.all(np.isfinite(action)):
        raise RuntimeError("action contains NaN or Inf")
    obs, _, _, _ = env.step(action)
    return obs


def collision_monitor_summary(monitor: dict[str, Any] | None) -> dict[str, Any] | None:
    from bench.execution.planning.collision_monitor import collision_monitor_summary as _summary

    return _summary(monitor)


def check_monitor_collision(env: Any, monitor: dict[str, Any] | None) -> dict[str, Any]:
    if monitor is None or not monitor.get("enabled", False):
        return {"enabled": False, "collision": False}
    check_fn = monitor.get("check") or monitor.get("check_fn")
    if callable(check_fn):
        result = dict(check_fn(env, monitor))
        result["enabled"] = True
        result.setdefault("collision", False)
        return result
    if "robot_geoms" in monitor or "obstacle_geoms" in monitor:
        from bench.execution.planning.collision_monitor import check_monitor_collision as _check

        return _check(env, monitor)
    return {
        "enabled": True,
        "collision": False,
        "reason": "monitor_check_unavailable",
    }


def _collision_abort_result(
    env: Any,
    right: Any,
    *,
    obs: dict[str, Any] | None,
    executed_steps: int,
    max_command_delta: float,
    collision_monitor: dict[str, Any],
    collision_summary: dict[str, Any] | None,
    collision: dict[str, Any],
    first_collision: dict[str, Any] | None,
) -> dict[str, Any]:
    final_qpos = qpos_for_controller(env, right)
    return {
        "obs": obs if obs is not None else env._get_observations(force_update=True),
        "executed_steps": int(executed_steps),
        "max_command_delta": float(max_command_delta),
        "final_qpos_error": float("inf"),
        "final_arm_qpos": final_qpos.astype(float).tolist(),
        "collision_monitor": collision_summary,
        "collision": {
            **collision,
            "first_collision": first_collision,
        },
        "aborted_on_collision": bool(collision_monitor.get("abort_on_collision", True)),
    }


def execute_arm_trajectory(
    env: Any,
    trajectory: list[list[float]],
    *,
    max_joint_step: float,
    settle_steps: int,
    gripper_action: float = 0.0,
    collision_monitor: dict[str, Any] | None = None,
    post_step_callback: PostStepCallback | None = None,
    trajectory_stop_check: TrajectoryStopCheck | None = None,
) -> dict[str, Any]:
    if not trajectory:
        raise ValueError("trajectory must contain at least one waypoint")
    if max_joint_step <= 0.0:
        raise ValueError("max_joint_step must be positive")
    if settle_steps < 0:
        raise ValueError("settle_steps must be non-negative")

    right = env.robots[0].composite_controller.part_controllers["right"]
    current = qpos_for_controller(env, right)
    obs = None
    executed_steps = 0
    max_command_delta = 0.0
    collision_summary = collision_monitor_summary(collision_monitor)
    checked_collision_steps = 0
    first_collision = None
    early_stop = None

    def maybe_stop_trajectory(phase: str) -> dict[str, Any] | None:
        if trajectory_stop_check is None:
            return None
        try:
            raw = trajectory_stop_check()
            result = dict(raw) if isinstance(raw, dict) else {"stop": bool(raw)}
        except Exception as exc:
            return {
                "stop": True,
                "reason": f"trajectory_stop_check_error:{type(exc).__name__}: {exc}",
                "phase": str(phase),
                "executed_steps": int(executed_steps),
            }
        if not bool(result.get("stop", result.get("pass", False))):
            return None
        result["stop"] = True
        result["phase"] = str(phase)
        result["executed_steps"] = int(executed_steps)
        return result

    def maybe_abort_on_collision(phase: str) -> dict[str, Any] | None:
        nonlocal checked_collision_steps, first_collision
        if collision_monitor is None or not collision_monitor.get("enabled", False):
            return None
        checked_collision_steps += 1
        collision = check_monitor_collision(env, collision_monitor)
        if not collision.get("collision", False):
            return None
        collision = {
            **collision,
            "phase": phase,
            "checked_collision_steps": int(checked_collision_steps),
        }
        if first_collision is None:
            first_collision = dict(collision)
        if not bool(collision_monitor.get("abort_on_collision", True)):
            return None
        return _collision_abort_result(
            env,
            right,
            obs=obs,
            executed_steps=executed_steps,
            max_command_delta=max_command_delta,
            collision_monitor=collision_monitor,
            collision_summary=collision_summary,
            collision=collision,
            first_collision=first_collision,
        )

    with _installed_post_step_observation_hook(env, post_step_callback):
        for target in trajectory:
            target_qpos = np.array(target, dtype=float)
            delta = target_qpos - current
            max_delta = float(np.max(np.abs(delta)))
            max_command_delta = max(max_command_delta, max_delta)
            substeps = max(1, int(np.ceil(max_delta / max_joint_step)))
            for i in range(1, substeps + 1):
                waypoint = current + delta * (i / substeps)
                obs = step_arm_and_gripper(
                    env,
                    arm_qpos=waypoint,
                    gripper_action=gripper_action,
                )
                obs = _apply_post_step_callback(env, obs, post_step_callback)
                executed_steps += 1
                abort_result = maybe_abort_on_collision("trajectory")
                if abort_result is not None:
                    return abort_result
                early_stop = maybe_stop_trajectory("trajectory")
                if early_stop is not None:
                    break
            if early_stop is not None:
                break
            current = target_qpos

        final_target = np.array(trajectory[-1], dtype=float)
        for _ in range(0 if early_stop is not None else settle_steps):
            obs = step_arm_and_gripper(
                env,
                arm_qpos=final_target,
                gripper_action=gripper_action,
            )
            obs = _apply_post_step_callback(env, obs, post_step_callback)
            executed_steps += 1
            abort_result = maybe_abort_on_collision("settle")
            if abort_result is not None:
                return abort_result
            early_stop = maybe_stop_trajectory("settle")
            if early_stop is not None:
                break

    assert obs is not None
    final_qpos = qpos_for_controller(env, right)
    final_collision = check_monitor_collision(env, collision_monitor)
    if collision_monitor is not None and collision_monitor.get("enabled", False):
        checked_collision_steps += 1
        final_collision = {
            **final_collision,
            "phase": "final",
            "checked_collision_steps": int(checked_collision_steps),
        }
        if final_collision.get("collision", False) and first_collision is None:
            first_collision = dict(final_collision)
    collision_detected = bool(
        first_collision is not None or final_collision.get("collision", False)
    )
    return {
        "obs": obs,
        "executed_steps": int(executed_steps),
        "max_command_delta": float(max_command_delta),
        "final_qpos_error": float(np.max(np.abs(final_qpos - final_target))),
        "final_arm_qpos": final_qpos.astype(float).tolist(),
        "collision_monitor": collision_summary,
        "collision": {
            **final_collision,
            "collision": collision_detected,
            "first_collision": first_collision,
            "checked_collision_steps": int(checked_collision_steps),
        },
        "aborted_on_collision": False,
        "early_stop": early_stop,
    }


def run_gripper_command(
    env: Any,
    *,
    arm_qpos: np.ndarray,
    gripper_action: float,
    steps: int,
    post_step_callback: PostStepCallback | None = None,
) -> dict[str, Any]:
    if steps < 0:
        raise ValueError("steps must be non-negative")

    robot = env.robots[0]
    right = robot.composite_controller.part_controllers["right"]
    action_dict = action_dict_for_arm_and_gripper(
        env,
        arm_qpos=np.array(arm_qpos, dtype=float),
        gripper_action=gripper_action,
    )
    if "right_gripper" not in action_dict:
        raise RuntimeError("right_gripper action part not found")
    action = robot.create_action_vector(action_dict)
    if not np.all(np.isfinite(action)):
        raise RuntimeError("action contains NaN or Inf")
    obs = None
    with _installed_post_step_observation_hook(env, post_step_callback):
        for _ in range(steps):
            obs, _, _, _ = env.step(action)
            obs = _apply_post_step_callback(env, obs, post_step_callback)
    if obs is None:
        obs = env._get_observations(force_update=True)
    final_qpos = qpos_for_controller(env, right)
    return {
        "obs": obs,
        "steps": int(steps),
        "gripper_action": float(gripper_action),
        "arm_qpos_error": float(np.max(np.abs(final_qpos - np.array(arm_qpos, dtype=float)))),
        "final_arm_qpos": final_qpos.astype(float).tolist(),
    }


@dataclass
class RoboCasaArmDriver:
    """Stateful arm driver for RoboCasa-like envs."""

    env: Any
    obs: dict[str, Any]

    def current_arm_qpos(self) -> np.ndarray:
        right = self.env.robots[0].composite_controller.part_controllers["right"]
        return qpos_for_controller(self.env, right)

    def execute_trajectory(
        self,
        trajectory: list[list[float]],
        *,
        max_joint_step: float,
        settle_steps: int,
        gripper_action: float = 0.0,
        collision_monitor: dict[str, Any] | None = None,
        post_step_callback: PostStepCallback | None = None,
        trajectory_stop_check: TrajectoryStopCheck | None = None,
    ) -> dict[str, Any]:
        result = execute_arm_trajectory(
            self.env,
            trajectory,
            max_joint_step=max_joint_step,
            settle_steps=settle_steps,
            gripper_action=gripper_action,
            collision_monitor=collision_monitor,
            post_step_callback=post_step_callback,
            trajectory_stop_check=trajectory_stop_check,
        )
        self.obs = result["obs"]
        return result

    def run_gripper(
        self,
        *,
        arm_qpos: np.ndarray,
        gripper_action: float,
        steps: int,
        post_step_callback: PostStepCallback | None = None,
    ) -> dict[str, Any]:
        result = run_gripper_command(
            self.env,
            arm_qpos=arm_qpos,
            gripper_action=gripper_action,
            steps=steps,
            post_step_callback=post_step_callback,
        )
        self.obs = result["obs"]
        return result


__all__ = [
    "ArmDriver",
    "PostStepCallback",
    "RoboCasaArmDriver",
    "TrajectoryStopCheck",
    "action_dict_for_arm_and_gripper",
    "check_monitor_collision",
    "collision_monitor_summary",
    "execute_arm_trajectory",
    "qpos_for_controller",
    "run_gripper_command",
    "step_arm_and_gripper",
]
