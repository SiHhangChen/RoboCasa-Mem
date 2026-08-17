"""Reset-time configuration and auditing for the WR-03 one-can scene."""

from __future__ import annotations

from typing import Any

import numpy as np

from bench.execution.navigation.base_motion import set_base_pose_kinematic
from bench.runtime.contracts import InitializationReport
from bench.tasks.wr03.domain.contracts import (
    CABINET_LEVELS,
    CAN_ID,
    EpisodePlan,
    RobotBaseConfig,
    RobotBaseMode,
)


def apply_robot_base_initial_state(
    env: Any,
    obs: dict[str, Any],
    *,
    config: RobotBaseConfig,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if config.mode is RobotBaseMode.PRESERVE_RESET_POSE:
        return obs, _base_summary(obs, enabled=False, mode=config.mode.value)
    if config.mode is not RobotBaseMode.EXPLICIT_POSE:
        raise ValueError(f"unsupported WR-03 base mode: {config.mode!r}")
    if config.xy is None or config.yaw is None:
        raise ValueError("explicit WR-03 base pose is incomplete")
    target_xy = np.asarray(config.xy, dtype=float)
    reset_xy = np.asarray(obs["robot0_base_pos"][:2], dtype=float)
    updated = set_base_pose_kinematic(
        env, target_xy=target_xy, target_yaw=float(config.yaw)
    )
    summary = _base_summary(updated, enabled=True, mode=config.mode.value)
    summary.update({
        "reset_base_xy": reset_xy.tolist(),
        "target_xy": target_xy.tolist(),
        "target_yaw": float(config.yaw),
    })
    return updated, summary


def apply_robot_torso_initial_state(
    env: Any,
    obs: dict[str, Any],
    *,
    target_height_m: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    target = float(target_height_m)
    robot = env.robots[0]
    composite = robot.composite_controller
    torso = composite.part_controllers.get("torso")
    if torso is None:
        raise RuntimeError("WR-03 robot does not expose a torso controller")
    qpos_index = np.asarray(torso.qpos_index, dtype=int).reshape(-1)
    joint_index = np.asarray(torso.joint_index, dtype=int).reshape(-1)
    if qpos_index.size != 1 or joint_index.size != 1:
        raise RuntimeError("WR-03 expects exactly one torso joint")
    joint_range = np.asarray(
        env.sim.model.jnt_range[int(joint_index[0])], dtype=float
    ).reshape(2)
    if not float(joint_range[0]) <= target <= float(joint_range[1]):
        raise ValueError(f"WR-03 torso target {target} is outside {joint_range.tolist()}")
    initial = float(env.sim.data.qpos[int(qpos_index[0])])
    env.sim.data.qpos[int(qpos_index[0])] = target
    qvel_index = np.asarray(getattr(torso, "qvel_index", ()), dtype=int).reshape(-1)
    if qvel_index.size == 1:
        env.sim.data.qvel[int(qvel_index[0])] = 0.0
    env.sim.forward()
    if hasattr(torso, "update_initial_joints"):
        torso.update_initial_joints(np.asarray([target], dtype=float))
    if hasattr(torso, "reset_goal"):
        torso.reset_goal()
    if hasattr(composite, "reset"):
        composite.reset()
    updated = env._get_observations(force_update=True)
    actual = float(env.sim.data.qpos[int(qpos_index[0])])
    if abs(actual - target) > 1e-6:
        raise RuntimeError(f"WR-03 torso reset failed: requested={target} actual={actual}")
    return updated, {
        "mode": "fixed_reset_height",
        "joint_name": str(env.sim.model.joint_id2name(int(joint_index[0]))),
        "joint_range_m": joint_range.astype(float).tolist(),
        "initial_height_m": initial,
        "target_height_m": target,
        "actual_height_m": actual,
        "fixed_for_episode": True,
    }


def _base_summary(obs: dict[str, Any], *, enabled: bool, mode: str) -> dict[str, Any]:
    return {
        "enabled": bool(enabled),
        "mode": str(mode),
        "actual_base_pos": np.asarray(obs["robot0_base_pos"], dtype=float).tolist(),
        "actual_base_quat_xyzw": np.asarray(obs["robot0_base_quat"], dtype=float).tolist(),
    }


def initialize_episode_state(
    env: Any,
    *,
    episode: EpisodePlan,
    cabinets: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Install the private demo route while preserving the sampled can pose."""

    env.configure_episode(demo_storage_level=episode.demo_storage_level)
    env.configure_canonical_physics()
    position = np.asarray(env.get_object_world_position(CAN_ID), dtype=float).reshape(-1)
    quat_xyzw = np.asarray(
        env.get_object_world_quaternion_xyzw(CAN_ID), dtype=float
    ).reshape(-1)
    if (
        position.size != 3
        or quat_xyzw.size != 4
        or not np.all(np.isfinite(position))
        or not np.all(np.isfinite(quat_xyzw))
    ):
        raise ValueError(f"invalid WR-03 sampled pose for {CAN_ID!r}")

    initial_fraction = float(cabinets.get("open_fraction", 1.0))
    doors_initially_open = bool(cabinets.get("doors_initially_open", True))
    if not doors_initially_open:
        raise ValueError("WR-03 requires both cabinet doors to start open")
    for level in CABINET_LEVELS:
        env.set_level_door_open_fraction(level, initial_fraction)
    env.sim.forward()
    env.start_physical_observer()
    obs = env._get_observations(force_update=True)
    return obs, {
        "episode_seed": episode.seed,
        "demo_storage_level": episode.demo_storage_level,
        "placement_pair_seed": episode.placement_pair_seed,
        "doors_initially_open": True,
        "door_open_fraction": initial_fraction,
        "placements": {
            CAN_ID: {
                "world_position": position.astype(float).tolist(),
                "world_quat_xyzw": quat_xyzw.astype(float).tolist(),
                "visible_before_storage": True,
            }
        },
        "placement_mode": "paired_sampled_region",
        "sampling_regions": env.item_sampling_regions(),
        "setup_recorded_in_hdf5_actions": False,
    }


__all__ = [
    "apply_robot_base_initial_state",
    "apply_robot_torso_initial_state",
    "initialize_episode_state",
]


def initialize_episode(runtime: Any, episode: Any) -> InitializationReport:
    """Expose reset facts already applied by the task runtime."""
    del episode
    return InitializationReport.applied(facts=dict(getattr(runtime, "initialization", {}) or {}))
