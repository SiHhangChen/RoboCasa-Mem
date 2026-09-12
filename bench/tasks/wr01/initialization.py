"""Reset-time configuration and auditing for the WR-01 two-item scene."""

from __future__ import annotations

from typing import Any

import numpy as np

from bench.execution.navigation.base_motion import set_base_pose_kinematic
from bench.runtime.contracts import InitializationReport
from bench.execution.sampling.circle_sampling import sample_uniform_disk
from bench.tasks.wr01.domain.contracts import (
    CABINET_LEVELS,
    EpisodePlan,
    ITEM_IDS,
    RobotBaseConfig,
    RobotBaseMode,
)


def apply_robot_base_initial_state(
    env: Any,
    obs: dict[str, Any],
    *,
    config: RobotBaseConfig,
    seed: int = 0,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Apply the one declared base-reset mode and return an auditable summary."""

    if config.mode is RobotBaseMode.PRESERVE_RESET_POSE:
        return obs, _base_summary(obs, enabled=False, mode=config.mode.value)
    if config.mode is RobotBaseMode.EXPLICIT_POSE:
        if config.xy is None or config.yaw is None:
            raise ValueError("explicit WR-01 base pose is incomplete")
        radius = float(getattr(config, "sampling_radius_m", 0.0))
        center = np.asarray(config.xy, dtype=float)
        target = (
            sample_uniform_disk(
                center,
                radius,
                seed=int(seed),
                salt=0x57523149,
            )
            if radius > 0.0
            else center
        )
        updated, summary = _set_base_pose(
            env,
            obs,
            target,
            float(config.yaw),
            config.mode.value,
        )
        summary.update(
            {
                "sampling_region": {
                    "shape": "disk",
                    "center_xy": center.tolist(),
                    "radius_m": radius,
                    "seed": int(seed),
                    "salt": 0x57523149,
                }
                if radius > 0.0
                else None,
                "sampled_xy": target.tolist(),
            }
        )
        return updated, summary
    raise ValueError(f"unsupported WR-01 base mode: {config.mode!r}")


def apply_robot_torso_initial_state(
    env: Any,
    obs: dict[str, Any],
    *,
    target_height_m: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Set the WR01 lift column once at reset and keep it fixed thereafter."""

    target = float(target_height_m)
    robot = env.robots[0]
    composite = robot.composite_controller
    torso = composite.part_controllers.get("torso")
    if torso is None:
        raise RuntimeError("WR-01 robot does not expose a torso controller")
    qpos_index = np.asarray(torso.qpos_index, dtype=int).reshape(-1)
    joint_index = np.asarray(torso.joint_index, dtype=int).reshape(-1)
    if qpos_index.size != 1 or joint_index.size != 1:
        raise RuntimeError(
            "WR-01 expects one torso joint, got "
            f"qpos_index={qpos_index.tolist()} joint_index={joint_index.tolist()}"
        )
    joint_range = np.asarray(
        env.sim.model.jnt_range[int(joint_index[0])],
        dtype=float,
    ).reshape(2)
    if not float(joint_range[0]) <= target <= float(joint_range[1]):
        raise ValueError(
            f"WR-01 torso target {target} is outside joint range "
            f"{joint_range.astype(float).tolist()}"
        )

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
        raise RuntimeError(
            f"WR-01 torso reset failed: requested={target} actual={actual}"
        )
    return updated, {
        "mode": "fixed_reset_height",
        "joint_name": str(env.sim.model.joint_id2name(int(joint_index[0]))),
        "joint_range_m": joint_range.astype(float).tolist(),
        "initial_height_m": initial,
        "target_height_m": target,
        "actual_height_m": actual,
        "fixed_for_episode": True,
    }


def initialize_episode_state(
    env: Any,
    *,
    episode: EpisodePlan,
    cabinets: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Place both cans in their sampled cabinet interiors at reset.

    The reset is not an expert action: no pick/place stages are emitted and the
    relation is recorded privately for evaluation while remaining unlatched
    until both doors are closed. Neutral reset is an expert-recording action.
    """

    env.configure_episode(
        item_by_level=episode.item_by_level,
        target_item_id=episode.target_item_id,
    )
    env.configure_canonical_physics()
    placements: dict[str, Any] = {}
    for item_id in ITEM_IDS:
        position = np.asarray(env.get_object_world_position(item_id), dtype=float).reshape(-1)
        quat_xyzw = np.asarray(
            env.get_object_world_quaternion_xyzw(item_id),
            dtype=float,
        ).reshape(-1)
        if (
            position.size != 3
            or quat_xyzw.size != 4
            or not np.all(np.isfinite(position))
            or not np.all(np.isfinite(quat_xyzw))
        ):
            raise ValueError(f"invalid WR-01 sampled pose for {item_id!r}")
        placements[item_id] = {
            "table_world_position": position.astype(float).tolist(),
            "world_quat_xyzw": quat_xyzw.astype(float).tolist(),
            "visible_before_storage": False,
        }
        env._table_target_positions[item_id] = position.copy()
    # Move the already-spawned rigid bodies into their assigned open cabinets.
    # This is reset-time scene construction, not a recorded manipulation.
    for level, item_id in episode.item_by_level.items():
        target = np.asarray(env.cabinet_for_level(level).interior_pose(), dtype=float)
        env.set_object_pose(item_id, target, placements[item_id]["world_quat_xyzw"])
        placements[item_id]["cabinet_level"] = level
        placements[item_id]["world_position"] = target.astype(float).tolist()
    initial_fraction = float(cabinets.get("open_fraction", 1.0))
    doors_initially_open = bool(cabinets.get("doors_initially_open", True))
    for level in CABINET_LEVELS:
        env.set_level_door_open_fraction(level, initial_fraction if doors_initially_open else 0.0)
    env.sim.forward()
    obs = env._get_observations(force_update=True)
    return obs, {
        "seed": episode.seed,
        "item_by_level": dict(episode.item_by_level),
        "target_item_id": episode.target_item_id,
        "doors_initially_open": doors_initially_open,
        "door_open_fraction": initial_fraction if doors_initially_open else 0.0,
        "placements": placements,
        "placement_mode": "reset_into_open_cabinets",
        "placement_seed": int(env.item_placement_seed),
        "sampling_regions": env.item_sampling_regions(),
        "setup_recorded_in_hdf5_actions": False,
        "storage_hidden": False,
    }


def _set_base_pose(
    env: Any,
    obs: dict[str, Any],
    target_xy: np.ndarray,
    target_yaw: float,
    mode: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    reset_xy = np.asarray(obs["robot0_base_pos"][:2], dtype=float)
    updated = set_base_pose_kinematic(env, target_xy=target_xy, target_yaw=target_yaw)
    summary = _base_summary(updated, enabled=True, mode=mode)
    summary.update(
        {
            "reset_base_xy": reset_xy.tolist(),
            "target_xy": target_xy.tolist(),
            "target_yaw": float(target_yaw),
        }
    )
    return updated, summary


def _base_summary(obs: dict[str, Any], *, enabled: bool, mode: str) -> dict[str, Any]:
    return {
        "enabled": bool(enabled),
        "mode": str(mode),
        "actual_base_pos": np.asarray(obs["robot0_base_pos"], dtype=float).tolist(),
        "actual_base_quat_xyzw": np.asarray(obs["robot0_base_quat"], dtype=float).tolist(),
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
