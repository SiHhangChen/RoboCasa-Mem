"""Reset-time physical state application for WR-05."""

from __future__ import annotations

from typing import Any

import numpy as np

from bench.execution.navigation.base_motion import set_base_pose_kinematic
from bench.runtime.contracts import InitializationReport
from bench.tasks.wr05.domain.contracts import (
    EpisodePlan,
    RobotBaseConfig,
    RobotBaseMode,
)


def initialize_episode_state(
    env: Any,
    obs: dict[str, Any],
    *,
    episode: EpisodePlan,
    robot_base: RobotBaseConfig,
    objects: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Apply declared reset mutations and return an auditable summary."""

    if dict(env.home_assignment) != dict(episode.home_assignment):
        raise ValueError("WR-05 environment home assignment differs from EpisodePlan")
    if dict(env.use_assignment) != dict(episode.use_assignment):
        raise ValueError("WR-05 environment use assignment differs from EpisodePlan")
    if objects:
        raise ValueError(
            "unknown WR-05 initial object settings: " + ", ".join(sorted(objects))
        )

    if robot_base.mode is RobotBaseMode.EXPLICIT_POSE:
        if robot_base.xy is None or robot_base.yaw is None:
            raise ValueError("explicit WR-05 robot base pose is incomplete")
        obs = set_base_pose_kinematic(
            env,
            target_xy=np.asarray(robot_base.xy, dtype=float),
            target_yaw=float(robot_base.yaw),
        )
    elif robot_base.mode is not RobotBaseMode.PRESERVE_RESET_POSE:
        raise ValueError(f"unsupported WR-05 robot base mode: {robot_base.mode!r}")

    return obs, {
        "setup_recorded_in_hdf5_actions": False,
        "episode_case_id": episode.case_id,
        "robot_base_mode": robot_base.mode.value,
        "actual_base_pos": np.asarray(obs["robot0_base_pos"], dtype=float).tolist(),
        "actual_base_quat_xyzw": np.asarray(obs["robot0_base_quat"], dtype=float).tolist(),
    }


__all__ = ["initialize_episode_state"]


def initialize_episode(runtime: Any, episode: Any) -> InitializationReport:
    """Expose reset facts already applied by the task runtime."""
    del episode
    return InitializationReport.applied(facts=dict(getattr(runtime, "initialization", {}) or {}))
