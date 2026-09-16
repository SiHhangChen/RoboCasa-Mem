"""Reset-time TS02 drawer and mobile-base initialization."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from bench.execution.navigation.base_motion import set_base_pose_kinematic
from bench.runtime.contracts import InitializationReport
from bench.execution.navigation.station_poses import (
    StationPoseSelector,
    load_station_pose_payload,
    station_sample_records,
)
from bench.execution.navigation.manual_base_path import load_manual_path_samples
from bench.execution.planning import NavigationConfig
from bench.tasks.ts02.domain.contracts import RobotBaseConfig, RobotBaseMode


def apply_initial_drawer_state(
    env: Any,
    *,
    enabled: bool = True,
    closed_fraction: float = 0.0,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Close all drawers before recording and clear all semantic history."""

    value = float(closed_fraction)
    if not 0.0 <= value <= float(env.closed_threshold):
        raise ValueError(
            "initial closed_fraction must satisfy the task closed threshold"
        )
    before = {
        fixture: float(env.slot_open_fraction(fixture)) for fixture in env.fixture_names
    }
    if enabled:
        env.close_all_drawers(fraction=value)
    env.reset_semantic_state()
    env.sim.forward()
    after = {
        fixture: float(env.slot_open_fraction(fixture)) for fixture in env.fixture_names
    }
    if enabled and not all(
        fraction <= env.closed_threshold for fraction in after.values()
    ):
        raise RuntimeError("TS02 drawer initialization did not close every fixture")
    return env._get_observations(force_update=True), {
        "enabled": bool(enabled),
        "recorded_in_hdf5_actions": False,
        "closed_fraction": value,
        "before_open_fractions": before,
        "after_open_fractions": after,
        "semantic_state_reset": True,
    }


def apply_robot_base_initial_state(
    env: Any,
    obs: dict[str, Any],
    *,
    config: RobotBaseConfig,
    nav_config: NavigationConfig,
    seed: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Apply exactly one declared reset-time Base mode."""

    if config.mode is RobotBaseMode.PRESERVE_RESET_POSE:
        return obs, _base_summary(obs, enabled=False, mode=config.mode.value)
    if config.mode is RobotBaseMode.EXPLICIT_POSE:
        if config.xy is None or config.yaw is None:
            raise ValueError("explicit_pose requires resolved xy and yaw")
        return _set_base_pose(
            env,
            obs,
            target_xy=np.asarray(config.xy, dtype=float),
            target_yaw=float(config.yaw),
            mode=config.mode.value,
        )
    if config.mode is RobotBaseMode.MANUAL_PATH_SAMPLE:
        path_value = nav_config.manual_base_path_file
        if path_value is None or config.sample_index is None:
            raise ValueError(
                "manual_path_sample requires a manual path and sample index"
            )
        path = Path(path_value)
        samples = load_manual_path_samples(path, seed=int(seed))
        sample_obj = next((item for item in samples if item.index == config.sample_index), None)
        if sample_obj is None:
            raise KeyError(
                f"manual base sample {config.sample_index} not found in {path}"
            )
        xy = np.asarray(sample_obj.xy, dtype=float).reshape(-1)
        yaw = float(sample_obj.yaw)
        if xy.size != 2 or not np.all(np.isfinite(xy)) or not np.isfinite(yaw):
            raise ValueError(f"invalid manual base sample {config.sample_index}")
        updated, summary = _set_base_pose(
            env,
            obs,
            target_xy=xy,
            target_yaw=yaw,
            mode=config.mode.value,
        )
        summary.update(
            {
                "manual_base_path_file": str(path),
                "sample_index": config.sample_index,
                "sample_name": sample_obj.name,
                "sample": {
                    "index": sample_obj.index,
                    "name": sample_obj.name,
                    "xy": xy.tolist(),
                    "yaw": yaw,
                },
            }
        )
        return updated, summary
    if config.mode is RobotBaseMode.STATION_SAMPLE:
        path = nav_config.station_base_pose_file
        if path is None or config.station_id is None:
            raise ValueError("station_sample requires a station pose resource and id")
        payload = load_station_pose_payload(Path(path))
        station = (payload.get("stations", {}) or {}).get(config.station_id)
        if station is None:
            raise KeyError(f"initial station {config.station_id!r} not found")
        records = station_sample_records(list(station.get("samples", [])))
        if not records:
            raise ValueError(f"initial station {config.station_id!r} has no samples")
        reset_xy = np.asarray(obs["robot0_base_pos"][:2], dtype=float)
        selector = StationPoseSelector.from_config(nav_config, seed=int(seed))
        record, selection = selector.select(
            config.station_id,
            records,
            base_xy=reset_xy,
        )
        updated, summary = _set_base_pose(
            env,
            obs,
            target_xy=np.asarray(record["target_xy"], dtype=float),
            target_yaw=float(record["target_yaw"]),
            mode=config.mode.value,
        )
        summary.update(
            {
                "station_id": config.station_id,
                "station_base_pose_file": str(path),
                "selection": selection,
                "sample_index": int(record["sample_index"]),
            }
        )
        return updated, summary
    raise ValueError(f"unsupported TS02 base mode: {config.mode!r}")


def _set_base_pose(
    env: Any,
    obs: dict[str, Any],
    *,
    target_xy: np.ndarray,
    target_yaw: float,
    mode: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    reset_xy = np.asarray(obs["robot0_base_pos"][:2], dtype=float)
    updated = set_base_pose_kinematic(
        env,
        target_xy=np.asarray(target_xy, dtype=float),
        target_yaw=float(target_yaw),
    )
    summary = _base_summary(updated, enabled=True, mode=mode)
    summary.update(
        {
            "reset_base_xy": reset_xy.tolist(),
            "target_xy": np.asarray(target_xy, dtype=float).tolist(),
            "target_yaw": float(target_yaw),
        }
    )
    return updated, summary


def _base_summary(obs: dict[str, Any], *, enabled: bool, mode: str) -> dict[str, Any]:
    return {
        "enabled": bool(enabled),
        "mode": str(mode),
        "actual_base_pos": np.asarray(obs["robot0_base_pos"], dtype=float).tolist(),
        "actual_base_quat_xyzw": np.asarray(
            obs["robot0_base_quat"], dtype=float
        ).tolist(),
    }


__all__ = ["apply_initial_drawer_state", "apply_robot_base_initial_state"]


def initialize_episode(runtime: Any, episode: Any) -> InitializationReport:
    """Expose reset facts already applied by the task runtime."""
    del episode
    return InitializationReport.applied(facts=dict(getattr(runtime, "initialization", {}) or {}))
