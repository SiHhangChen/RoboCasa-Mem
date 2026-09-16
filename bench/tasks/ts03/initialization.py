"""Reset-time TS03 state application."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from bench.execution.navigation.base_motion import set_base_pose_kinematic
from bench.runtime.contracts import InitializationReport
from bench.execution.planning import NavigationConfig
from bench.tasks.ts03.domain.contracts import (
    EpisodePlan,
    RobotBaseConfig,
    RobotBaseMode,
)
from bench.tasks.ts03.domain.geometry import (
    SUGAR_PLATE_EDGE_MARGIN,
    SUGAR_PLATE_SAMPLING_RADIUS_M,
    sample_sugar_cube_yaws_for_seed,
    sample_sugar_plate_offsets_xy,
    validate_sugar_plate_offsets_xy,
)


def apply_robot_base_initial_state(
    env: Any,
    obs: dict[str, Any],
    *,
    config: RobotBaseConfig,
    nav_config: NavigationConfig,
    seed: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Apply the configured reset-time Base pose and return diagnostics."""

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
        path = nav_config.manual_base_path_file
        if path is None or config.sample_index is None:
            raise ValueError("manual_path_sample requires a path and sample index")
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        samples = {
            int(item.get("index", fallback)): dict(item)
            for fallback, item in enumerate(payload.get("samples", []))
        }
        sample = samples.get(int(config.sample_index))
        if sample is None:
            raise KeyError(f"manual base sample {config.sample_index} not found in {path}")
        xy = np.asarray(sample.get("xy", []), dtype=float).reshape(-1)
        yaw = float(sample.get("yaw"))
        if xy.size != 2 or not np.all(np.isfinite(xy)) or not np.isfinite(yaw):
            raise ValueError(f"invalid manual base sample {config.sample_index}")
        new_obs, summary = _set_base_pose(
            env,
            obs,
            target_xy=xy,
            target_yaw=yaw,
            mode=config.mode.value,
        )
        summary.update(
            {
                "manual_base_path_file": str(path),
                "sample_index": int(config.sample_index),
                "sample_name": sample.get("name"),
                "sample": sample,
                "seed": int(seed),
            }
        )
        return new_obs, summary
    if config.mode is RobotBaseMode.STATION_SAMPLE:
        raise ValueError("TS03 station_sample requires an explicit station pose resource")
    raise ValueError(f"unsupported TS03 base mode: {config.mode!r}")


def apply_initial_object_state(
    env: Any,
    *,
    episode: EpisodePlan,
    settings: dict[str, Any] | None = None,
    enabled: bool = True,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Place all N cubes in a seeded, separated tray-local layout.

    RoboCasa has already created the plate and cubes when this function runs;
    only free-joint state and task-local physical coefficients are changed.
    """

    values = dict(settings or {})
    summary: dict[str, Any] = {
        "enabled": bool(enabled),
        "seed": int(episode.seed),
        "layout_seed": int(episode.sugar_plate_layout_seed),
        "objects": {},
    }
    if not enabled:
        return env._get_observations(force_update=True), summary
    plate_name = str(getattr(env, "sugar_plate_name", "sugar_plate"))
    plate = env.objects[plate_name]
    plate_qpos = np.asarray(env.sim.data.get_joint_qpos(plate.joints[0]), dtype=float).reshape(-1)
    if plate_qpos.size < 7:
        raise ValueError("sugar plate must have a free joint")
    plate_pos = plate_qpos[:3].copy()
    import mujoco
    from robosuite.utils import transform_utils as T

    plate_rot = T.quat2mat(T.convert_quat(plate_qpos[3:7], to="xyzw"))
    cube_radii = []
    for name in episode.sugar_names:
        size = np.asarray(getattr(env.objects[name], "size", (0.02, 0.02, 0.02)), dtype=float)
        cube_radii.append(0.5 * float(np.linalg.norm(size)))
    plate_half_extents = 0.5 * np.asarray(getattr(plate, "size", (0.36, 0.32, 0.02)), dtype=float)[:2]
    edge_margin = float(values.get("sugar_plate_edge_margin", SUGAR_PLATE_EDGE_MARGIN))
    sampling_radius = float(
        values.get("sugar_plate_sampling_radius_m", SUGAR_PLATE_SAMPLING_RADIUS_M)
    )
    yaw_half_range = np.deg2rad(float(values.get("sugar_cube_yaw_half_range_deg", 180.0)))
    safe_half_extents = plate_half_extents - max(cube_radii) - edge_margin
    if not np.all(np.isfinite(safe_half_extents)) or np.any(safe_half_extents <= 0.0):
        raise ValueError("sugar cubes do not fit inside the tray after applying edge margin")
    offsets = sample_sugar_plate_offsets_xy(
        count=episode.total_sugar_count,
        rng=np.random.default_rng(int(episode.sugar_plate_layout_seed)),
        safe_half_extents_xy=safe_half_extents,
        sampling_radius_m=sampling_radius,
    )
    validate_sugar_plate_offsets_xy(
        offsets,
        count=episode.total_sugar_count,
        safe_half_extents_xy=safe_half_extents,
    )
    yaws = sample_sugar_cube_yaws_for_seed(
        count=episode.total_sugar_count,
        seed=episode.seed,
        half_range_rad=yaw_half_range,
    )
    env.sugar_plate_offsets_xy = tuple(offsets)
    env.sugar_cube_yaws_rad = yaws
    env.sugar_plate_safe_half_extents_xy = tuple(float(value) for value in safe_half_extents)
    summary["sugar_plate_offsets_xy"] = [list(row) for row in offsets]
    summary["sugar_plate_safe_half_extents_xy"] = list(env.sugar_plate_safe_half_extents_xy)
    summary["sugar_plate_sampling_radius_m"] = sampling_radius
    summary["sugar_cube_yaws_rad"] = list(yaws)
    summary["sugar_cube_yaw_half_range_deg"] = float(np.rad2deg(yaw_half_range))

    for name, offset, yaw in zip(episode.sugar_names, offsets, yaws, strict=True):
        obj = env.objects[name]
        qpos = np.asarray(env.sim.data.get_joint_qpos(obj.joints[0]), dtype=float).reshape(-1)
        if qpos.size < 7:
            raise ValueError(f"sugar object {name!r} must have a free joint")
        local = np.asarray([float(offset[0]), float(offset[1]), 0.0], dtype=float)
        qpos[:3] = plate_pos + plate_rot @ local
        # Change world yaw while preserving the cube's reset-time roll/pitch.
        rotation = np.empty(9, dtype=float)
        mujoco.mju_quat2Mat(rotation, qpos[3:7])
        delta_yaw = yaw - float(np.arctan2(rotation[3], rotation[0]))
        half_yaw = 0.5 * delta_yaw
        yaw_quaternion = np.asarray([np.cos(half_yaw), 0.0, 0.0, np.sin(half_yaw)])
        rotated_quaternion = np.empty(4, dtype=float)
        mujoco.mju_mulQuat(rotated_quaternion, yaw_quaternion, qpos[3:7])
        qpos[3:7] = rotated_quaternion
        env.sim.data.set_joint_qpos(obj.joints[0], qpos)
        if name in getattr(env, "object_placements", {}):
            env.object_placements[name] = (
                tuple(float(value) for value in qpos[:3]),
                tuple(float(value) for value in qpos[3:7]),
                obj,
            )
        summary["objects"][name] = {
            "after_pos": qpos[:3].astype(float).tolist(),
            "after_quat_wxyz": qpos[3:7].astype(float).tolist(),
            "offset_xy": list(offset),
            "yaw_rad": yaw,
        }

    _apply_object_coefficients(env, episode.sugar_names, values, summary)
    env.sim.forward()
    if hasattr(env, "original_sugar_poses"):
        env.original_sugar_poses = {
            name: _placement_pose(env, name) for name in episode.sugar_names
        }
    return env._get_observations(force_update=True), summary


def apply_fixed_initial_object_regions(
    env: Any,
    *,
    episode: EpisodePlan,
    settings: dict[str, Any] | None = None,
    enabled: bool = True,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Compatibility name used by composition roots."""

    return apply_initial_object_state(
        env,
        episode=episode,
        settings=settings,
        enabled=enabled,
    )


def _apply_object_coefficients(
    env: Any,
    names: tuple[str, ...],
    settings: dict[str, Any],
    summary: dict[str, Any],
) -> None:
    friction = float(settings.get("sugar_cube_sliding_friction", 1.0))
    mass_scale = float(settings.get("sugar_cube_mass_scale", 1.0))
    if friction <= 0.0 or mass_scale <= 0.0:
        raise ValueError("sugar cube friction and mass scale must be positive")
    model = env.sim.model
    parents = np.asarray(model.body_parentid, dtype=int)
    for name in names:
        root = int(env.obj_body_id[name])

        def belongs(body_id: int) -> bool:
            current = int(body_id)
            while current > 0:
                if current == root:
                    return True
                current = int(parents[current])
            return current == root

        geom_ids = [
            int(index)
            for index, body_id in enumerate(np.asarray(model.geom_bodyid, dtype=int))
            if belongs(int(body_id))
        ]
        for geom_id in geom_ids:
            model.geom_friction[geom_id, 0] = friction
        body_ids = [body_id for body_id in range(int(model.nbody)) if belongs(body_id)]
        for body_id in body_ids:
            model.body_mass[body_id] *= mass_scale
            model.body_inertia[body_id] *= mass_scale
        summary["objects"].setdefault(name, {}).update(
            {
                "sliding_friction": friction,
                "mass_scale": mass_scale,
                "geom_count": len(geom_ids),
                "body_count": len(body_ids),
            }
        )


def _set_base_pose(
    env: Any,
    obs: dict[str, Any],
    *,
    target_xy: np.ndarray,
    target_yaw: float,
    mode: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    reset_xy = np.asarray(obs["robot0_base_pos"][:2], dtype=float)
    new_obs = set_base_pose_kinematic(
        env,
        target_xy=np.asarray(target_xy, dtype=float),
        target_yaw=float(target_yaw),
    )
    summary = _base_summary(new_obs, enabled=True, mode=mode)
    summary.update(
        {
            "reset_base_xy": reset_xy.tolist(),
            "target_xy": np.asarray(target_xy, dtype=float).tolist(),
            "target_yaw": float(target_yaw),
        }
    )
    return new_obs, summary


def _base_summary(obs: dict[str, Any], *, enabled: bool, mode: str) -> dict[str, Any]:
    return {
        "enabled": bool(enabled),
        "mode": str(mode),
        "actual_base_pos": np.asarray(obs["robot0_base_pos"], dtype=float).tolist(),
        "actual_base_quat_xyzw": np.asarray(obs["robot0_base_quat"], dtype=float).tolist(),
    }


def _placement_pose(env: Any, name: str) -> list[float] | None:
    placement = getattr(env, "object_placements", {}).get(str(name))
    if placement is not None:
        return np.asarray(placement[0], dtype=float).tolist()
    if name in getattr(env, "obj_body_id", {}):
        return np.asarray(env.sim.data.body_xpos[env.obj_body_id[name]], dtype=float).tolist()
    return None


__all__ = [
    "apply_fixed_initial_object_regions",
    "apply_initial_object_state",
    "apply_robot_base_initial_state",
]


def initialize_episode(runtime: Any, episode: Any) -> InitializationReport:
    """Expose reset facts already applied by the task runtime."""
    del episode
    return InitializationReport.applied(facts=dict(getattr(runtime, "initialization", {}) or {}))
