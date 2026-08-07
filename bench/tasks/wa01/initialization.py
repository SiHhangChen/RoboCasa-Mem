"""WA-01 在 reset 后执行的模拟器状态修改。"""

from __future__ import annotations

import json
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
from bench.execution.planning import NavigationConfig
from bench.tasks.wa01.domain.contracts import (
    COFFEE_IDS,
    RobotBaseConfig,
    RobotBaseMode,
)
from bench.tasks.wa01.domain.episode import (
    COFFEE_INITIAL_REACH_CENTER_XY,
    COFFEE_INITIAL_REGION_BOUNDS,
    COFFEE_INITIAL_YAW_BOUNDS,
    MAX_INITIAL_CUP_REACH_DISTANCE,
    MIN_INITIAL_CUP_SEPARATION,
    coffee_initial_poses_for_seed,
)


COFFEE_MACHINE_FIXED_OFFSET_WORLD = (0.0, -0.20, 0.0)
SUGAR_CUBE_INITIAL_OFFSET_XY = (0.0, 0.0)


def apply_robot_base_initial_state(
    env: Any,
    obs: dict[str, Any],
    *,
    config: RobotBaseConfig,
    nav_config: NavigationConfig,
    seed: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """按照唯一 Base 模式设置 reset 后位姿，并返回可诊断摘要。"""

    if config.mode is RobotBaseMode.PRESERVE_RESET_POSE:
        return obs, _base_summary(obs, enabled=False, mode=config.mode.value)
    if config.mode is RobotBaseMode.EXPLICIT_POSE:
        if config.xy is None or config.yaw is None:
            raise ValueError("explicit_pose requires resolved xy and yaw")
        return _set_base_pose(
            env,
            obs,
            target_xy=np.asarray(config.xy, dtype=float),
            target_yaw=config.yaw,
            mode=config.mode.value,
        )
    if config.mode is RobotBaseMode.MANUAL_PATH_SAMPLE:
        if nav_config.manual_base_path_file is None or config.sample_index is None:
            raise ValueError("manual_path_sample requires a path and sample index")
        path = Path(nav_config.manual_base_path_file)
        payload = json.loads(path.read_text(encoding="utf-8"))
        samples = {
            int(item.get("index", fallback)): dict(item)
            for fallback, item in enumerate(payload.get("samples", []))
        }
        sample = samples.get(config.sample_index)
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
                "sample_index": config.sample_index,
                "sample_name": sample.get("name"),
                "sample": sample,
            }
        )
        return new_obs, summary
    if config.mode is RobotBaseMode.STATION_SAMPLE:
        if nav_config.station_base_pose_file is None or config.station_id is None:
            raise ValueError("station_sample requires station poses and station id")
        payload = load_station_pose_payload(nav_config.station_base_pose_file)
        station = (payload.get("stations", {}) or {}).get(config.station_id)
        if station is None:
            raise KeyError(f"initial station {config.station_id!r} not found")
        records = station_sample_records(list(station.get("samples", [])))
        if not records:
            raise ValueError(f"initial station {config.station_id!r} has no samples")
        reset_xy = np.asarray(obs["robot0_base_pos"][:2], dtype=float)
        selector = StationPoseSelector.from_config(nav_config, seed=int(seed))
        record, selection = selector.select(config.station_id, records, base_xy=reset_xy)
        new_obs, summary = _set_base_pose(
            env,
            obs,
            target_xy=np.asarray(record["target_xy"], dtype=float),
            target_yaw=float(record["target_yaw"]),
            mode=config.mode.value,
        )
        summary.update(
            {
                "station_id": config.station_id,
                "station_base_pose_file": str(nav_config.station_base_pose_file),
                "selection": selection,
                "sample_index": int(record["sample_index"]),
                "sample": dict(record["sample"]),
            }
        )
        return new_obs, summary
    raise ValueError(f"unsupported WA-01 base mode: {config.mode!r}")


def apply_fixed_initial_object_regions(
    env: Any,
    *,
    seed: int,
    enabled: bool = True,
    coffee_initial_z: float = 0.97748,
    coffee_mug_sliding_friction: float = 30.0,
    coffee_mug_mass_scale: float = 0.35,
    sugar_plate_initial_region_center_xy: tuple[float, float] = (-0.14, -4.76),
    sugar_plate_initial_region_size_xy: tuple[float, float] = (0.0, 0.0),
    canonical_scene_seed: int = 0,
    sugar_plate_initial_z: float = 0.93172,
    sugar_plate_initial_yaw_deg: float = 0.0,
    sugar_cube_initial_z: float = 0.94582,
    sugar_cube_initial_yaw_deg: float = 0.0,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """应用杯子随机采样、糖碟固定位置以及杯子物理参数。"""

    summary: dict[str, Any] = {
        "enabled": bool(enabled),
        "seed": int(seed),
        "fixed_background": True,
        "coordinate_frame": "world",
        "objects": {},
    }
    if not enabled:
        return env._get_observations(force_update=True), summary
    summary["coffee_machine_fixed_offset"] = apply_fixed_coffee_machine_offset(env)
    summary["coffee_initial_region_bounds"] = {
        axis: list(bounds) for axis, bounds in COFFEE_INITIAL_REGION_BOUNDS.items()
    }
    summary["coffee_initial_reach_center_xy"] = list(COFFEE_INITIAL_REACH_CENTER_XY)
    summary["coffee_initial_max_reach_distance"] = MAX_INITIAL_CUP_REACH_DISTANCE
    summary["coffee_sampling_mode"] = "shared_reachable_region_rejection_sampling"
    sampled = coffee_initial_poses_for_seed(seed)
    for coffee_id in COFFEE_IDS:
        values = sampled[coffee_id]
        object_summary = _place_sampled_coffee(
            env,
            coffee_id,
            target_xy=np.asarray(values["xy"], dtype=float),
            yaw_deg=float(values["yaw_deg"]),
            z=float(coffee_initial_z),
            seed=int(seed),
            sampling_attempt=int(values["sampling_attempt"]),
        )
        object_summary["friction"] = _set_object_sliding_friction(
            env,
            coffee_id,
            sliding_friction=float(coffee_mug_sliding_friction),
        )
        object_summary["mass"] = _scale_object_mass(
            env,
            coffee_id,
            scale=float(coffee_mug_mass_scale),
        )
        summary["objects"][coffee_id] = object_summary
    coffee_positions = [
        np.asarray(summary["objects"][coffee_id]["after_pos"], dtype=float)[:2]
        for coffee_id in COFFEE_IDS
    ]
    reach_center = np.asarray(COFFEE_INITIAL_REACH_CENTER_XY, dtype=float)
    for coffee_id, position in zip(COFFEE_IDS, coffee_positions, strict=True):
        if not _xy_in_initial_region(position):
            raise ValueError(f"{coffee_id} is outside the WA-01 initial region")
        reach = float(np.linalg.norm(position - reach_center))
        if reach > MAX_INITIAL_CUP_REACH_DISTANCE + 1e-9:
            raise ValueError(f"{coffee_id} is outside the WA-01 reach radius")
    separation = float(np.linalg.norm(coffee_positions[0] - coffee_positions[1]))
    if separation < MIN_INITIAL_CUP_SEPARATION:
        raise ValueError("WA-01 initial coffee pair is too close")
    summary["initial_cup_separation"] = separation

    plate_center = np.asarray(sugar_plate_initial_region_center_xy, dtype=float)
    plate_size = np.asarray(sugar_plate_initial_region_size_xy, dtype=float)
    rng = np.random.RandomState(int(canonical_scene_seed))
    plate_xy = plate_center + rng.uniform(-0.5 * plate_size, 0.5 * plate_size)
    plate_target = np.asarray(
        [plate_xy[0], plate_xy[1], float(sugar_plate_initial_z)],
        dtype=float,
    )
    cube_target = np.asarray(
        [plate_target[0], plate_target[1], float(sugar_cube_initial_z)],
        dtype=float,
    )
    plate_before = _object_joint_position(env, "sugar_plate")
    cube_before = _object_joint_position(env, "sugar_cube")
    plate_applied = _set_object_pose(
        env,
        "sugar_plate",
        plate_target,
        quat_wxyz=_yaw_quaternion(float(sugar_plate_initial_yaw_deg)),
    )
    cube_applied = _set_object_pose(
        env,
        "sugar_cube",
        cube_target,
        quat_wxyz=_yaw_quaternion(float(sugar_cube_initial_yaw_deg)),
    )
    summary["objects"]["sugar_plate"] = {
        "applied": plate_applied,
        "before_pos": plate_before.tolist(),
        "after_pos": plate_target.tolist(),
        "center_xy": plate_center.tolist(),
        "size_xy": plate_size.tolist(),
        "yaw_deg": float(sugar_plate_initial_yaw_deg),
        "seed": int(canonical_scene_seed),
    }
    summary["objects"]["sugar_cube"] = {
        "applied": cube_applied,
        "before_pos": cube_before.tolist(),
        "after_pos": cube_target.tolist(),
        "offset_xy": list(SUGAR_CUBE_INITIAL_OFFSET_XY),
        "centered_on_plate": True,
        "yaw_deg": float(sugar_cube_initial_yaw_deg),
        "moved_with_sugar_plate": True,
    }
    env.sim.forward()
    return env._get_observations(force_update=True), summary


def apply_fixed_coffee_machine_offset(env: Any) -> dict[str, Any]:
    """将咖啡机移动到已校准位置，并验证仍位于操作台范围内。"""

    cached = getattr(env, "_wa01_coffee_machine_offset_summary", None)
    if isinstance(cached, dict):
        return dict(cached)
    fixture = env.coffee_machine
    root_body = str(fixture.root_body)
    body_id = int(env.sim.model.body_name2id(root_body))
    if int(env.sim.model.body_parentid[body_id]) != 0:
        raise ValueError("WA-01 coffee-machine root must be world-rooted")
    delta = np.asarray(COFFEE_MACHINE_FIXED_OFFSET_WORLD, dtype=float)
    before_pos = np.asarray(fixture.pos, dtype=float).copy()
    before_bbox = np.asarray(fixture.get_bbox_points(), dtype=float)
    before_pour = np.asarray(env.get_coffee_machine_pour_site_world_position(), dtype=float)
    target_pos = before_pos + delta
    fixture.set_pos(target_pos)
    env.sim.model.body_pos[body_id, :3] = target_pos
    env.sim.forward()
    after_bbox = np.asarray(fixture.get_bbox_points(), dtype=float)
    after_pour = np.asarray(env.get_coffee_machine_pour_site_world_position(), dtype=float)
    counter_bbox = np.asarray(env.counter.get_bbox_points(), dtype=float)
    machine_min, machine_max = np.min(after_bbox, axis=0), np.max(after_bbox, axis=0)
    counter_min, counter_max = np.min(counter_bbox, axis=0), np.max(counter_bbox, axis=0)
    inside = bool(
        np.all(machine_min[:2] >= counter_min[:2] - 1e-9)
        and np.all(machine_max[:2] <= counter_max[:2] + 1e-9)
    )
    if not inside:
        raise ValueError("shifted WA-01 coffee machine is outside counter bounds")
    summary = {
        "applied": True,
        "coordinate_frame": "world",
        "offset_world": delta.tolist(),
        "before_pos": before_pos.tolist(),
        "after_pos": target_pos.tolist(),
        "before_aabb_min": np.min(before_bbox, axis=0).tolist(),
        "before_aabb_max": np.max(before_bbox, axis=0).tolist(),
        "after_aabb_min": machine_min.tolist(),
        "after_aabb_max": machine_max.tolist(),
        "before_pour_site": before_pour.tolist(),
        "after_pour_site": after_pour.tolist(),
        "pour_site_delta": (after_pour - before_pour).tolist(),
        "inside_counter_xy": inside,
        "root_body": root_body,
    }
    env._wa01_coffee_machine_offset_summary = dict(summary)
    return summary


def _set_base_pose(
    env: Any,
    obs: dict[str, Any],
    *,
    target_xy: np.ndarray,
    target_yaw: float,
    mode: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """调用通用 Base 设置器并记录设置前后位姿。"""

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
    """从 observation 生成统一的 Base 初始化摘要。"""

    return {
        "enabled": bool(enabled),
        "mode": str(mode),
        "actual_base_pos": np.asarray(obs["robot0_base_pos"], dtype=float).tolist(),
        "actual_base_quat_xyzw": np.asarray(obs["robot0_base_quat"], dtype=float).tolist(),
    }


def _place_sampled_coffee(
    env: Any,
    coffee_id: str,
    *,
    target_xy: np.ndarray,
    yaw_deg: float,
    z: float,
    seed: int,
    sampling_attempt: int,
) -> dict[str, Any]:
    """把一只杯子放到已采样位置，并记录采样和放置信息。"""

    before = _object_joint_position(env, coffee_id)
    target = np.asarray([target_xy[0], target_xy[1], z], dtype=float)
    applied = _set_object_pose(
        env,
        coffee_id,
        target,
        quat_wxyz=_yaw_quaternion(yaw_deg),
    )
    return {
        "applied": applied,
        "before_pos": before.tolist(),
        "after_pos": target.tolist(),
        "sampled_center_xy": target_xy.tolist(),
        "center_sampling_bounds": {
            axis: list(bounds) for axis, bounds in COFFEE_INITIAL_REGION_BOUNDS.items()
        },
        "within_center_sampling_region": _xy_in_initial_region(target_xy),
        "yaw_deg": yaw_deg,
        "yaw_deg_range": list(COFFEE_INITIAL_YAW_BOUNDS[coffee_id]),
        "seed": seed,
        "sampling_attempt": sampling_attempt,
    }


def _yaw_quaternion(yaw_deg: float) -> np.ndarray:
    """将世界 Z 轴 yaw 角转换为 MuJoCo 使用的 wxyz 四元数。"""

    yaw = np.deg2rad(float(yaw_deg))
    return np.asarray([np.cos(0.5 * yaw), 0.0, 0.0, np.sin(0.5 * yaw)], dtype=float)


def _xy_in_initial_region(xy: np.ndarray, *, tolerance: float = 1e-9) -> bool:
    """检查杯子中心是否位于初始共同采样区域。"""

    point = np.asarray(xy, dtype=float).reshape(-1)
    if point.size < 2 or not np.all(np.isfinite(point[:2])):
        return False
    x_min, x_max = COFFEE_INITIAL_REGION_BOUNDS["x"]
    y_min, y_max = COFFEE_INITIAL_REGION_BOUNDS["y"]
    return bool(
        x_min - tolerance <= point[0] <= x_max + tolerance
        and y_min - tolerance <= point[1] <= y_max + tolerance
    )


def _object_joint_position(env: Any, object_name: str) -> np.ndarray:
    """读取 free joint 物品当前的位置部分并验证有效性。"""

    obj = env.objects[object_name]
    qpos = np.asarray(env.sim.data.get_joint_qpos(obj.joints[0]), dtype=float).reshape(-1)
    if qpos.size < 7 or not np.all(np.isfinite(qpos[:7])):
        raise ValueError(f"invalid free-joint pose for {object_name}")
    return qpos[:3].copy()


def _set_object_pose(
    env: Any,
    object_name: str,
    position: np.ndarray,
    *,
    quat_wxyz: np.ndarray,
) -> bool:
    """直接更新物品 free joint 位姿并同步 object_placements 缓存。"""

    obj = env.objects[object_name]
    qpos = np.asarray(env.sim.data.get_joint_qpos(obj.joints[0]), dtype=float).reshape(-1)
    if qpos.size < 7:
        return False
    updated = qpos.copy()
    updated[:3] = np.asarray(position, dtype=float)
    updated[3:7] = np.asarray(quat_wxyz, dtype=float)
    env.sim.data.set_joint_qpos(obj.joints[0], updated)
    if object_name in getattr(env, "object_placements", {}):
        env.object_placements[object_name] = (
            tuple(float(value) for value in updated[:3]),
            tuple(float(value) for value in updated[3:7]),
            obj,
        )
    return True


def _set_object_sliding_friction(
    env: Any,
    object_name: str,
    *,
    sliding_friction: float,
) -> dict[str, Any]:
    """更新杯子全部子几何体的滑动摩擦系数。"""

    if not np.isfinite(sliding_friction) or sliding_friction <= 0.0:
        raise ValueError("coffee_mug_sliding_friction must be positive")
    model = env.sim.model
    root = int(env.obj_body_id[object_name])
    parents = np.asarray(model.body_parentid, dtype=int)

    def belongs(body_id: int) -> bool:
        """判断给定 body 是否属于目标物品的 body 子树。"""

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
    before = [float(model.geom_friction[index, 0]) for index in geom_ids]
    for index in geom_ids:
        model.geom_friction[index, 0] = float(sliding_friction)
    return {
        "applied": bool(geom_ids),
        "geom_count": len(geom_ids),
        "sliding_before_min": None if not before else min(before),
        "sliding_before_max": None if not before else max(before),
        "sliding_after": float(sliding_friction),
    }


def _scale_object_mass(env: Any, object_name: str, *, scale: float) -> dict[str, Any]:
    """按比例缩放物品全部子 body 的质量和惯量。"""

    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("coffee_mug_mass_scale must be positive")
    model = env.sim.model
    root = int(env.obj_body_id[object_name])
    parents = np.asarray(model.body_parentid, dtype=int)

    def belongs(body_id: int) -> bool:
        """判断给定 body 是否属于需要缩放质量的物品子树。"""

        current = int(body_id)
        while current > 0:
            if current == root:
                return True
            current = int(parents[current])
        return current == root

    body_ids = [body_id for body_id in range(int(model.nbody)) if belongs(body_id)]
    before = [float(model.body_mass[body_id]) for body_id in body_ids]
    for body_id in body_ids:
        model.body_mass[body_id] *= scale
        model.body_inertia[body_id] *= scale
    return {
        "applied": bool(body_ids),
        "body_count": len(body_ids),
        "scale": float(scale),
        "total_before": float(sum(before)),
        "total_after": float(sum(before) * scale),
    }


__all__ = [
    "apply_fixed_coffee_machine_offset",
    "apply_fixed_initial_object_regions",
    "apply_robot_base_initial_state",
]


def initialize_episode(runtime: Any, episode: Any) -> InitializationReport:
    """Expose reset facts already applied by the task runtime."""
    del episode
    return InitializationReport.applied(facts=dict(getattr(runtime, "initialization", {}) or {}))
