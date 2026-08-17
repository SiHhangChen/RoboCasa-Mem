"""WA-03 唯一且严格校验的配置边界。"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from math import isfinite
from pathlib import Path
from typing import Any

import yaml

from bench.tasks.wa03.domain.contracts import (
    COFFEE_IDS,
    ActionConfig,
    InitialStateConfig,
    PoseDefaults,
    RobotBaseConfig,
    RobotBaseMode,
    RunContext,
    TASK_ID,
    TaskConfig,
)
from bench.tasks.wa03.domain.geometry import (
    compute_dining_regions,
    dining_region_config_summary,
    resolve_dining_region_config,
    sample_dining_targets,
)


PACKAGE_ROOT = Path(__file__).resolve().parent
DEFAULT_TASK_CONFIG_FILE = PACKAGE_ROOT / "resources" / "task.yaml"
DEFAULT_INITIAL_STATION_ID = "coffee_machine_station"
SUGAR_TARGET_MODES = frozenset({"unmarked", "marked"})


def resolve_task_options(values: Mapping[str, Any] | None) -> dict[str, Any]:
    """Validate the optional WA-03 marked compatibility mode."""

    options = dict(values or {})
    _reject_unknown(
        options,
        {"sugar_target_mode", "target_sugared_coffee_id"},
        "WA-03 task options",
    )
    resolved: dict[str, Any] = {}
    if "sugar_target_mode" in options:
        mode = str(options["sugar_target_mode"]).strip().lower()
        if mode not in SUGAR_TARGET_MODES:
            raise ValueError(f"unsupported WA-03 sugar_target_mode: {mode!r}")
        resolved["sugar_target_mode"] = mode
    if "target_sugared_coffee_id" in options:
        target = str(options["target_sugared_coffee_id"])
        if target not in COFFEE_IDS:
            raise ValueError(f"invalid WA-03 target coffee id: {target!r}")
        resolved["target_sugared_coffee_id"] = target
    return resolved


def load_task_config(path: str | Path = DEFAULT_TASK_CONFIG_FILE) -> TaskConfig:
    """读取并严格校验 WA-03 顶层 YAML 及所有资源引用。"""

    config_path = Path(path)
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, Mapping):
        raise ValueError(f"WA-03 config root must be a mapping: {config_path}")
    allowed = {
        "schema_version",
        "task",
        "scene",
        "episode",
        "initial_state",
        "sensors",
        "resources",
        "navigation",
        "manipulation",
        "operations",
        "recording",
    }
    _reject_unknown(payload, allowed, "config sections")
    task = dict(_mapping(payload, "task", config_path))
    if str(task.get("id")) != TASK_ID:
        raise ValueError(f"WA-03 config must declare task.id={TASK_ID!r}")
    initial_state = _mapping(payload, "initial_state", config_path)
    _reject_unknown(initial_state, {"robot_base", "objects"}, "initial_state fields")
    robot_base_payload = _mapping(initial_state, "robot_base", config_path)
    _reject_unknown(
        robot_base_payload,
        {"mode", "xy", "yaw", "sample_index", "station_id"},
        "initial_state.robot_base fields",
    )
    recording = dict(_mapping(payload, "recording", config_path))
    _reject_unknown(
        recording,
        {
            "record_depth",
            "depth_dtype",
            "depth_scale",
            "action_schema",
        },
        "task-owned recording fields",
    )
    config = TaskConfig(
        source_path=config_path,
        task=task,
        scene=dict(_mapping(payload, "scene", config_path)),
        episode=dict(_mapping(payload, "episode", config_path)),
        initial_state=InitialStateConfig(
            robot_base=resolve_robot_base_config(**dict(robot_base_payload)),
            objects=dict(_mapping(initial_state, "objects", config_path)),
        ),
        sensors=dict(_mapping(payload, "sensors", config_path)),
        resources=dict(_mapping(payload, "resources", config_path)),
        navigation=dict(_mapping(payload, "navigation", config_path)),
        manipulation=dict(_mapping(payload, "manipulation", config_path)),
        operations=dict(_mapping(payload, "operations", config_path)),
        recording=recording,
    )
    _validate_resources(config)
    return config


def resolve_robot_base_config(
    *,
    mode: str | RobotBaseMode = RobotBaseMode.MANUAL_PATH_SAMPLE,
    xy: Sequence[float] | None = None,
    yaw: float | None = None,
    sample_index: int | None = None,
    station_id: str | None = None,
) -> RobotBaseConfig:
    """校验互斥的 Base 初始化模式，拒绝模式外参数。"""

    raw = mode.value if isinstance(mode, RobotBaseMode) else str(mode).strip().lower()
    try:
        resolved_mode = RobotBaseMode(raw)
    except ValueError as exc:
        raise ValueError(f"unsupported WA-03 robot-base mode: {raw!r}") from exc
    supplied = {
        "xy": xy,
        "yaw": yaw,
        "sample_index": sample_index,
        "station_id": station_id,
    }
    if resolved_mode is RobotBaseMode.EXPLICIT_POSE:
        _reject_supplied(resolved_mode, supplied, allowed={"xy", "yaw"})
        if xy is None or isinstance(xy, (str, bytes)) or len(xy) != 2:
            raise ValueError("explicit_pose requires two-value robot_base.xy")
        resolved_xy = (float(xy[0]), float(xy[1]))
        resolved_yaw = _finite(yaw, "robot_base.yaw")
        if not all(isfinite(value) for value in resolved_xy):
            raise ValueError("robot_base.xy must be finite")
        return RobotBaseConfig(resolved_mode, xy=resolved_xy, yaw=resolved_yaw)
    if resolved_mode is RobotBaseMode.MANUAL_PATH_SAMPLE:
        _reject_supplied(resolved_mode, supplied, allowed={"sample_index"})
        index = 0 if sample_index is None else int(sample_index)
        if index < 0:
            raise ValueError("robot_base.sample_index must be non-negative")
        return RobotBaseConfig(resolved_mode, sample_index=index)
    if resolved_mode is RobotBaseMode.STATION_SAMPLE:
        _reject_supplied(resolved_mode, supplied, allowed={"station_id"})
        resolved_station = DEFAULT_INITIAL_STATION_ID if station_id is None else str(station_id).strip()
        if not resolved_station:
            raise ValueError("robot_base.station_id must be non-empty")
        return RobotBaseConfig(resolved_mode, station_id=resolved_station)
    _reject_supplied(resolved_mode, supplied, allowed=set())
    return RobotBaseConfig(resolved_mode)


def load_pose_defaults(config: TaskConfig) -> PoseDefaults:
    """读取杯子和糖块校准 JSON，并提取操作默认位姿。"""

    cup_path = _required_resource(config, "cup_place_poses")
    sugar_path = _required_resource(config, "sugar_cube_poses")
    cup = _read_json_object(cup_path)
    sugar = _read_json_object(sugar_path)
    if str(cup.get("task")) != TASK_ID or str(sugar.get("task")) != TASK_ID:
        raise ValueError("WA-03 pose files must declare task=WA-03")
    return PoseDefaults(
        cup_place_file=cup_path,
        sugar_cube_file=sugar_path,
        dining_place=_operation(cup, "place_at_dining"),
        sugar_delivery=_operation(sugar, "deliver_to_coffee"),
        sugar_pick=_operation(sugar, "pick"),
    )


def resolve_action_config(
    *,
    episode: Any,
    pose_defaults: PoseDefaults,
    operations: Mapping[str, Any],
) -> ActionConfig:
    """将 YAML 操作参数覆盖到校准默认值并生成有效动作配置。"""

    sugar_pick_settings = _section(operations, "sugar_pick")
    sugar_delivery_settings = _section(operations, "sugar_delivery")
    dining_settings = _section(operations, "dining_place")
    transport_settings = _section(operations, "coffee_transport")
    final_settings = _section(operations, "final_selection")
    return_settings = _section(operations, "return_arm")
    if bool(transport_settings.get("use_manual_grasp", False)):
        raise ValueError("WA-03 production architecture supports top-down mug grasp only")

    sugar_pick = _resolve_sugar_pick(sugar_pick_settings, pose_defaults.sugar_pick)
    sugar_delivery = _resolve_sugar_delivery(
        sugar_delivery_settings,
        pose_defaults.sugar_delivery,
    )
    (
        dining_by_mug,
        dining_region_config,
        dining_regions,
        sampled_targets,
    ) = _resolve_dining(
        dining_settings,
        pose_defaults.dining_place,
        final_slot_assignment=episode.final_slot_assignment,
        episode_seed=episode.seed,
    )
    transport_pick = {
        coffee_id: {
            "orientation_mode": "object-y-topdown",
            "pregrasp_height": float(transport_settings.get("topdown_pregrasp_height", 0.15)),
            "grasp_clearance": float(transport_settings.get("topdown_grasp_clearance", -0.05)),
            "pregrasp_settle_steps": int(transport_settings.get("pregrasp_settle_steps", 2)),
            "grasp_settle_steps": int(transport_settings.get("grasp_settle_steps", 5)),
            "lift_settle_steps": int(transport_settings.get("lift_settle_steps", 5)),
            "preshape_gripper_steps": 0,
            "descent_gripper_action": -1.0,
            "close_gripper_action": 1.0,
            "held_gripper_action": 1.0,
        }
        for coffee_id in COFFEE_IDS
    }
    mug_settings = _section(transport_settings, "mugs")
    transport_lift: dict[str, dict[str, Any]] = {}
    for coffee_id in COFFEE_IDS:
        values = _section(mug_settings, coffee_id)
        transport_lift[coffee_id] = {
            "stage": f"raise_delivery_{coffee_id}_for_transport",
            "lift_height": float(values.get("lift_height", 0.10)),
            "settle_steps": int(values.get("lift_settle_steps", 5)),
        }
    final_pick = {
        "orientation_mode": str(final_settings.get("orientation_mode", "object-y-topdown")),
        "pregrasp_height": float(final_settings.get("pregrasp_height", 0.15)),
        "grasp_clearance": float(final_settings.get("grasp_clearance", -0.05)),
        "lift_height": float(final_settings.get("lift_height", 0.12)),
        "grasp_yaw_offset_deg": float(final_settings.get("grasp_yaw_offset_deg", 0.0)),
        "preshape_gripper_steps": int(final_settings.get("preshape_gripper_steps", 0)),
        "descent_gripper_action": float(final_settings.get("descent_gripper_action", -1.0)),
        "close_gripper_action": float(final_settings.get("close_gripper_action", 1.0)),
        "held_gripper_action": float(final_settings.get("held_gripper_action", 1.0)),
        "pregrasp_settle_steps": int(final_settings.get("pregrasp_settle_steps", 2)),
        "grasp_settle_steps": int(final_settings.get("grasp_settle_steps", 5)),
        "lift_settle_steps": int(final_settings.get("lift_settle_steps", 5)),
    }
    return ActionConfig(
        sugar_pick_metadata=sugar_pick,
        sugar_delivery_metadata=sugar_delivery,
        dining_place_metadata_by_mug=dining_by_mug,
        dining_region_config=dining_region_config,
        dining_regions_by_slot=dining_regions,
        sampled_dining_targets_by_slot=sampled_targets,
        transport_pick_metadata_by_mug=transport_pick,
        transport_lift_metadata_by_mug=transport_lift,
        final_selection_pick_metadata=final_pick,
        return_arm_metadata={"settle_steps": max(0, int(return_settings.get("settle_steps", 2)))},
        handler_settings=dict(_section(operations, "handlers")),
    )


def resolve_run_context(
    *,
    split: str,
    variant: str,
    execute: bool,
    force_manual_base_path_navigation: bool,
    extra_metadata: Mapping[str, Any] | None = None,
) -> RunContext:
    """封装 split、variant 和 execute 等本次调用信息。"""

    return RunContext(
        split=str(split),
        variant=str(variant),
        execute=bool(execute),
        force_manual_base_path_navigation=bool(force_manual_base_path_navigation),
        extra_metadata=dict(extra_metadata or {}),
    )


def _resolve_sugar_pick(settings: Mapping[str, Any], pose: Mapping[str, Any]) -> dict[str, Any]:
    """合并糖块抓取参数并生成 Pick Stage metadata。"""

    return {
        "orientation_mode": "object-y-topdown",
        "pregrasp_height": float(settings.get("pregrasp_height", pose.get("pregrasp_height", 0.15))),
        "grasp_clearance": float(settings.get("grasp_clearance", pose.get("grasp_clearance", -0.011))),
        "preshape_gripper_action": 1.0,
        "preshape_gripper_steps": int(settings.get("preshape_gripper_steps", 4)),
        "descent_gripper_action": 0.0,
        "close_gripper_action": 1.0,
        "held_gripper_action": 1.0,
        "pregrasp_settle_steps": int(settings.get("pregrasp_settle_steps", pose.get("pregrasp_settle_steps", 10))),
        "grasp_settle_steps": int(settings.get("grasp_settle_steps", pose.get("grasp_settle_steps", 10))),
        "lift_settle_steps": int(settings.get("lift_settle_steps", pose.get("lift_settle_steps", 10))),
    }


def _resolve_sugar_delivery(settings: Mapping[str, Any], pose: Mapping[str, Any]) -> dict[str, Any]:
    """合并糖块投递与释放阶段的参数。"""

    def value(name: str, default: Any) -> Any:
        """按 YAML、校准值、代码兜底的顺序读取单个字段。"""

        return settings.get(name, pose.get(name, default))

    return {
        "place_height": float(value("place_height", 0.08)),
        "retreat_height": float(value("retreat_height", 0.08)),
        "retreat_x": float(value("retreat_x", 0.0)),
        "retreat_y": float(value("retreat_y", 0.0)),
        "entry_settle_steps": int(value("entry_settle_steps", 2)),
        "place_settle_steps": int(value("place_settle_steps", 5)),
        "clearance_settle_steps": int(value("clearance_settle_steps", 2)),
        "retreat_settle_steps": int(value("retreat_settle_steps", 2)),
        "release_settle_steps": int(value("release_settle_steps", 5)),
    }


def _resolve_dining(
    settings: Mapping[str, Any],
    pose: Mapping[str, Any],
    *,
    final_slot_assignment: Mapping[str, str],
    episode_seed: int,
) -> tuple[dict[str, dict[str, Any]], Any, dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    """解析三块平行餐桌区域并为三杯生成各自稳定的放置 metadata。"""

    values = {**dict(pose), **dict(settings)}
    region_config = resolve_dining_region_config(_section(values, "dining_regions"))
    regions = compute_dining_regions(region_config)
    sampled = sample_dining_targets(region_config, episode_seed=episode_seed)
    base = {
        "target_kind": str(values.get("target_kind", "dining")),
        "place_height": float(values.get("place_height", 0.0)),
        "retreat_height": float(values.get("retreat_height", 0.08)),
        "retreat_x": float(values.get("retreat_x", 0.0)),
        "retreat_y": float(values.get("retreat_y", 0.0)),
        "release_offset_x": float(values.get("release_offset_x", 0.0)),
        "release_offset_y": float(values.get("release_offset_y", 0.0)),
        "post_release_clearance_distance": float(values.get("post_release_clearance_distance", 0.08)),
        "post_release_clearance_z_offset": float(values.get("post_release_clearance_z_offset", 0.0)),
        "placement_reference": str(values.get("placement_reference", "object_origin")),
        "dining_region_config": dining_region_config_summary(region_config),
        "entry_settle_steps": int(values.get("entry_settle_steps", 2)),
        "place_settle_steps": int(values.get("place_settle_steps", 5)),
        "clearance_settle_steps": int(values.get("clearance_settle_steps", 2)),
        "retreat_settle_steps": int(values.get("retreat_settle_steps", 2)),
        "release_settle_steps": int(values.get("release_settle_steps", 10)),
        "allow_place_near_miss": bool(values.get("allow_place_near_miss", True)),
        "allow_place_qpos_near_miss": bool(values.get("allow_place_qpos_near_miss", True)),
        "place_near_miss_max_position_error": float(values.get("place_near_miss_max_position_error", 0.08)),
        "place_near_miss_max_qpos_error": float(values.get("place_near_miss_max_qpos_error", 0.22)),
    }
    optional_keys = (
        "placement_yaw_offset_deg",
        "placement_yaw_offsets_deg",
        "placement_approach_angles_deg",
        "placement_entry_standoff",
        "placement_exit_standoff",
        "placement_entry_z_offset",
        "placement_exit_z_offset",
    )
    for key in optional_keys:
        if values.get(key) is not None:
            base[key] = values[key]
    yaw_by_mug = _section(values, "placement_yaw_offsets_by_mug_deg")
    approach_by_mug = _section(values, "placement_approach_angles_by_mug_deg")
    by_mug: dict[str, dict[str, Any]] = {}
    for coffee_id in COFFEE_IDS:
        slot = str(final_slot_assignment[coffee_id])
        metadata = dict(base)
        metadata["dining_target_slot"] = slot
        metadata["dining_region"] = dict(regions[slot])
        metadata["dining_sampled_target"] = dict(sampled[slot])
        if coffee_id in yaw_by_mug:
            metadata.pop("placement_yaw_offset_deg", None)
            metadata["placement_yaw_offsets_deg"] = yaw_by_mug[coffee_id]
        if coffee_id in approach_by_mug:
            metadata["placement_approach_angles_deg"] = approach_by_mug[coffee_id]
        by_mug[coffee_id] = metadata
    return by_mug, region_config, regions, sampled


def _read_json_object(path: Path) -> dict[str, Any]:
    """读取必须以 JSON object 为根节点的校准资源。"""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON resource must contain an object: {path}")
    return payload


def _operation(payload: Mapping[str, Any], name: str) -> dict[str, Any]:
    """从 pose JSON 的 operations 中提取指定操作。"""

    return dict(_section(_section(payload, "operations"), name))


def _required_resource(config: TaskConfig, name: str) -> Path:
    """解析并验证一个必须存在的任务资源文件。"""

    path = config.path(config.resources.get(name))
    if path is None or not path.is_file():
        raise FileNotFoundError(f"missing WA-03 resource {name!r}: {path}")
    return path


def _validate_resources(config: TaskConfig) -> None:
    """在构建环境前一次性验证所有生产资源均存在。"""

    for name in ("cup_place_poses", "sugar_cube_poses", "station_poses", "manual_base_path", "planner_script"):
        _required_resource(config, name)


def _mapping(payload: Mapping[str, Any], name: str, path: Path) -> Mapping[str, Any]:
    """从顶层配置读取必须为 mapping 的分区。"""

    value = payload.get(name, {})
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping in {path}")
    return value


def _section(payload: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    """从已解析字典读取必须为 mapping 的子分区。"""

    value = payload.get(name, {})
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def _reject_unknown(payload: Mapping[str, Any], allowed: set[str], label: str) -> None:
    """发现未声明字段时立即失败，防止拼写错误被静默忽略。"""

    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise ValueError(f"unknown WA-03 {label}: {', '.join(unknown)}")


def _reject_supplied(mode: RobotBaseMode, values: Mapping[str, Any], *, allowed: set[str]) -> None:
    """拒绝与当前 Base 初始化模式不相容的附加字段。"""

    rejected = sorted(name for name, value in values.items() if name not in allowed and value is not None)
    if rejected:
        raise ValueError(f"robot_base.mode={mode.value!r} does not accept: {', '.join(rejected)}")


def _finite(value: Any, name: str) -> float:
    """将必填数值转换为有限浮点数。"""

    if value is None:
        raise ValueError(f"{name} is required")
    result = float(value)
    if not isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


__all__ = [
    "DEFAULT_TASK_CONFIG_FILE",
    "load_pose_defaults",
    "load_task_config",
    "resolve_action_config",
    "resolve_robot_base_config",
    "resolve_run_context",
]
