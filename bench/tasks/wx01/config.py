"""Single strict configuration boundary for production WX-01."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from math import isfinite
from pathlib import Path
from typing import Any

import yaml

from bench.tasks.wx01.domain.contracts import (
    TASK_ID,
    ActionConfig,
    InitialStateConfig,
    RobotBaseConfig,
    RobotBaseMode,
    RunContext,
    StartStateConfig,
    TaskConfig,
)


PACKAGE_ROOT = Path(__file__).resolve().parent
DEFAULT_TASK_CONFIG_FILE = PACKAGE_ROOT / "resources" / "task.yaml"
DEFAULT_INITIAL_STATION_ID = "tool_origin_station"


def load_task_config(path: str | Path = DEFAULT_TASK_CONFIG_FILE) -> TaskConfig:
    """Load WX-01 YAML, reject unknown fields and validate local resources."""

    config_path = Path(path)
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, Mapping):
        raise ValueError(f"WX-01 config root must be a mapping: {config_path}")
    _reject_unknown(
        payload,
        {
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
            "semantics",
            "recording",
        },
        "config sections",
    )
    if int(payload.get("schema_version", 0)) != 1:
        raise ValueError("WX-01 config requires schema_version=1")

    task = dict(_mapping(payload, "task", config_path))
    _reject_unknown(
        task, {"id", "display_name", "description", "success_conditions"}, "task fields"
    )
    if str(task.get("id")) != TASK_ID:
        raise ValueError(f"WX-01 config must declare task.id={TASK_ID!r}")

    scene = dict(_mapping(payload, "scene", config_path))
    _reject_unknown(
        scene,
        {
            "env_name",
            "fixed_scene",
            "layout_id",
            "style_id",
            "scene_seed",
            "blocking_object_scale",
            "blocking_object_placement_size",
            "blocking_object_placement_pos",
            "blocking_object_rotation_deg",
        },
        "scene fields",
    )
    episode = dict(_mapping(payload, "episode", config_path))
    _reject_unknown(
        episode,
        {
            "start_mode",
            "temporary_targets",
            "wipe_sampling",
            "cup_material_candidates",
            "sponge_type_candidates",
            "max_horizon",
        },
        "episode fields",
    )

    initial = _mapping(payload, "initial_state", config_path)
    _reject_unknown(
        initial, {"robot_base", "objects", "short_start"}, "initial_state fields"
    )
    base_payload = _mapping(initial, "robot_base", config_path)
    _reject_unknown(
        base_payload,
        {"mode", "xy", "yaw", "sample_index", "station_id"},
        "initial_state.robot_base fields",
    )
    objects = dict(_mapping(initial, "objects", config_path))
    _reject_unknown(
        objects,
        {"enabled", "blocking_object", "cleaning_tool"},
        "initial_state.objects fields",
    )
    start_payload = _mapping(initial, "short_start", config_path)
    _reject_unknown(
        start_payload,
        {
            "mode",
            "setup_operation",
            "cleaning_tool",
        },
        "initial_state.short_start fields",
    )
    cleaning_tool_start = _mapping(start_payload, "cleaning_tool", config_path)
    _reject_unknown(
        cleaning_tool_start,
        {
            "lift_height",
            "transport_probe_height",
            "max_end_effector_distance",
            "max_relative_offset_drift",
            "minimum_motion_ratio",
        },
        "initial_state.short_start.cleaning_tool fields",
    )
    start = StartStateConfig(
        mode=str(start_payload.get("mode", "short_preplaced_blocking_pregrasped_tool")),
        setup_operation=str(start_payload.get("setup_operation", "physical_pick")),
        cleaning_tool_lift_height=float(cleaning_tool_start.get("lift_height", 0.20)),
        transport_probe_height=float(
            cleaning_tool_start.get("transport_probe_height", 0.03)
        ),
        max_end_effector_distance=float(
            cleaning_tool_start.get("max_end_effector_distance", 0.10)
        ),
        max_relative_offset_drift=float(
            cleaning_tool_start.get("max_relative_offset_drift", 0.02)
        ),
        minimum_motion_ratio=float(
            cleaning_tool_start.get("minimum_motion_ratio", 0.70)
        ),
    )
    if start.mode != "short_preplaced_blocking_pregrasped_tool":
        raise ValueError(
            "WX-01 production config supports the six-stage short start only"
        )
    if start.setup_operation != "physical_pick":
        raise ValueError("WX-01 short start requires setup_operation='physical_pick'")
    if start.cleaning_tool_lift_height <= 0.0:
        raise ValueError("cleaning_tool_lift_height must be positive")
    if start.transport_probe_height <= 0.0:
        raise ValueError("transport_probe_height must be positive")
    if start.max_end_effector_distance <= 0.0:
        raise ValueError("max_end_effector_distance must be positive")
    if start.max_relative_offset_drift <= 0.0:
        raise ValueError("max_relative_offset_drift must be positive")
    if not 0.0 < start.minimum_motion_ratio <= 1.0:
        raise ValueError("minimum_motion_ratio must be in (0, 1]")

    sensors = dict(_mapping(payload, "sensors", config_path))
    _reject_unknown(
        sensors,
        {
            "cameras",
            "wrist_rgb_camera",
            "width",
            "height",
            "depth",
            "use_camera_observations",
            "has_renderer",
            "has_offscreen_renderer",
        },
        "sensor fields",
    )
    resources = dict(_mapping(payload, "resources", config_path))
    _reject_unknown(
        resources,
        {
            "station_poses",
            "manual_base_path",
            "planner_script",
            "cleaning_tool_grasp_pose",
            "blocking_cup_grasp_candidates",
        },
        "resource fields",
    )
    semantics = dict(_mapping(payload, "semantics", config_path))
    _reject_unknown(
        semantics,
        {
            "origin_threshold",
            "wipe_radius",
            "contact_separation",
            "minimum_contact_count",
            "minimum_wipe_sweep",
            "release_distance",
            "stain_color_rgba",
            "stain_circle_radius",
            "stain_z_offset",
        },
        "semantic fields",
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
        scene=scene,
        episode=episode,
        initial_state=InitialStateConfig(
            robot_base=resolve_robot_base_config(**dict(base_payload)),
            objects=objects,
            short_start=start,
        ),
        sensors=sensors,
        resources=resources,
        navigation=dict(_mapping(payload, "navigation", config_path)),
        manipulation=dict(_mapping(payload, "manipulation", config_path)),
        operations=dict(_mapping(payload, "operations", config_path)),
        semantics=semantics,
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
    """Resolve exactly one Base initialization mode."""

    raw = mode.value if isinstance(mode, RobotBaseMode) else str(mode).strip().lower()
    try:
        resolved = RobotBaseMode(raw)
    except ValueError as exc:
        raise ValueError(f"unsupported WX-01 robot-base mode: {raw!r}") from exc
    supplied = {
        "xy": xy,
        "yaw": yaw,
        "sample_index": sample_index,
        "station_id": station_id,
    }
    if resolved is RobotBaseMode.EXPLICIT_POSE:
        _reject_supplied(resolved, supplied, allowed={"xy", "yaw"})
        if xy is None or isinstance(xy, (str, bytes)) or len(xy) != 2:
            raise ValueError("explicit_pose requires two-value robot_base.xy")
        resolved_xy = (float(xy[0]), float(xy[1]))
        resolved_yaw = _finite(yaw, "robot_base.yaw")
        if not all(isfinite(value) for value in resolved_xy):
            raise ValueError("robot_base.xy must be finite")
        return RobotBaseConfig(resolved, xy=resolved_xy, yaw=resolved_yaw)
    if resolved is RobotBaseMode.MANUAL_PATH_SAMPLE:
        _reject_supplied(resolved, supplied, allowed={"sample_index"})
        index = 0 if sample_index is None else int(sample_index)
        if index < 0:
            raise ValueError("robot_base.sample_index must be non-negative")
        return RobotBaseConfig(resolved, sample_index=index)
    if resolved is RobotBaseMode.STATION_SAMPLE:
        _reject_supplied(resolved, supplied, allowed={"station_id"})
        station = (
            DEFAULT_INITIAL_STATION_ID
            if station_id is None
            else str(station_id).strip()
        )
        if not station:
            raise ValueError("robot_base.station_id must be non-empty")
        return RobotBaseConfig(resolved, station_id=station)
    _reject_supplied(resolved, supplied, allowed=set())
    return RobotBaseConfig(resolved)


def resolve_action_config(
    *, episode: Any, operations: Mapping[str, Any]
) -> ActionConfig:
    """Strictly resolve metadata used by the six workflow stages."""

    _reject_unknown(
        operations,
        {
            "wipe",
            "cleaning_tool_place",
            "return_arm",
            "blocking_pick",
            "blocking_origin_place",
            "handlers",
            "pick_proxy",
        },
        "operation sections",
    )
    wipe = _strict_section(
        operations,
        "wipe",
        {
            "plan_execute_wipe_sweep",
            "contact_z_offset",
            "settle_steps",
            "allow_final_qpos_near_miss",
            "max_final_qpos_error",
        },
    )
    tool_place = _strict_section(
        operations,
        "cleaning_tool_place",
        {
            "place_height",
            "retreat_height",
            "retreat_x",
            "retreat_y",
            "post_release_clearance_z_offset",
            "target_max_base_x",
        },
    )
    return_arm = _strict_section(
        operations, "return_arm", {"settle_steps", "gripper_action"}
    )
    blocking_pick = _strict_section(
        operations,
        "blocking_pick",
        {
            "grasp_strategy",
            "lift_height",
            "blocking_object_max_base_x",
        },
    )
    origin_place = _strict_section(
        operations,
        "blocking_origin_place",
        {
            "place_height",
            "retreat_height",
            "retreat_x",
            "retreat_y",
            "placement_object_upright",
            "placement_approach_angles_deg",
            "placement_entry_standoff",
            "placement_entry_z_offset",
            "placement_exit_standoff",
            "placement_exit_z_offset",
            "post_release_clearance_distance",
            "post_release_clearance_z_offset",
            "entry_settle_steps",
            "place_settle_steps",
            "clearance_settle_steps",
            "retreat_settle_steps",
            "require_pre_release_check",
            "pre_release_xy_tolerance",
            "pre_release_z_tolerance",
            "pre_release_upright_threshold_deg",
            "target_max_base_x",
        },
    )
    handlers = _strict_section(
        operations,
        "handlers",
        {"dry_run_wipe_semantic_proxy", "accessible_place_dry_run_only"},
    )
    pick_proxy = _strict_section(
        operations, "pick_proxy", {"dry_run_only", "blocking_object_max_base_x"}
    )
    return ActionConfig(
        wipe_metadata={
            "target_pose_attr": "original_blocking_object_pose",
            "wipe_offsets_xy": [list(value) for value in episode.wipe_offsets_xy],
            "wipe_sampling": dict(episode.wipe_metadata),
            **dict(wipe),
        },
        cleaning_tool_place_metadata={
            "target_name": "post_wipe_dining_sponge",
            **dict(tool_place),
        },
        return_arm_metadata={
            "stage": "return_after_post_wipe_sponge_place",
            **dict(return_arm),
        },
        blocking_pick_metadata=dict(blocking_pick),
        blocking_origin_place_metadata={
            "target_name": "blocking_origin",
            **dict(origin_place),
        },
        handler_settings=dict(handlers),
        pick_proxy_settings=dict(pick_proxy),
    )


def resolve_run_context(
    *,
    split: str,
    variant: str,
    execute: bool,
    extra_metadata: Mapping[str, Any] | None = None,
) -> RunContext:
    """Keep invocation metadata separate from episode sampling."""

    return RunContext(
        split=str(split),
        variant=str(variant),
        execute=bool(execute),
        extra_metadata=dict(extra_metadata or {}),
    )


def _validate_resources(config: TaskConfig) -> None:
    required = {
        "station_poses",
        "manual_base_path",
        "planner_script",
        "cleaning_tool_grasp_pose",
        "blocking_cup_grasp_candidates",
    }
    missing = required - set(config.resources)
    if missing:
        raise ValueError("missing WX-01 resources: " + ", ".join(sorted(missing)))
    for name in required:
        path = config.path(config.resources[name])
        if path is None or not path.is_file():
            raise FileNotFoundError(f"missing WX-01 resource {name!r}: {path}")


def _mapping(payload: Mapping[str, Any], name: str, path: Path) -> Mapping[str, Any]:
    value = payload.get(name)
    if not isinstance(value, Mapping):
        raise ValueError(f"WX-01 config section {name!r} must be a mapping: {path}")
    return value


def _strict_section(
    payload: Mapping[str, Any], name: str, allowed: set[str]
) -> Mapping[str, Any]:
    value = payload.get(name, {})
    if not isinstance(value, Mapping):
        raise ValueError(f"WX-01 operations.{name} must be a mapping")
    _reject_unknown(value, allowed, f"operations.{name} fields")
    return value


def _reject_unknown(payload: Mapping[str, Any], allowed: set[str], label: str) -> None:
    unknown = set(payload) - allowed
    if unknown:
        raise ValueError(f"unknown WX-01 {label}: " + ", ".join(sorted(unknown)))


def _reject_supplied(
    mode: RobotBaseMode, values: Mapping[str, Any], *, allowed: set[str]
) -> None:
    invalid = sorted(
        name
        for name, value in values.items()
        if value is not None and name not in allowed
    )
    if invalid:
        raise ValueError(
            f"robot_base mode {mode.value!r} does not allow: " + ", ".join(invalid)
        )


def _finite(value: Any, label: str) -> float:
    resolved = float(value)
    if not isfinite(resolved):
        raise ValueError(f"{label} must be finite")
    return resolved


__all__ = [
    "DEFAULT_TASK_CONFIG_FILE",
    "load_task_config",
    "resolve_action_config",
    "resolve_robot_base_config",
    "resolve_run_context",
]
