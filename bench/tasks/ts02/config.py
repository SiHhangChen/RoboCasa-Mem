"""Strict YAML, resource, and invocation boundary for TS02."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from math import isfinite
from pathlib import Path
from typing import Any

import yaml

from bench.tasks.ts02.domain.contracts import (
    ENV_NAME,
    TASK_ID,
    ActionConfig,
    InitialStateConfig,
    RobotBaseConfig,
    RobotBaseMode,
    RunContext,
    TaskConfig,
)


PACKAGE_ROOT = Path(__file__).resolve().parent
DEFAULT_TASK_CONFIG_FILE = PACKAGE_ROOT / "resources" / "task.yaml"
DEFAULT_INITIAL_STATION_ID = "drawer_station"


def load_task_config(path: str | Path = DEFAULT_TASK_CONFIG_FILE) -> TaskConfig:
    """Load all production settings and reject misspelled or stale fields."""

    config_path = Path(path)
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, Mapping):
        raise ValueError(f"TS02 config root must be a mapping: {config_path}")
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
            "recording",
        },
        "config sections",
    )
    if int(payload.get("schema_version", 1)) != 1:
        raise ValueError("TS02 only supports schema_version=1")

    task = dict(_mapping(payload, "task", config_path))
    _reject_unknown(
        task, {"id", "display_name", "description", "success_conditions"}, "task fields"
    )
    if str(task.get("id")) != TASK_ID:
        raise ValueError(f"TS02 config must declare task.id={TASK_ID!r}")

    scene = dict(_mapping(payload, "scene", config_path))
    _reject_unknown(
        scene,
        {
            "env_name",
            "fixed_scene",
            "layout_id",
            "style_id",
            "scene_seed",
            "fixture_names",
            "drawer_world_xyzs",
            "drawer_world_yaw_deg",
            "completion_button_world_xyz",
            "open_threshold",
            "closed_threshold",
        },
        "scene fields",
    )
    episode = dict(_mapping(payload, "episode", config_path))
    _reject_unknown(
        episode,
        {"demo_fixture_order", "target_fixture", "max_horizon"},
        "episode fields",
    )

    initial = _mapping(payload, "initial_state", config_path)
    _reject_unknown(initial, {"robot_base", "drawers"}, "initial_state fields")
    base_payload = _mapping(initial, "robot_base", config_path)
    _reject_unknown(
        base_payload,
        {"mode", "xy", "yaw", "station_id", "sample_index"},
        "initial_state.robot_base fields",
    )
    drawers = dict(_mapping(initial, "drawers", config_path))
    _reject_unknown(
        drawers, {"enabled", "closed_fraction"}, "initial_state.drawers fields"
    )

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
            "drawer_grasp_pose",
        },
        "resource fields",
    )

    navigation = dict(_mapping(payload, "navigation", config_path))
    _reject_unknown(
        navigation,
        {
            "kinematic_base_navigation",
            "base_action_frame",
            "base_translation_action_sign",
            "base_tolerance",
            "base_yaw_tolerance",
            "base_max_steps",
            "manual_base_path_enabled",
            "manual_base_path_start_index",
            "manual_base_path_xy_tolerance",
            "manual_base_path_yaw_tolerance",
            "base_single_axis_tracking",
            "base_terminal_yaw_distance",
            "base_snap_to_target_after_success",
            "base_velocity_profile_enabled",
            "base_profile_v_max",
            "base_profile_a_max",
            "base_profile_a_dec",
            "base_profile_v_min",
            "base_profile_w_max",
            "base_profile_alpha_max",
            "base_profile_alpha_dec",
            "base_profile_w_min",
            "use_station_astar_paths",
            "station_nav_use_docking",
            "station_nav_require_footprint",
            "use_station_base_poses",
        },
        "navigation fields",
    )

    manipulation = dict(_mapping(payload, "manipulation", config_path))
    _reject_unknown(
        manipulation,
        {"pick_place", "planner", "runtime", "static_scene"},
        "manipulation sections",
    )
    _reject_unknown(
        _mapping(manipulation, "pick_place", config_path),
        {
            "orientation_mode",
            "mug_orientation_mode",
            "pregrasp_height",
            "grasp_clearance",
            "lift_height",
            "mug_lift_height",
            "grasp_yaw_offset_deg",
            "place_height",
            "retreat_height",
            "retreat_x",
            "retreat_y",
            "release_offset_x",
            "release_offset_y",
            "post_release_clearance_distance",
            "post_release_clearance_z_offset",
        },
        "manipulation.pick_place fields",
    )
    _reject_unknown(
        _mapping(manipulation, "planner", config_path),
        {
            "conda_env",
            "subprocess_timeout_s",
            "planning_time",
            "time_step",
            "qpos_step",
            "rrt_range",
        },
        "manipulation.planner fields",
    )
    _reject_unknown(
        _mapping(manipulation, "runtime", config_path),
        {
            "max_joint_step",
            "settle_steps",
            "position_threshold",
            "orientation_threshold_deg",
            "qpos_threshold",
            "open_steps",
            "close_steps",
            "release_steps",
            "release_settle_steps",
        },
        "manipulation.runtime fields",
    )
    _reject_unknown(
        _mapping(manipulation, "static_scene", config_path),
        {
            "enabled",
            "validate_collisions",
            "fixture_keywords",
            "exclude_keywords",
            "spacing",
            "inflate",
            "point_radius",
            "faces",
            "point_cloud_stages",
            "min_z",
            "max_points",
        },
        "manipulation.static_scene fields",
    )

    operations = dict(_mapping(payload, "operations", config_path))
    _reject_unknown(
        operations,
        {"drawer_open", "inspect", "drawer_close", "completion_button"},
        "operation sections",
    )
    _reject_unknown(
        _mapping(operations, "drawer_open", config_path),
        {
            "actuation_mode",
            "open_fraction",
            "enforce_proximity",
            "max_base_distance",
            "hook_orientation_blend",
            "handle_side_offset",
            "handle_vertical_offset",
            "prehook_distance",
            "hook_offset",
            "pull_distance",
            "open_fraction_threshold",
            "gripper_steps",
            "grasp_pose_resource",
            "close_gripper_on_hook",
            "close_gripper_action",
            "close_gripper_steps",
        },
        "operations.drawer_open fields",
    )
    _reject_unknown(
        _mapping(operations, "inspect", config_path),
        {"dwell_steps"},
        "operations.inspect fields",
    )
    _reject_unknown(
        _mapping(operations, "drawer_close", config_path),
        {
            "actuation_mode",
            "closed_fraction",
            "close_overshoot",
            "closed_fraction_threshold",
            "return_arm_after_close",
        },
        "operations.drawer_close fields",
    )
    _reject_unknown(
        _mapping(operations, "completion_button", config_path),
        {
            "prepress_z_offset",
            "press_z_offset",
            "hold_steps",
            "gripper_action",
            "near_miss_position_tolerance",
            "press_depth_threshold",
            "release_depth_threshold",
            "release_dwell_steps",
            "require_physical_button_motion",
        },
        "operations.completion_button fields",
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
        "recording fields",
    )
    config = TaskConfig(
        source_path=config_path,
        task=task,
        scene=scene,
        episode=episode,
        initial_state=InitialStateConfig(
            robot_base=resolve_robot_base_config(**dict(base_payload)),
            drawers=drawers,
        ),
        sensors=sensors,
        resources=resources,
        navigation=navigation,
        manipulation=manipulation,
        operations=operations,
        recording=recording,
    )
    _validate_resources(config)
    return config


def resolve_robot_base_config(
    *,
    mode: str | RobotBaseMode = RobotBaseMode.PRESERVE_RESET_POSE,
    xy: Sequence[float] | None = None,
    yaw: float | None = None,
    station_id: str | None = None,
    sample_index: int | None = None,
) -> RobotBaseConfig:
    raw = mode.value if isinstance(mode, RobotBaseMode) else str(mode).strip().lower()
    try:
        resolved = RobotBaseMode(raw)
    except ValueError as exc:
        raise ValueError(f"unsupported TS02 robot-base mode: {raw!r}") from exc
    supplied = {
        "xy": xy,
        "yaw": yaw,
        "station_id": station_id,
        "sample_index": sample_index,
    }
    if resolved is RobotBaseMode.EXPLICIT_POSE:
        _reject_supplied(resolved, supplied, {"xy", "yaw"})
        if xy is None or isinstance(xy, (str, bytes)) or len(xy) != 2:
            raise ValueError("explicit_pose requires two-value robot_base.xy")
        resolved_xy = (float(xy[0]), float(xy[1]))
        if not all(isfinite(value) for value in resolved_xy):
            raise ValueError("robot_base.xy must be finite")
        return RobotBaseConfig(
            resolved, xy=resolved_xy, yaw=_finite(yaw, "robot_base.yaw")
        )
    if resolved is RobotBaseMode.MANUAL_PATH_SAMPLE:
        _reject_supplied(resolved, supplied, {"sample_index"})
        index = 0 if sample_index is None else int(sample_index)
        if index < 0:
            raise ValueError("manual_path_sample requires a non-negative sample_index")
        return RobotBaseConfig(resolved, sample_index=index)
    if resolved is RobotBaseMode.STATION_SAMPLE:
        _reject_supplied(resolved, supplied, {"station_id"})
        station = (
            DEFAULT_INITIAL_STATION_ID
            if station_id is None
            else str(station_id).strip()
        )
        if not station:
            raise ValueError("station_sample requires robot_base.station_id")
        return RobotBaseConfig(resolved, station_id=station)
    _reject_supplied(resolved, supplied, set())
    return RobotBaseConfig(resolved)


def resolve_action_config(*, operations: Mapping[str, Any]) -> ActionConfig:
    """Convert operation YAML into immutable stage metadata."""

    open_values = dict(_section(operations, "drawer_open"))
    close_values = dict(_section(operations, "drawer_close"))
    inspect_values = dict(_section(operations, "inspect"))
    button_values = dict(_section(operations, "completion_button"))
    open_mode = _actuation_mode(
        open_values.get("actuation_mode", "semantic"), "drawer_open"
    )
    close_mode = _actuation_mode(
        close_values.get("actuation_mode", "semantic"), "drawer_close"
    )
    open_fraction = float(open_values.get("open_fraction", 1.0))
    closed_fraction = float(close_values.get("closed_fraction", 0.0))
    if not 0.0 <= open_fraction <= 1.0 or not 0.0 <= closed_fraction <= 1.0:
        raise ValueError("TS02 semantic drawer fractions must be in [0, 1]")
    max_distance = float(open_values.get("max_base_distance", 1.0))
    if max_distance <= 0.0:
        raise ValueError("TS02 max_base_distance must be positive")
    open_metadata = {
        "actuation_mode": open_mode,
        "open_fraction": open_fraction,
        "enforce_proximity": bool(open_values.get("enforce_proximity", True)),
        "max_base_distance": max_distance,
        "hook_orientation_blend": float(
            open_values.get("hook_orientation_blend", 0.90)
        ),
        "handle_side_offset": float(open_values.get("handle_side_offset", 0.0)),
        "handle_vertical_offset": float(open_values.get("handle_vertical_offset", 0.0)),
        "prehook_distance": float(open_values.get("prehook_distance", 0.125)),
        "hook_offset": float(open_values.get("hook_offset", 0.005)),
        "pull_distance": float(open_values.get("pull_distance", 0.125)),
        "open_fraction_threshold": float(
            open_values.get("open_fraction_threshold", 0.70)
        ),
        "gripper_steps": int(open_values.get("gripper_steps", 25)),
        "grasp_pose_resource": str(
            open_values.get("grasp_pose_resource", "")
        ).strip(),
        "close_gripper_on_hook": bool(
            open_values.get("close_gripper_on_hook", False)
        ),
        "close_gripper_action": float(
            open_values.get("close_gripper_action", 1.0)
        ),
        "close_gripper_steps": int(open_values.get("close_gripper_steps", 25)),
    }
    if not 0.0 <= open_metadata["hook_orientation_blend"] <= 1.0:
        raise ValueError("TS02 hook_orientation_blend must be in [0, 1]")
    if any(
        open_metadata[name] <= 0.0
        for name in ("prehook_distance", "pull_distance", "open_fraction_threshold")
    ):
        raise ValueError("TS02 drawer pull distances and threshold must be positive")
    if open_metadata["gripper_steps"] < 0 or open_metadata["close_gripper_steps"] < 0:
        raise ValueError("TS02 drawer gripper step counts must be non-negative")
    if not -1.0 <= open_metadata["close_gripper_action"] <= 1.0:
        raise ValueError("TS02 close_gripper_action must be in [-1, 1]")
    close_metadata = {
        "actuation_mode": close_mode,
        "closed_fraction": closed_fraction,
        "close_overshoot": float(close_values.get("close_overshoot", 0.005)),
        "closed_fraction_threshold": float(
            close_values.get("closed_fraction_threshold", 0.08)
        ),
        "return_arm_after_close": bool(
            close_values.get("return_arm_after_close", False)
        ),
    }
    if close_metadata["close_overshoot"] < 0.0:
        raise ValueError("TS02 close_overshoot must be non-negative")
    if not 0.0 <= close_metadata["closed_fraction_threshold"] <= 1.0:
        raise ValueError("TS02 closed_fraction_threshold must be in [0, 1]")
    dwell_steps = int(inspect_values.get("dwell_steps", 0))
    if dwell_steps < 0:
        raise ValueError("TS02 inspect.dwell_steps must be non-negative")
    button_metadata = {
        "prepress_z_offset": float(button_values.get("prepress_z_offset", 0.08)),
        "press_z_offset": float(button_values.get("press_z_offset", -0.010)),
        "hold_steps": int(button_values.get("hold_steps", 12)),
        "gripper_action": float(button_values.get("gripper_action", 1.0)),
        "near_miss_position_tolerance": float(
            button_values.get("near_miss_position_tolerance", 0.09)
        ),
        "press_depth_threshold": float(
            button_values.get("press_depth_threshold", 0.004)
        ),
        "release_depth_threshold": float(
            button_values.get("release_depth_threshold", 0.001)
        ),
        "release_dwell_steps": int(button_values.get("release_dwell_steps", 2)),
        "require_physical_button_motion": bool(
            button_values.get("require_physical_button_motion", True)
        ),
    }
    if button_metadata["prepress_z_offset"] <= 0.0:
        raise ValueError("TS02 completion-button prepress offset must be positive")
    if button_metadata["press_z_offset"] >= 0.0:
        raise ValueError("TS02 completion-button press offset must be negative")
    if button_metadata["hold_steps"] < 1:
        raise ValueError("TS02 completion-button hold_steps must be positive")
    if not -1.0 <= button_metadata["gripper_action"] <= 1.0:
        raise ValueError("TS02 completion-button gripper_action must be in [-1, 1]")
    if button_metadata["near_miss_position_tolerance"] <= 0.0:
        raise ValueError("TS02 completion-button near-miss tolerance must be positive")
    if not 0.0 < button_metadata["press_depth_threshold"] <= 0.012:
        raise ValueError(
            "TS02 completion-button press_depth_threshold must be in (0, 0.012]"
        )
    if not (
        0.0
        <= button_metadata["release_depth_threshold"]
        < button_metadata["press_depth_threshold"]
    ):
        raise ValueError(
            "TS02 completion-button release threshold must be below press threshold"
        )
    if button_metadata["release_dwell_steps"] < 1:
        raise ValueError(
            "TS02 completion-button release_dwell_steps must be positive"
        )
    return ActionConfig(
        drawer_open_metadata=open_metadata,
        inspect_metadata={"dwell_steps": dwell_steps},
        drawer_close_metadata=close_metadata,
        completion_button_metadata=button_metadata,
    )


def resolve_run_context(
    *,
    split: str,
    env_split: str,
    variant: str,
    execute: bool,
    extra_metadata: Mapping[str, Any] | None = None,
) -> RunContext:
    return RunContext(
        split=str(split),
        env_split=str(env_split),
        variant=str(variant),
        execute=bool(execute),
        extra_metadata=dict(extra_metadata or {}),
    )


def resolve_task_options(task_options: Mapping[str, Any] | None) -> dict[str, Any]:
    """TS02 production decisions are seed/YAML owned, like WA01."""

    if task_options:
        raise ValueError("TS02 task options are YAML-owned")
    return {}


def _validate_resources(config: TaskConfig) -> None:
    for name in (
        "station_poses",
        "manual_base_path",
        "planner_script",
        "drawer_grasp_pose",
    ):
        path = config.path(config.resources.get(name))
        if path is None or not path.is_file():
            raise FileNotFoundError(f"missing TS02 resource {name!r}: {path}")
    station_path = config.path(config.resources["station_poses"])
    assert station_path is not None
    payload = json.loads(station_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("TS02 station resource must contain a JSON object")
    if str(payload.get("task")) != TASK_ID or str(payload.get("env_name")) != ENV_NAME:
        raise ValueError("TS02 station resource declares the wrong task or environment")


def _mapping(payload: Mapping[str, Any], name: str, path: Path) -> Mapping[str, Any]:
    value = payload.get(name, {})
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping in {path}")
    return value


def _section(payload: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = payload.get(name, {})
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"TS02 {name} must be a mapping")
    return value


def _reject_unknown(payload: Mapping[str, Any], allowed: set[str], label: str) -> None:
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise ValueError(f"unknown TS02 {label}: {', '.join(unknown)}")


def _reject_supplied(
    mode: RobotBaseMode,
    values: Mapping[str, Any],
    allowed: set[str],
) -> None:
    rejected = sorted(
        name
        for name, value in values.items()
        if name not in allowed and value is not None
    )
    if rejected:
        raise ValueError(
            f"robot_base.mode={mode.value!r} does not accept: {', '.join(rejected)}"
        )


def _finite(value: Any, name: str) -> float:
    if value is None:
        raise ValueError(f"{name} is required")
    result = float(value)
    if not isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _actuation_mode(value: Any, section: str) -> str:
    mode = str(value).strip().lower()
    if mode not in {"semantic", "contact"}:
        raise ValueError(f"TS02 {section}.actuation_mode must be semantic or contact")
    return mode


__all__ = [
    "DEFAULT_TASK_CONFIG_FILE",
    "load_task_config",
    "resolve_action_config",
    "resolve_robot_base_config",
    "resolve_run_context",
    "resolve_task_options",
]
