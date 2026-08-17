"""The strict YAML and task-option boundary for TS03."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from math import isfinite
from pathlib import Path
from typing import Any

import yaml

from bench.tasks.ts03.domain.contracts import (
    ActionConfig,
    InitialStateConfig,
    RobotBaseConfig,
    RobotBaseMode,
    RunContext,
    TASK_ID,
    TaskConfig,
)
from bench.tasks.ts03.domain.episode import (
    normalize_target_sugar_count,
    normalize_total_sugar_count,
    resolve_profile_selection,
)
from bench.tasks.ts03.domain.geometry import (
    SUGAR_RELEASE_REGION_NAMES,
    SUGAR_RELEASE_REGIONS_XY,
    sugar_release_offset_is_inside,
)


PACKAGE_ROOT = Path(__file__).resolve().parent
DEFAULT_TASK_CONFIG_FILE = PACKAGE_ROOT / "resources" / "task.yaml"


def load_task_config(path: str | Path = DEFAULT_TASK_CONFIG_FILE) -> TaskConfig:
    """Load one TS03 YAML file and reject undeclared fields."""

    config_path = Path(path)
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, Mapping):
        raise ValueError(f"TS03 config root must be a mapping: {config_path}")
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
    if int(payload.get("schema_version", 0)) != 1:
        raise ValueError("TS03 config requires schema_version=1")

    task = dict(_mapping(payload, "task", config_path))
    _reject_unknown(task, {"id", "display_name", "description", "success_conditions"}, "task fields")
    if str(task.get("id")) != TASK_ID:
        raise ValueError(f"TS03 config must declare task.id={TASK_ID!r}")

    scene = dict(_mapping(payload, "scene", config_path))
    _reject_unknown(
        scene,
        {
            "env_name",
            "fixed_scene",
            "layout_id",
            "style_id",
            "scene_seed",
            "sugar_plate_name",
            "receptacle_name",
            "sugar_cube_scale",
            "sugar_plate_scale",
            "sugar_plate_reference_offset",
            "sugar_plate_placement_size",
            "sugar_burner_id",
            "saucepan_xml",
            "sugar_cube_xml",
            "sugar_plate_xml",
            "liquid_rgba",
        },
        "scene fields",
    )
    episode = dict(_mapping(payload, "episode", config_path))
    _reject_unknown(
        episode,
        {
            "target_sugar_count",
            "total_sugar_count",
            "max_total_sugar_count",
            "sugar_name_prefix",
            "target_sugar_names",
            "max_horizon",
        },
        "episode fields",
    )
    _validate_episode_defaults(episode)

    initial = _mapping(payload, "initial_state", config_path)
    _reject_unknown(initial, {"robot_base", "objects"}, "initial_state fields")
    base_payload = _mapping(initial, "robot_base", config_path)
    _reject_unknown(
        base_payload,
        {"mode", "xy", "yaw", "sample_index", "station_id"},
        "initial_state.robot_base fields",
    )
    objects = dict(_mapping(initial, "objects", config_path))
    _reject_unknown(
        objects,
        {
            "enabled",
            "sugar_plate_edge_margin",
            "sugar_cube_sliding_friction",
            "sugar_cube_mass_scale",
        },
        "initial_state.objects fields",
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
        {"manual_base_path", "planner_script", "completion_button_position"},
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
            "base_waypoint_tolerance",
            "base_yaw_tolerance",
            "base_max_steps",
            "manual_base_path_enabled",
            "manual_base_path_start_index",
            "manual_base_path_xy_tolerance",
            "manual_base_path_yaw_tolerance",
            "manual_base_path_segment_yaw_threshold",
            "manual_base_path_max_rotate_steps",
            "manual_base_path_max_translate_steps",
            "manual_base_path_xy_gain",
            "manual_base_path_yaw_gain",
            "manual_base_path_nearest_yaw_weight",
            "manual_base_path_rotate_xy_tolerance",
            "base_velocity_profile_enabled",
            "base_profile_v_max",
            "base_profile_a_max",
            "base_profile_a_dec",
            "base_profile_v_min",
            "base_profile_w_max",
            "base_profile_alpha_max",
            "base_profile_alpha_dec",
            "base_profile_w_min",
            "base_profile_dock_v_max",
            "base_profile_dock_v_min",
            "base_lookahead_distance",
            "base_heading_tolerance",
            "base_terminal_yaw_distance",
            "base_terminal_heading_tolerance",
            "base_terminal_translation_gain_scale",
            "base_terminal_translation_min_action",
            "base_terminal_translation_max_action",
            "base_single_axis_tracking",
            "base_terminal_dock_enabled",
            "base_terminal_dock_standoff",
            "base_footprint_half_x",
            "base_footprint_half_y",
            "base_footprint_center_offset_x",
            "base_footprint_center_offset_y",
            "base_footprint_safety_margin",
            "base_stuck_window",
            "base_stuck_progress",
            "base_abort_on_stuck",
            "base_snap_to_target_after_success",
            "base_trace_stride",
            "base_candidate_mode",
            "base_candidate_offsets",
            "base_candidate_step",
            "base_candidate_range_x_values",
            "base_candidate_range_y_values",
            "base_candidate_yaw_mode",
            "base_candidate_yaw_offsets_deg",
            "base_candidate_max_count",
            "base_candidate_workspace_margin",
            "base_candidate_score_static_scene",
            "base_retry_max_attempts",
            "base_candidate_dry_run_entry_stage",
            "validate_base_path_collisions",
            "base_path_fixture_keywords",
            "base_path_exclude_keywords",
            "abort_on_base_path_collision",
            "base_holding_drop_tolerance",
            "use_station_astar_paths",
            "station_nav_grid_resolution",
            "station_nav_grid_inflation",
            "station_nav_grid_margin",
            "station_nav_use_docking",
            "station_nav_dock_standoff",
            "station_nav_require_footprint",
            "station_nav_allow_endpoint_snap",
            "station_nav_endpoint_snap_radius",
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
            "preserve_ik_branch_stages",
            "nearest_ik_samples",
            "nearest_ik_threshold",
            "nearest_ik_max_joint_delta",
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
        {
            "sugar_pick",
            "sugar_delivery",
            "completion_button",
            "return_arm",
            "handlers",
        },
        "operation sections",
    )
    _reject_unknown(
        _mapping(operations, "sugar_pick", config_path),
        {
            "orientation_mode",
            "pregrasp_height",
            "grasp_clearance",
            "lift_height",
            "preshape_gripper_steps",
            "descent_gripper_action",
            "close_gripper_action",
            "held_gripper_action",
            "grasp_yaw_offsets_deg",
            "allow_pregrasp_near_miss",
            "allow_grasp_near_miss",
            "pregrasp_settle_steps",
            "grasp_settle_steps",
            "lift_settle_steps",
        },
        "operations.sugar_pick fields",
    )
    _reject_unknown(
        _mapping(operations, "sugar_delivery", config_path),
        {
            "place_height",
            "retreat_height",
            "retreat_x",
            "retreat_y",
            "allow_place_near_miss",
            "place_near_miss_max_position_error",
            "entry_settle_steps",
            "place_settle_steps",
            "clearance_settle_steps",
            "retreat_settle_steps",
            "release_settle_steps",
            "placement_yaw_offsets_deg",
            "release_target_max_radius_m",
            "saucepan_translation_guard_m",
            "saucepan_tilt_guard_deg",
            "collision_fixture_keywords",
            "protected_object_names",
            "holding_allowed_fixture_keywords",
            "validate_collisions",
        },
        "operations.sugar_delivery fields",
    )
    _reject_unknown(
        _mapping(operations, "completion_button", config_path),
        {
            "prepress_z_offset",
            "press_z_offset",
            "hold_steps",
            "gripper_action",
            "near_miss_position_tolerance",
            "minimum_press_displacement_m",
        },
        "operations.completion_button fields",
    )
    _reject_unknown(
        _mapping(operations, "return_arm", config_path),
        {"settle_steps", "gripper_action"},
        "operations.return_arm fields",
    )
    _reject_unknown(
        _mapping(operations, "handlers", config_path),
        {
            "accept_sugar_delivery",
            "sugar_gripper_force_limit",
            "default_gripper_force_limit",
        },
        "operations.handlers fields",
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
            objects=objects,
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
    mode: str | RobotBaseMode = RobotBaseMode.MANUAL_PATH_SAMPLE,
    xy: Sequence[float] | None = None,
    yaw: float | None = None,
    sample_index: int | None = None,
    station_id: str | None = None,
) -> RobotBaseConfig:
    """Resolve one mutually exclusive reset-time Base mode."""

    raw = mode.value if isinstance(mode, RobotBaseMode) else str(mode).strip().lower()
    try:
        resolved = RobotBaseMode(raw)
    except ValueError as exc:
        raise ValueError(f"unsupported TS03 robot-base mode: {raw!r}") from exc
    supplied = {"xy": xy, "yaw": yaw, "sample_index": sample_index, "station_id": station_id}
    if resolved is RobotBaseMode.EXPLICIT_POSE:
        _reject_supplied(resolved, supplied, {"xy", "yaw"})
        if xy is None or isinstance(xy, (str, bytes)) or len(xy) != 2:
            raise ValueError("explicit_pose requires two-value robot_base.xy")
        values = (float(xy[0]), float(xy[1]))
        if not all(isfinite(value) for value in values):
            raise ValueError("robot_base.xy must be finite")
        return RobotBaseConfig(resolved, xy=values, yaw=_finite(yaw, "robot_base.yaw"))
    if resolved is RobotBaseMode.MANUAL_PATH_SAMPLE:
        _reject_supplied(resolved, supplied, {"sample_index"})
        index = 0 if sample_index is None else int(sample_index)
        if index < 0:
            raise ValueError("robot_base.sample_index must be non-negative")
        return RobotBaseConfig(resolved, sample_index=index)
    if resolved is RobotBaseMode.STATION_SAMPLE:
        _reject_supplied(resolved, supplied, {"station_id"})
        value = "saucepan_station" if station_id is None else str(station_id).strip()
        if not value:
            raise ValueError("robot_base.station_id must be non-empty")
        return RobotBaseConfig(resolved, station_id=value)
    _reject_supplied(resolved, supplied, set())
    return RobotBaseConfig(resolved)


def resolve_action_config(*, episode: Any, operations: Mapping[str, Any]) -> ActionConfig:
    """Turn YAML operation settings into immutable stage metadata."""

    pick_values = dict(_section(operations, "sugar_pick"))
    pick = {
        "orientation_mode": str(pick_values.get("orientation_mode", "object-y-topdown")),
        "pregrasp_height": float(pick_values.get("pregrasp_height", 0.15)),
        "grasp_clearance": float(pick_values.get("grasp_clearance", -0.018)),
        "lift_height": float(pick_values.get("lift_height", 0.24)),
        "preshape_gripper_action": 1.0,
        "preshape_gripper_steps": int(pick_values.get("preshape_gripper_steps", 5)),
        "descent_gripper_action": float(pick_values.get("descent_gripper_action", 0.0)),
        "close_gripper_action": float(pick_values.get("close_gripper_action", 1.0)),
        "held_gripper_action": float(pick_values.get("held_gripper_action", 1.0)),
        "grasp_yaw_offsets_deg": _float_sequence(
            pick_values.get("grasp_yaw_offsets_deg", (0.0, 90.0, 180.0, 270.0))
        ),
        "allow_pregrasp_near_miss": bool(pick_values.get("allow_pregrasp_near_miss", True)),
        "allow_grasp_near_miss": bool(pick_values.get("allow_grasp_near_miss", True)),
        "pregrasp_settle_steps": int(pick_values.get("pregrasp_settle_steps", 0)),
        "grasp_settle_steps": int(pick_values.get("grasp_settle_steps", 0)),
        "lift_settle_steps": int(pick_values.get("lift_settle_steps", 0)),
    }
    delivery_values = dict(_section(operations, "sugar_delivery"))
    placement_yaws = _float_sequence(
        delivery_values.get("placement_yaw_offsets_deg", (0.0, 90.0, 180.0, 270.0))
    )
    release_target_max_radius = float(
        delivery_values.get("release_target_max_radius_m", 0.060)
    )
    for region_name, offsets in SUGAR_RELEASE_REGIONS_XY.items():
        for offset in offsets:
            if not sugar_release_offset_is_inside(
                offset,
                max_radius_m=release_target_max_radius,
            ):
                raise ValueError(
                    f"TS03 release candidate {region_name}{offset} lies outside "
                    f"release_target_max_radius_m={release_target_max_radius}"
                )
    translation_guard = float(
        delivery_values.get("saucepan_translation_guard_m", 0.010)
    )
    tilt_guard = float(delivery_values.get("saucepan_tilt_guard_deg", 5.0))
    if translation_guard <= 0.0 or tilt_guard <= 0.0:
        raise ValueError("TS03 saucepan motion guards must be positive")
    protected_objects = _string_sequence(
        delivery_values.get("protected_object_names", ("saucepan",))
    )
    if not protected_objects:
        raise ValueError("TS03 protected_object_names must not be empty")
    delivery_base = {
        "target_name": "saucepan",
        "place_height": float(delivery_values.get("place_height", 0.07)),
        "retreat_height": float(delivery_values.get("retreat_height", 0.24)),
        "retreat_x": float(delivery_values.get("retreat_x", -0.10)),
        "retreat_y": float(delivery_values.get("retreat_y", -0.18)),
        "allow_place_near_miss": bool(delivery_values.get("allow_place_near_miss", True)),
        "place_near_miss_max_position_error": float(
            delivery_values.get("place_near_miss_max_position_error", 0.18)
        ),
        "entry_settle_steps": int(delivery_values.get("entry_settle_steps", 0)),
        "place_settle_steps": int(delivery_values.get("place_settle_steps", 0)),
        "clearance_settle_steps": int(delivery_values.get("clearance_settle_steps", 0)),
        "retreat_settle_steps": int(delivery_values.get("retreat_settle_steps", 0)),
        "release_settle_steps": int(delivery_values.get("release_settle_steps", 0)),
        "release_regions_xy": {
            name: [list(offset) for offset in SUGAR_RELEASE_REGIONS_XY[name]]
            for name in SUGAR_RELEASE_REGION_NAMES
        },
        "placement_yaw_offsets_deg": placement_yaws,
        "release_target_max_radius_m": release_target_max_radius,
        "saucepan_translation_guard_m": translation_guard,
        "saucepan_tilt_guard_deg": tilt_guard,
        "collision_options": {
            "validate_collisions": bool(
                delivery_values.get("validate_collisions", True)
            ),
            "fixture_keywords": str(
                delivery_values.get("collision_fixture_keywords", "stove")
            ),
            "protected_object_names": list(protected_objects),
            "holding_allowed_fixture_keywords": str(
                delivery_values.get("holding_allowed_fixture_keywords", "stove")
            ),
        },
    }
    delivery_by_index = {
        index: {
            **delivery_base,
            "release_region_order": list(
                episode.sugar_release_region_orders[index - 1]
            ),
        }
        for index in range(1, int(episode.target_sugar_count) + 1)
    }
    completion_values = dict(_section(operations, "completion_button"))
    completion = {
        "prepress_z_offset": float(
            completion_values.get("prepress_z_offset", 0.08)
        ),
        "press_z_offset": float(completion_values.get("press_z_offset", -0.010)),
        "hold_steps": int(completion_values.get("hold_steps", 12)),
        "gripper_action": float(completion_values.get("gripper_action", 1.0)),
        "near_miss_position_tolerance": float(
            completion_values.get("near_miss_position_tolerance", 0.09)
        ),
        "minimum_press_displacement_m": float(
            completion_values.get("minimum_press_displacement_m", 0.004)
        ),
    }
    if completion["hold_steps"] < 1:
        raise ValueError("TS03 completion-button hold_steps must be positive")
    if completion["minimum_press_displacement_m"] <= 0.0:
        raise ValueError(
            "TS03 completion-button minimum_press_displacement_m must be positive"
        )
    if completion["near_miss_position_tolerance"] <= 0.0:
        raise ValueError(
            "TS03 completion-button near_miss_position_tolerance must be positive"
        )
    if completion["prepress_z_offset"] <= completion["press_z_offset"]:
        raise ValueError("TS03 completion-button prepress must be above press")
    return_values = dict(_section(operations, "return_arm"))
    return ActionConfig(
        sugar_pick_metadata=pick,
        sugar_delivery_metadata_by_index=delivery_by_index,
        completion_button_metadata=completion,
        return_arm_metadata={
            "settle_steps": int(return_values.get("settle_steps", 80)),
            "gripper_action": float(return_values.get("gripper_action", -1.0)),
        },
        handler_settings=dict(_section(operations, "handlers")),
        grasp_yaw_offsets_deg=pick["grasp_yaw_offsets_deg"],
    )


def resolve_run_context(
    *,
    split: str,
    env_split: str,
    variant: str,
    execute: bool,
    force_manual_base_path_navigation: bool = False,
    extra_metadata: Mapping[str, Any] | None = None,
) -> RunContext:
    """Package invocation metadata without influencing episode randomness."""

    return RunContext(
        split=str(split),
        env_split=str(env_split),
        variant=str(variant),
        execute=bool(execute),
        force_manual_base_path_navigation=bool(force_manual_base_path_navigation),
        extra_metadata=dict(extra_metadata or {}),
    )


def resolve_task_options(task_options: Mapping[str, Any] | None) -> dict[str, Any]:
    """Resolve TS03 profile or explicit K/N overrides."""

    values = dict(task_options or {})
    allowed = {"profile_id", "target_sugar_count", "total_sugar_count"}
    unknown = sorted(set(values) - allowed)
    if unknown:
        raise ValueError("unknown TS03 task options: " + ", ".join(unknown))
    profile_value = values.get("profile_id")
    target_value = values.get("target_sugar_count")
    total_value = values.get("total_sugar_count")
    has_profile = profile_value is not None and str(profile_value).strip() != ""
    has_target = target_value is not None and target_value != ""
    has_total = total_value is not None and total_value != ""
    if not has_profile and not has_target and not has_total:
        return {}
    selected = resolve_profile_selection(
        seed=0,
        profile_id=profile_value,
        target_sugar_count=target_value,
        total_sugar_count=total_value,
    )
    result: dict[str, Any] = {}
    if has_profile:
        result["profile_id"] = selected.profile_id
    if has_target or has_total:
        result["target_sugar_count"] = selected.target_sugar_count
        result["total_sugar_count"] = selected.total_sugar_count
    return result


def _validate_episode_defaults(episode: Mapping[str, Any]) -> None:
    """Validate YAML K/N values before the episode resolver is called."""

    target = normalize_target_sugar_count(episode.get("target_sugar_count"))
    normalize_total_sugar_count(episode.get("total_sugar_count"), target_sugar_count=target)
    maximum = normalize_total_sugar_count(episode.get("max_total_sugar_count", 3))
    if maximum is None:
        raise ValueError("episode.max_total_sugar_count is required")
    if target is not None and maximum < target:
        raise ValueError("episode.max_total_sugar_count must satisfy max N >= K")
    if int(episode.get("max_horizon", 2400)) <= 0:
        raise ValueError("episode.max_horizon must be positive")


def _validate_resources(config: TaskConfig) -> None:
    for name in (
        "manual_base_path",
        "planner_script",
        "completion_button_position",
    ):
        path = config.path(config.resources.get(name))
        if path is None or not path.is_file():
            raise FileNotFoundError(f"missing TS03 resource {name!r}: {path}")


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
        raise ValueError(f"{name} must be a mapping")
    return value


def _reject_unknown(payload: Mapping[str, Any], allowed: set[str], label: str) -> None:
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise ValueError(f"unknown TS03 {label}: {', '.join(unknown)}")


def _reject_supplied(mode: RobotBaseMode, values: Mapping[str, Any], allowed: set[str]) -> None:
    rejected = sorted(name for name, value in values.items() if name not in allowed and value is not None)
    if rejected:
        raise ValueError(f"robot_base.mode={mode.value!r} does not accept: {', '.join(rejected)}")


def _finite(value: Any, name: str) -> float:
    if value is None:
        raise ValueError(f"{name} is required")
    result = float(value)
    if not isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _float_sequence(value: Any) -> tuple[float, ...]:
    if isinstance(value, str):
        raw = value.replace(";", ",").split(",")
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        raw = list(value)
    else:
        raw = [value]
    result = tuple(float(item) for item in raw if str(item).strip())
    if not result:
        raise ValueError("TS03 grasp_yaw_offsets_deg must not be empty")
    if not all(isfinite(item) for item in result):
        raise ValueError("TS03 angle sequences must be finite")
    return result


def _string_sequence(value: Any) -> tuple[str, ...]:
    raw = value.replace(";", ",").split(",") if isinstance(value, str) else value
    result = tuple(str(item).strip() for item in raw if str(item).strip())
    if len(set(result)) != len(result):
        raise ValueError("TS03 name sequences must not contain duplicates")
    return result


__all__ = [
    "DEFAULT_TASK_CONFIG_FILE",
    "load_task_config",
    "resolve_action_config",
    "resolve_robot_base_config",
    "resolve_run_context",
    "resolve_task_options",
]
