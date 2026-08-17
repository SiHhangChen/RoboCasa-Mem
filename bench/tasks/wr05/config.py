"""Single strict configuration boundary for WR-05."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from math import isfinite
from pathlib import Path
from typing import Any

import yaml

from bench.tasks.wr05.domain.contracts import (
    HOME_REGIONS,
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
_RESOURCE_NAMES = {
    "ketchup_shelf_grasp",
    "mustard_shelf_grasp",
    "plate_ketchup_return_grasp",
    "plate_mustard_return_grasp",
    "planner_script",
    "stations",
}


def load_task_config(path: str | Path = DEFAULT_TASK_CONFIG_FILE) -> TaskConfig:
    config_path = Path(path).resolve()
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, Mapping):
        raise ValueError(f"WR-05 config root must be a mapping: {config_path}")
    _reject_unknown(
        payload,
        {
            "schema_version", "task", "scene", "episode", "initial_state",
            "sensors", "resources", "navigation", "manipulation", "operations",
            "semantics", "recording",
        },
        "config sections",
    )
    if int(payload.get("schema_version", 0)) != 1:
        raise ValueError("WR-05 config requires schema_version=1")
    task = dict(_mapping(payload, "task", config_path))
    scene = dict(_mapping(payload, "scene", config_path))
    if task.get("id") != TASK_ID:
        raise ValueError(f"WR-05 config must declare task.id={TASK_ID!r}")
    if scene.get("env_name") != "WR05ReturnCondimentPair":
        raise ValueError(f"WR-05 config must declare scene.env_name=WR05ReturnCondimentPair")
    initial = _mapping(payload, "initial_state", config_path)
    _reject_unknown(initial, {"robot_base", "home_regions", "objects"}, "initial_state fields")
    base = _mapping(initial, "robot_base", config_path)
    _reject_unknown(base, {"mode", "xy", "yaw"}, "robot_base fields")
    home_regions = _resolve_home_regions(
        _mapping(initial, "home_regions", config_path)
    )
    resources = dict(_mapping(payload, "resources", config_path))
    _reject_unknown(resources, _RESOURCE_NAMES, "resource names")
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
        episode=dict(_mapping(payload, "episode", config_path)),
        initial_state=InitialStateConfig(
            robot_base=resolve_robot_base_config(**dict(base)),
            home_regions=home_regions,
            objects=dict(_mapping(initial, "objects", config_path)),
        ),
        sensors=dict(_mapping(payload, "sensors", config_path)),
        resources=resources,
        navigation=dict(_mapping(payload, "navigation", config_path)),
        manipulation=dict(_mapping(payload, "manipulation", config_path)),
        operations=dict(_mapping(payload, "operations", config_path)),
        semantics=dict(_mapping(payload, "semantics", config_path)),
        recording=recording,
    )
    _validate_resources(config)
    return config


def _resolve_home_regions(
    values: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    if set(values) != set(HOME_REGIONS):
        raise ValueError(
            "WR-05 initial_state.home_regions must define exactly "
            + ", ".join(HOME_REGIONS)
        )
    resolved: dict[str, dict[str, Any]] = {}
    for region_name in HOME_REGIONS:
        raw = values[region_name]
        if not isinstance(raw, Mapping):
            raise ValueError(f"WR-05 home region {region_name!r} must be a mapping")
        _reject_unknown(
            raw,
            {"shelf_level", "size", "pos", "margin"},
            f"home region {region_name!r} fields",
        )
        shelf_level = int(raw.get("shelf_level", -1))
        size = _finite_pair(raw.get("size"), label=f"{region_name}.size")
        pos = _finite_pair(raw.get("pos"), label=f"{region_name}.pos")
        margin = float(raw.get("margin", 0.04))
        if shelf_level < 0:
            raise ValueError(f"WR-05 {region_name}.shelf_level must be non-negative")
        if size[0] <= 0.0 or size[1] <= 0.0:
            raise ValueError(f"WR-05 {region_name}.size must be positive")
        if any(abs(value) > 1.0 for value in pos):
            raise ValueError(f"WR-05 {region_name}.pos must stay within [-1, 1]")
        if not isfinite(margin) or margin < 0.0:
            raise ValueError(f"WR-05 {region_name}.margin must be non-negative")
        resolved[region_name] = {
            "shelf_level": shelf_level,
            "size": size,
            "pos": pos,
            "margin": margin,
        }
    if resolved["shelf_lower"]["shelf_level"] >= resolved["shelf_upper"]["shelf_level"]:
        raise ValueError("WR-05 shelf_lower must be below shelf_upper")
    return resolved


def _finite_pair(value: Any, *, label: str) -> tuple[float, float]:
    if (
        value is None
        or isinstance(value, (str, bytes))
        or not isinstance(value, Sequence)
        or len(value) != 2
    ):
        raise ValueError(f"WR-05 {label} must contain two values")
    pair = (float(value[0]), float(value[1]))
    if not all(isfinite(item) for item in pair):
        raise ValueError(f"WR-05 {label} must be finite")
    return pair


def resolve_robot_base_config(
    *,
    mode: str | RobotBaseMode = RobotBaseMode.PRESERVE_RESET_POSE,
    xy: Sequence[float] | None = None,
    yaw: float | None = None,
) -> RobotBaseConfig:
    raw = mode.value if isinstance(mode, RobotBaseMode) else str(mode).strip().lower()
    try:
        resolved = RobotBaseMode(raw)
    except ValueError as exc:
        raise ValueError(f"unsupported WR-05 robot-base mode: {raw!r}") from exc
    if resolved is RobotBaseMode.PRESERVE_RESET_POSE:
        if xy is not None or yaw is not None:
            raise ValueError("preserve_reset_pose does not accept xy or yaw")
        return RobotBaseConfig(resolved)
    if xy is None or isinstance(xy, (str, bytes)) or len(xy) != 2:
        raise ValueError("explicit_pose requires a two-value robot_base.xy")
    resolved_xy = (float(xy[0]), float(xy[1]))
    resolved_yaw = float(yaw) if yaw is not None else float("nan")
    if not all(isfinite(value) for value in (*resolved_xy, resolved_yaw)):
        raise ValueError("WR-05 explicit robot base pose must be finite")
    return RobotBaseConfig(resolved, xy=resolved_xy, yaw=resolved_yaw)


def resolve_action_config(*, config: TaskConfig) -> ActionConfig:
    operations = config.operations
    shelf_common = dict(_section(operations, "shelf_pick"))
    shelf_by_name = {
        "ketchup": {
            **shelf_common,
            **dict(_section(operations, "ketchup_shelf_pick")),
            "manual_grasp_pose_file": str(_resource(config, "ketchup_shelf_grasp")),
        },
        "mustard": {
            **shelf_common,
            **dict(_section(operations, "mustard_shelf_pick")),
            "manual_grasp_pose_file": str(_resource(config, "mustard_shelf_grasp")),
        },
    }
    return_common = dict(_section(operations, "return_pick"))
    ketchup_return_pick = {
        **shelf_by_name["ketchup"],
        **return_common,
        **dict(_section(operations, "ketchup_return_pick")),
    }
    # The plate-return grasp file is calibration provenance; the operation also
    # enables the WA-02-style multi-yaw side-grasp selector, which replaces that
    # single pose whenever the selector option is present.
    ketchup_return_pick["manual_grasp_pose_file"] = str(
        _resource(config, "plate_ketchup_return_grasp")
    )
    mustard_return_pick = {
        **shelf_by_name["mustard"],
        **return_common,
        **dict(_section(operations, "mustard_return_pick")),
    }
    mustard_return_pick["manual_grasp_pose_file"] = str(
        _resource(config, "plate_mustard_return_grasp")
    )
    plate_pick_by_name = {
        "ketchup": ketchup_return_pick,
        "mustard": mustard_return_pick,
    }
    plate_place_common = dict(_section(operations, "plate_place"))
    plate_place_by_region = {
        "plate_left": {**plate_place_common},
        "plate_right": {**plate_place_common},
    }
    home_common = dict(_section(operations, "home_place"))
    home_by_name = {
        "ketchup": {
            **home_common,
            **dict(_section(operations, "ketchup_home_place")),
        },
        "mustard": {
            **home_common,
            **dict(_section(operations, "mustard_home_place")),
        },
    }
    return ActionConfig(
        shelf_pick_by_condiment=shelf_by_name,
        plate_pick_by_condiment=plate_pick_by_name,
        plate_place_by_region=plate_place_by_region,
        home_place_by_condiment=home_by_name,
        lower_torso_after_upper_pick=dict(
            _section(operations, "lower_torso_after_upper_pick")
        ),
        return_arm_empty=dict(_section(operations, "return_arm_empty")),
        return_arm_holding=dict(_section(operations, "return_arm_holding")),
        return_navigation=dict(_section(operations, "return_navigation")),
        handler_settings=dict(_section(operations, "handlers")),
    )


def resolve_run_context(
    *, split: str, variant: str, execute: bool,
    extra_metadata: Mapping[str, Any] | None = None,
) -> RunContext:
    if str(variant) not in {"memory", "oracle_reveal"}:
        raise ValueError("WR-05 variant must be 'memory' or 'oracle_reveal'")
    return RunContext(str(split), str(variant), bool(execute), dict(extra_metadata or {}))


def _mapping(payload: Mapping[str, Any], name: str, path: Path) -> Mapping[str, Any]:
    value = payload.get(name)
    if not isinstance(value, Mapping):
        raise ValueError(f"WR-05 {name} must be a mapping: {path}")
    return value


def _section(payload: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = payload.get(name, {})
    if not isinstance(value, Mapping):
        raise ValueError(f"WR-05 operations.{name} must be a mapping")
    return value


def _resource(config: TaskConfig, name: str) -> Path:
    path = config.path(config.resources.get(name))
    if path is None or not path.is_file():
        raise FileNotFoundError(f"missing WR-05 resource {name!r}: {path}")
    return path


def _validate_resources(config: TaskConfig) -> None:
    missing = sorted(_RESOURCE_NAMES - set(config.resources))
    if missing:
        raise ValueError("missing WR-05 resources: " + ", ".join(missing))
    for name in sorted(_RESOURCE_NAMES):
        _resource(config, name)


def _reject_unknown(payload: Mapping[str, Any], allowed: set[str], label: str) -> None:
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise ValueError(f"unknown WR-05 {label}: " + ", ".join(unknown))


__all__ = [
    "DEFAULT_TASK_CONFIG_FILE",
    "load_task_config",
    "resolve_action_config",
    "resolve_robot_base_config",
    "resolve_run_context",
]
