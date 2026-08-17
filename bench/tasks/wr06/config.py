"""Strict task-owned configuration boundary for WR-06."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from math import isfinite
from pathlib import Path
from typing import Any

import yaml

from bench.tasks.wr06.domain.contracts import (
    ActionConfig,
    ENV_NAME,
    HOME_REGIONS,
    InitialStateConfig,
    RobotBaseConfig,
    RobotBaseMode,
    RunContext,
    TASK_ID,
    TaskConfig,
)
from bench.tasks.wr06.domain.episode import resolve_episode_plan


PACKAGE_ROOT = Path(__file__).resolve().parent
DEFAULT_TASK_CONFIG_FILE = PACKAGE_ROOT / "resources" / "task.yaml"
_RESOURCE_NAMES = {"item_shelf_grasp", "plate_return_grasp", "planner_script", "stations"}


def load_task_config(path: str | Path = DEFAULT_TASK_CONFIG_FILE) -> TaskConfig:
    config_path = Path(path).resolve()
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, Mapping):
        raise ValueError(f"WR-06 config root must be a mapping: {config_path}")
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
        raise ValueError("WR-06 config requires schema_version=1")
    task = dict(_mapping(payload, "task", config_path))
    scene = dict(_mapping(payload, "scene", config_path))
    if str(task.get("id")) != TASK_ID:
        raise ValueError(f"WR-06 config must declare task.id={TASK_ID!r}")
    if str(scene.get("env_name")) != ENV_NAME:
        raise ValueError(f"WR-06 config must declare scene.env_name={ENV_NAME!r}")
    initial = _mapping(payload, "initial_state", config_path)
    _reject_unknown(initial, {"robot_base", "home_regions", "objects"}, "initial_state fields")
    resources = dict(_mapping(payload, "resources", config_path))
    if set(resources) != _RESOURCE_NAMES:
        raise ValueError("WR-06 resources must equal " + repr(sorted(_RESOURCE_NAMES)))
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
        episode=dict(_mapping(payload, "episode", config_path)),
        initial_state=InitialStateConfig(
            robot_base=resolve_robot_base_config(**dict(_mapping(initial, "robot_base", config_path))),
            home_regions=_resolve_home_regions(_mapping(initial, "home_regions", config_path)),
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
    resolve_episode_plan(seed=0, **dict(config.episode))
    for name in sorted(_RESOURCE_NAMES):
        _resource(config, name)
    return config


def _resolve_home_regions(values: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    if set(values) != set(HOME_REGIONS):
        raise ValueError("WR-06 home_regions must define exactly " + ", ".join(HOME_REGIONS))
    resolved: dict[str, dict[str, Any]] = {}
    for region in HOME_REGIONS:
        raw = values[region]
        if not isinstance(raw, Mapping):
            raise ValueError(f"WR-06 home region {region!r} must be a mapping")
        _reject_unknown(raw, {"shelf_level", "size", "pos", "margin"}, f"home region {region!r} fields")
        size = _finite_pair(raw.get("size"), label=f"{region}.size")
        pos = _finite_pair(raw.get("pos"), label=f"{region}.pos")
        level = int(raw.get("shelf_level", -1))
        margin = float(raw.get("margin", 0.04))
        if level < 0 or any(value <= 0.0 for value in size):
            raise ValueError(f"invalid WR-06 home region {region!r}")
        if not isfinite(margin) or margin < 0.0:
            raise ValueError(f"WR-06 {region}.margin must be non-negative")
        resolved[region] = {"shelf_level": level, "size": size, "pos": pos, "margin": margin}
    if resolved["shelf_lower"]["shelf_level"] >= resolved["shelf_upper"]["shelf_level"]:
        raise ValueError("WR-06 shelf_lower must be below shelf_upper")
    return resolved


def resolve_robot_base_config(
    *, mode: str | RobotBaseMode = RobotBaseMode.PRESERVE_RESET_POSE,
    xy: Sequence[float] | None = None, yaw: float | None = None,
) -> RobotBaseConfig:
    raw = mode.value if isinstance(mode, RobotBaseMode) else str(mode).strip().lower()
    try:
        resolved = RobotBaseMode(raw)
    except ValueError as exc:
        raise ValueError(f"unsupported WR-06 robot-base mode: {raw!r}") from exc
    if resolved is RobotBaseMode.PRESERVE_RESET_POSE:
        if xy is not None or yaw is not None:
            raise ValueError("preserve_reset_pose does not accept xy or yaw")
        return RobotBaseConfig(resolved)
    if xy is None or isinstance(xy, (str, bytes)) or len(xy) != 2 or yaw is None:
        raise ValueError("explicit_pose requires robot_base.xy and robot_base.yaw")
    values = (float(xy[0]), float(xy[1]), float(yaw))
    if not all(isfinite(value) for value in values):
        raise ValueError("WR-06 explicit robot base pose must be finite")
    return RobotBaseConfig(resolved, xy=values[:2], yaw=values[2])


def resolve_action_config(*, config: TaskConfig) -> ActionConfig:
    operations = config.operations
    shelf = {**_section(operations, "shelf_pick"), **_section(operations, "item_shelf_pick")}
    shelf["manual_grasp_pose_file"] = str(_resource(config, "item_shelf_grasp"))
    plate_pick = {
        **shelf,
        **_section(operations, "return_pick"),
        **_section(operations, "item_return_pick"),
        "manual_grasp_pose_file": str(_resource(config, "plate_return_grasp")),
    }
    plate_place = {
        **_section(operations, "plate_place"),
    }
    home_place = {
        **_section(operations, "home_place"),
    }
    return ActionConfig(
        shelf_pick_metadata=shelf,
        plate_pick_metadata=plate_pick,
        plate_place_metadata=plate_place,
        home_place_metadata=home_place,
        normalize_torso_metadata=dict(_section(operations, "normalize_transport_posture")),
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
        raise ValueError("WR-06 variant must be 'memory' or 'oracle_reveal'")
    return RunContext(str(split), str(variant), bool(execute), dict(extra_metadata or {}))


def resolve_task_options(task_options: Mapping[str, Any] | None) -> dict[str, Any]:
    if task_options:
        raise ValueError("WR-06 task options are seed/YAML-owned")
    return {}


def _finite_pair(value: Any, *, label: str) -> tuple[float, float]:
    if value is None or isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or len(value) != 2:
        raise ValueError(f"WR-06 {label} must contain two values")
    pair = (float(value[0]), float(value[1]))
    if not all(isfinite(item) for item in pair):
        raise ValueError(f"WR-06 {label} must be finite")
    return pair


def _mapping(payload: Mapping[str, Any], name: str, path: Path) -> Mapping[str, Any]:
    value = payload.get(name)
    if not isinstance(value, Mapping):
        raise ValueError(f"WR-06 {name} must be a mapping: {path}")
    return value


def _section(payload: Mapping[str, Any], name: str) -> dict[str, Any]:
    value = payload.get(name, {})
    if not isinstance(value, Mapping):
        raise ValueError(f"WR-06 operations.{name} must be a mapping")
    return dict(value)


def _resource(config: TaskConfig, name: str) -> Path:
    path = config.path(config.resources.get(name))
    if path is None or not path.is_file():
        raise FileNotFoundError(f"missing WR-06 resource {name!r}: {path}")
    return path


def _reject_unknown(payload: Mapping[str, Any], allowed: set[str], label: str) -> None:
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise ValueError(f"unknown WR-06 {label}: " + ", ".join(unknown))


__all__ = [
    "DEFAULT_TASK_CONFIG_FILE", "load_task_config", "resolve_action_config",
    "resolve_robot_base_config", "resolve_run_context", "resolve_task_options",
]
