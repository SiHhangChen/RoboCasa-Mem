"""Strict task-owned configuration boundary for WR-03."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from math import isfinite
from pathlib import Path
from typing import Any

import yaml

from bench.tasks.wr03.domain.contracts import (
    ActionConfig,
    ENV_NAME,
    InitialStateConfig,
    RobotBaseConfig,
    RobotBaseMode,
    RunContext,
    TASK_ID,
    TaskConfig,
)
from bench.tasks.wr03.domain.episode import resolve_episode_plan


PACKAGE_ROOT = Path(__file__).resolve().parent
DEFAULT_TASK_CONFIG_FILE = PACKAGE_ROOT / "resources" / "task.yaml"
_RESOURCE_NAMES = {
    "item_grasp_pose",
    "lower_door_grasp_pose",
    "lower_preinsert_pose",
    "lower_retrieval_prepose",
    "lower_release_pose",
    "manual_base_path",
    "planner_script",
    "stations",
    "upper_door_grasp_pose",
}


def load_task_config(path: str | Path = DEFAULT_TASK_CONFIG_FILE) -> TaskConfig:
    config_path = Path(path).resolve()
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, Mapping):
        raise ValueError(f"WR-03 config root must be a mapping: {config_path}")
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
        raise ValueError("WR-03 config requires schema_version=1")
    task = dict(_mapping(payload, "task", config_path))
    if str(task.get("id")) != TASK_ID:
        raise ValueError(f"WR-03 config must declare task.id={TASK_ID!r}")
    scene = dict(_mapping(payload, "scene", config_path))
    if str(scene.get("env_name")) != ENV_NAME:
        raise ValueError(f"WR-03 config must declare scene.env_name={ENV_NAME!r}")
    initial = _mapping(payload, "initial_state", config_path)
    _reject_unknown(initial, {"robot_base", "robot_torso", "cabinets", "objects"}, "initial_state fields")
    robot_base = _mapping(initial, "robot_base", config_path)
    robot_torso = _mapping(initial, "robot_torso", config_path)
    torso_height = _finite(robot_torso.get("height_m"), "robot_torso.height_m")
    if not 0.0 <= torso_height <= 0.34:
        raise ValueError("WR-03 robot_torso.height_m must be within [0.0, 0.34]")
    resources = dict(_mapping(payload, "resources", config_path))
    if set(resources) != _RESOURCE_NAMES:
        raise ValueError("WR-03 resources must equal " + repr(sorted(_RESOURCE_NAMES)))
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
            robot_base=resolve_robot_base_config(**dict(robot_base)),
            robot_torso_height_m=torso_height,
            cabinets=dict(_mapping(initial, "cabinets", config_path)),
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


def resolve_robot_base_config(
    *,
    mode: str | RobotBaseMode = RobotBaseMode.EXPLICIT_POSE,
    xy: Sequence[float] | None = None,
    yaw: float | None = None,
) -> RobotBaseConfig:
    raw = mode.value if isinstance(mode, RobotBaseMode) else str(mode).strip().lower()
    try:
        resolved = RobotBaseMode(raw)
    except ValueError as exc:
        raise ValueError(f"unsupported WR-03 robot-base mode: {raw!r}") from exc
    if resolved is RobotBaseMode.PRESERVE_RESET_POSE:
        if xy is not None or yaw is not None:
            raise ValueError("preserve_reset_pose does not accept xy or yaw")
        return RobotBaseConfig(resolved)
    if xy is None or isinstance(xy, (str, bytes)) or len(xy) != 2 or yaw is None:
        raise ValueError("explicit_pose requires robot_base.xy and robot_base.yaw")
    values = (float(xy[0]), float(xy[1]), float(yaw))
    if not all(isfinite(value) for value in values):
        raise ValueError("WR-03 explicit robot base pose must be finite")
    return RobotBaseConfig(resolved, xy=values[:2], yaw=values[2])


def resolve_action_config(*, operations: Mapping[str, Any]) -> ActionConfig:
    storage_pick = _section(operations, "storage_pick")
    storage_place = _section(operations, "storage_place")
    retrieval_pick = _section(operations, "retrieval_pick")
    door = _section(operations, "door")
    return_arm = _section(operations, "return_arm")
    return ActionConfig(
        storage_pick_metadata=_pick_metadata(storage_pick, default_lift=0.10),
        storage_place_metadata=_place_metadata(storage_place),
        retrieval_pick_metadata=_retrieval_pick_metadata(retrieval_pick),
        door_metadata={
            "open_fraction": float(door.get("open_fraction", 1.0)),
            "closed_threshold": float(door.get("closed_threshold", 0.08)),
            "opened_threshold": float(door.get("opened_threshold", 0.70)),
            "control_mode": str(door.get("control_mode", "physical_handle")),
            "pregrasp_distance": float(door.get("pregrasp_distance", 0.14)),
            "retreat_distance": float(door.get("retreat_distance", door.get("pregrasp_distance", 0.14))),
            "open_distances": tuple(float(value) for value in door.get("open_distances", (0.10, 0.22, 0.36))),
            "close_deltas": tuple(float(value) for value in door.get("close_deltas", (0.12, 0.27, 0.36))),
            "gripper_open_steps": int(door.get("gripper_open_steps", 25)),
            "gripper_close_steps": int(door.get("gripper_close_steps", 60)),
            "gripper_release_steps": int(door.get("gripper_release_steps", 45)),
            "motion_settle_steps": max(0, int(door.get("motion_settle_steps", 2))),
            "final_motion_settle_steps": max(0, int(door.get("final_motion_settle_steps", 5))),
            "retreat_settle_steps": max(0, int(door.get("retreat_settle_steps", 5))),
        },
        return_arm_metadata={"settle_steps": max(0, int(return_arm.get("settle_steps", 2)))},
    )


def resolve_run_context(
    *, split: str, variant: str, execute: bool,
    extra_metadata: Mapping[str, Any] | None = None,
) -> RunContext:
    if str(variant) not in {"memory", "oracle_reveal"}:
        raise ValueError("WR-03 variant must be 'memory' or 'oracle_reveal'")
    return RunContext(str(split), str(variant), bool(execute), dict(extra_metadata or {}))


def resolve_task_options(task_options: Mapping[str, Any] | None) -> dict[str, Any]:
    if task_options:
        raise ValueError("WR-03 task options are seed/YAML-owned")
    return {}


def _pick_metadata(values: Mapping[str, Any], *, default_lift: float) -> dict[str, Any]:
    return {
        "orientation_mode": str(values.get("orientation_mode", "object-y-topdown")),
        "pregrasp_height": float(values.get("pregrasp_height", 0.15)),
        "grasp_clearance": float(values.get("grasp_clearance", -0.04)),
        "lift_height": float(values.get("lift_height", default_lift)),
        "preshape_gripper_steps": int(values.get("preshape_gripper_steps", 0)),
        "pregrasp_settle_steps": int(values.get("pregrasp_settle_steps", 2)),
        "grasp_settle_steps": int(values.get("grasp_settle_steps", 5)),
        "lift_settle_steps": int(values.get("lift_settle_steps", 5)),
        "manual_grasp_pregrasp_distance": float(values.get("manual_grasp_pregrasp_distance", 0.075)),
        "post_close_hold_steps": int(values.get("post_close_hold_steps", 25)),
        "held_gripper_action": float(values.get("held_gripper_action", 0.0)),
        "disable_collisions_during_transport": bool(values.get("disable_collisions_during_transport", True)),
    }


def _place_metadata(values: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "place_height": float(values.get("place_height", 0.0)),
        "retreat_height": float(values.get("retreat_height", 0.16)),
        "retreat_x": float(values.get("retreat_x", 0.0)),
        "retreat_y": float(values.get("retreat_y", -0.10)),
        "release_offset_x": float(values.get("release_offset_x", 0.0)),
        "release_offset_y": float(values.get("release_offset_y", 0.0)),
        "release_settle_steps": int(values.get("release_settle_steps", 8)),
        "placement_reference": str(values.get("placement_reference", "bbox_bottom_center")),
        "insertion_plan_attempts": max(1, int(values.get("insertion_plan_attempts", 3))),
    }


def _retrieval_pick_metadata(values: Mapping[str, Any]) -> dict[str, Any]:
    metadata = _pick_metadata(values, default_lift=0.12)
    metadata.update({
        "mode": str(values.get("mode", "cabinet_front_corridor")),
        "front_clearance_distance": float(values.get("front_clearance_distance", 0.10)),
        "front_pregrasp_distance": float(values.get("front_pregrasp_distance", 0.10)),
        "front_clearance_z_offset": float(values.get("front_clearance_z_offset", 0.10)),
        "grasp_z_offset": float(values.get("grasp_z_offset", 0.010)),
        "grasp_lateral_offset": float(values.get("grasp_lateral_offset", -0.0067)),
        "grasp_inward_offset": float(values.get("grasp_inward_offset", 0.0042)),
        "close_gripper_steps": max(1, int(values.get("close_gripper_steps", 6))),
    })
    return metadata


def _mapping(payload: Mapping[str, Any], name: str, path: Path) -> Mapping[str, Any]:
    value = payload.get(name)
    if not isinstance(value, Mapping):
        raise ValueError(f"WR-03 {name} must be a mapping: {path}")
    return value


def _section(payload: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = payload.get(name, {})
    if not isinstance(value, Mapping):
        raise ValueError(f"WR-03 operations.{name} must be a mapping")
    return value


def _resource(config: TaskConfig, name: str) -> Path:
    path = config.path(config.resources.get(name))
    if path is None or not path.is_file():
        raise FileNotFoundError(f"missing WR-03 resource {name!r}: {path}")
    return path


def _reject_unknown(payload: Mapping[str, Any], allowed: set[str], label: str) -> None:
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise ValueError(f"unknown WR-03 {label}: " + ", ".join(unknown))


def _finite(value: Any, label: str) -> float:
    if value is None:
        raise ValueError(f"{label} is required")
    result = float(value)
    if not isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


__all__ = [
    "DEFAULT_TASK_CONFIG_FILE", "load_task_config", "resolve_action_config",
    "resolve_robot_base_config", "resolve_run_context", "resolve_task_options",
]
