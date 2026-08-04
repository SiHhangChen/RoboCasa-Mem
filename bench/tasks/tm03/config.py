"""Strict TM03 YAML boundary and physical-calibration gate."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from bench.tasks.tm03.domain.contracts import (
    CANONICAL_PROMPT,
    CANONICAL_PROMPT_SHA256,
    ENV_NAME,
    EXPECTED_TOASTER_OVEN_ASSET,
    TASK_ID,
    ActionConfig,
    ObserverConfig,
    RunContext,
    TaskConfig,
)


PACKAGE_ROOT = Path(__file__).resolve().parent
DEFAULT_TASK_CONFIG_FILE = PACKAGE_ROOT / "resources" / "task.yaml"
CALIBRATION_RESOURCE_NAMES = (
    "stove_knob_interaction_poses",
    "stove_knob_pinch_off_pose",
    "stove_knob_pinch_on_pose",
    "manual_base_path",
    "drawer_grasp_pose",
    "toaster_oven_interaction_poses",
    "toaster_oven_time_pinch_off_pose",
    "toaster_oven_time_pinch_on_pose",
    "toaster_oven_time_pinch_on_for_off_pose",
)


class TM03CalibrationError(RuntimeError):
    """Raised before simulator construction for incomplete physical resources."""


def load_task_config(path: str | Path = DEFAULT_TASK_CONFIG_FILE) -> TaskConfig:
    source = Path(path)
    payload = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, Mapping):
        raise ValueError("TM03 config root must be a mapping")
    allowed = {
        "schema_version",
        "task",
        "scene",
        "episode",
        "initial_state",
        "semantics",
        "appliance",
        "toaster_oven",
        "sensors",
        "resources",
        "navigation",
        "manipulation",
        "operations",
        "recording",
    }
    _reject_unknown(payload, allowed, "config sections")
    if int(payload.get("schema_version", 1)) != 1:
        raise ValueError("TM03 only supports schema_version=1")
    sections = {
        name: dict(_mapping(payload, name))
        for name in allowed
        if name != "schema_version"
    }
    task = sections["task"]
    if str(task.get("id")) != TASK_ID:
        raise ValueError(f"TM03 config must declare task.id={TASK_ID!r}")
    prompt = str(task.get("canonical_prompt", ""))
    declared_hash = str(task.get("canonical_prompt_sha256", ""))
    if prompt != CANONICAL_PROMPT:
        raise ValueError("TM03 canonical_prompt differs from its contract")
    if (
        declared_hash != CANONICAL_PROMPT_SHA256
        or hashlib.sha256(prompt.encode("utf-8")).hexdigest() != declared_hash
    ):
        raise ValueError("TM03 canonical_prompt_sha256 mismatch")
    scene = sections["scene"]
    if str(scene.get("env_name")) != ENV_NAME:
        raise ValueError(f"TM03 config must declare scene.env_name={ENV_NAME!r}")
    if (
        str(scene.get("expected_toaster_oven_asset"))
        != EXPECTED_TOASTER_OVEN_ASSET
    ):
        raise ValueError("TM03 must freeze ToasterOven059")
    toaster_oven = sections["toaster_oven"]
    for field in (
        "door_must_remain_closed",
        "food_must_remain_inside",
        "forbid_direct_state_write_after_reset",
        "forbid_knob_actuator_command_after_reset",
    ):
        if toaster_oven.get(field) is not True:
            raise ValueError(f"TM03 toaster_oven.{field} must be true")
    return TaskConfig(source_path=source, **sections)


def resolve_observer_config(config: TaskConfig) -> ObserverConfig:
    semantic = config.semantics
    appliance = config.appliance
    toaster_oven = config.toaster_oven
    result = ObserverConfig(
        drawer_open_threshold=float(semantic["drawer_open_threshold"]),
        drawer_closed_threshold=float(semantic["drawer_closed_threshold"]),
        inspect_dwell_steps=int(semantic["inspect_dwell_steps"]),
        button_pressed_threshold_m=float(
            semantic["completion_button_pressed_threshold_m"]
        ),
        button_released_threshold_m=float(
            semantic["completion_button_released_threshold_m"]
        ),
        button_release_dwell_steps=int(
            semantic["completion_button_release_dwell_steps"]
        ),
        off_commit_max_rad=float(appliance["off_commit_max_rad"]),
        burner_on_min_rad=float(appliance["burner_on_min_rad"]),
        on_commit_min_rad=float(appliance["on_commit_min_rad"]),
        commit_dwell_steps=int(appliance["commit_dwell_steps"]),
        min_gripper_contact_steps=int(appliance["min_gripper_contact_steps"]),
        contact_evidence_window_steps=int(appliance["contact_evidence_window_steps"]),
        min_interaction_motion_rad=float(appliance["min_interaction_motion_rad"]),
        max_unexplained_delta_rad_per_step=float(
            appliance["max_unexplained_delta_rad_per_step"]
        ),
        pot_burner_xy_threshold_m=float(semantic["pot_burner_xy_threshold_m"]),
        pot_translation_guard_m=float(semantic["pot_translation_guard_m"]),
        pot_tilt_guard_deg=float(semantic["pot_tilt_guard_deg"]),
        toaster_oven_off_commit_max_rad=float(
            toaster_oven["off_commit_max_rad"]
        ),
        toaster_oven_on_min_rad=float(toaster_oven["on_min_rad"]),
        toaster_oven_on_commit_min_rad=float(
            toaster_oven["on_commit_min_rad"]
        ),
        toaster_oven_commit_dwell_steps=int(toaster_oven["commit_dwell_steps"]),
        toaster_oven_min_gripper_contact_steps=int(
            toaster_oven["min_gripper_contact_steps"]
        ),
        toaster_oven_contact_evidence_window_steps=int(
            toaster_oven["contact_evidence_window_steps"]
        ),
        toaster_oven_min_interaction_motion_rad=float(
            toaster_oven["min_interaction_motion_rad"]
        ),
        toaster_oven_max_unexplained_delta_rad_per_step=float(
            toaster_oven["max_unexplained_delta_rad_per_step"]
        ),
    )
    if not (
        0.0 <= result.drawer_closed_threshold < result.drawer_open_threshold <= 1.0
    ):
        raise ValueError("TM03 drawer thresholds are invalid")
    if not (
        0.0
        <= result.toaster_oven_off_commit_max_rad
        < result.toaster_oven_on_min_rad
        <= result.toaster_oven_on_commit_min_rad
    ):
        raise ValueError("TM03 toaster-oven time-knob thresholds are invalid")
    for name in (
        "inspect_dwell_steps",
        "button_release_dwell_steps",
        "commit_dwell_steps",
        "min_gripper_contact_steps",
        "contact_evidence_window_steps",
        "toaster_oven_commit_dwell_steps",
        "toaster_oven_min_gripper_contact_steps",
        "toaster_oven_contact_evidence_window_steps",
    ):
        if int(getattr(result, name)) <= 0:
            raise ValueError(f"TM03 {name} must be positive")
    return result


def resolve_action_config(config: TaskConfig) -> ActionConfig:
    operations = config.operations
    return ActionConfig(
        knob_on_metadata=dict(_mapping(operations, "stove_knob_on")),
        knob_off_metadata=dict(_mapping(operations, "stove_knob_off")),
        drawer_open_metadata=dict(_mapping(operations, "drawer_open")),
        inspect_metadata=dict(_mapping(operations, "inspect")),
        drawer_close_metadata=dict(_mapping(operations, "drawer_close")),
        toaster_oven_start_metadata=dict(
            _mapping(operations, "toaster_oven_start")
        ),
        toaster_oven_stop_metadata=dict(
            _mapping(operations, "toaster_oven_stop")
        ),
        completion_button_metadata=dict(_mapping(operations, "completion_button")),
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
    if task_options:
        raise ValueError("TM03 route, drawer order, and prompt are seed/YAML-owned")
    return {}


def missing_calibration_resources(config: TaskConfig) -> tuple[str, ...]:
    missing: list[str] = []
    for name in CALIBRATION_RESOURCE_NAMES:
        path = config.path(config.resources.get(name))
        if path is None or not path.is_file():
            missing.append(name)
            continue
        if name == "manual_base_path":
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                missing.append(name)
                continue
            metadata = dict(payload.get("metadata") or {})
            if metadata.get("production_calibrated") is not True:
                missing.append(name)
        if name == "toaster_oven_interaction_poses":
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                missing.append(name)
                continue
            pose_files = dict(payload.get("pose_files") or {})
            calibrated = payload.get("calibrated") is True
            resolved = [path.parent / str(value) for value in pose_files.values()]
            if not calibrated or len(resolved) != 3 or not all(
                item.is_file() for item in resolved
            ):
                missing.append(name)
    return tuple(missing)


def require_production_calibration(config: TaskConfig) -> None:
    missing = missing_calibration_resources(config)
    if missing:
        raise TM03CalibrationError(
            "TM03 physical calibration is incomplete: " + ", ".join(missing)
        )


def _mapping(payload: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = payload.get(name, {})
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"TM03 {name} must be a mapping")
    return value


def _reject_unknown(payload: Mapping[str, Any], allowed: set[str], label: str) -> None:
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise ValueError(f"unknown TM03 {label}: " + ", ".join(unknown))


__all__ = [
    "CALIBRATION_RESOURCE_NAMES",
    "DEFAULT_TASK_CONFIG_FILE",
    "TM03CalibrationError",
    "load_task_config",
    "missing_calibration_resources",
    "require_production_calibration",
    "resolve_action_config",
    "resolve_observer_config",
    "resolve_run_context",
    "resolve_task_options",
]
