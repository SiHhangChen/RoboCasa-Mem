"""Strict configuration boundary for the TM01 semantic implementation."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from bench.tasks.tm01.domain.contracts import (
    CANONICAL_PROMPT,
    CANONICAL_PROMPT_SHA256,
    COMPLETION_BUTTON_NAMES,
    ENV_NAME,
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
)


class TM01CalibrationError(RuntimeError):
    """Raised before simulator construction when physical resources are missing."""


def load_task_config(path: str | Path = DEFAULT_TASK_CONFIG_FILE) -> TaskConfig:
    config_path = Path(path)
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, Mapping):
        raise ValueError("TM01 config root must be a mapping")
    allowed_sections = {
        "schema_version",
        "task",
        "scene",
        "episode",
        "initial_state",
        "semantics",
        "appliance",
        "sensors",
        "resources",
        "navigation",
        "manipulation",
        "operations",
        "recording",
    }
    _reject_unknown(payload, allowed_sections, "config sections")
    if int(payload.get("schema_version", 1)) != 1:
        raise ValueError("TM01 only supports schema_version=1")
    sections = {
        name: dict(_mapping(payload, name))
        for name in allowed_sections
        if name != "schema_version"
    }
    task = sections["task"]
    if str(task.get("id")) != TASK_ID:
        raise ValueError(f"TM01 config must declare task.id={TASK_ID!r}")
    prompt = str(task.get("canonical_prompt", ""))
    declared_hash = str(task.get("canonical_prompt_sha256", ""))
    actual_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    if prompt != CANONICAL_PROMPT:
        raise ValueError("TM01 canonical_prompt differs from the frozen contract")
    if declared_hash != CANONICAL_PROMPT_SHA256 or actual_hash != declared_hash:
        raise ValueError("TM01 canonical_prompt_sha256 mismatch")
    scene = sections["scene"]
    if str(scene.get("env_name")) != ENV_NAME:
        raise ValueError(f"TM01 config must declare scene.env_name={ENV_NAME!r}")
    if tuple(scene.get("task_added_button_names", ())) != COMPLETION_BUTTON_NAMES:
        raise ValueError(
            "TM01 must add exactly the drawer-side and stove-side completion buttons"
        )
    if str(scene.get("completion_button_visual_role")) != "green":
        raise ValueError("TM01 completion buttons must retain the green visual role")
    if str(scene.get("completion_button_success_rule")) != "any":
        raise ValueError("TM01 must accept a valid press of either completion button")
    appliance = sections["appliance"]
    if str(appliance.get("control_mode")) != "native_stove_knob_contact":
        raise ValueError("TM01 appliance control must use native stove-knob contact")
    for field in (
        "forbid_direct_qpos_write_after_reset",
        "forbid_knob_actuator_command_after_reset",
        "require_gripper_or_finger_contact",
        "require_all_other_burners_off",
    ):
        if appliance.get(field) is not True:
            raise ValueError(f"TM01 appliance.{field} must be true")
    return TaskConfig(source_path=config_path, **sections)


def resolve_observer_config(config: TaskConfig) -> ObserverConfig:
    semantic = config.semantics
    appliance = config.appliance
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
    )
    if not (
        0.0 <= result.drawer_closed_threshold < result.drawer_open_threshold <= 1.0
    ):
        raise ValueError("TM01 drawer thresholds are invalid")
    if not (
        0.0 < result.button_released_threshold_m < result.button_pressed_threshold_m
    ):
        raise ValueError("TM01 completion-button hysteresis is invalid")
    if not (
        0.0
        <= result.off_commit_max_rad
        < result.burner_on_min_rad
        < result.on_commit_min_rad
    ):
        raise ValueError("TM01 native-knob thresholds are invalid")
    for name in (
        "inspect_dwell_steps",
        "button_release_dwell_steps",
        "commit_dwell_steps",
        "min_gripper_contact_steps",
        "contact_evidence_window_steps",
    ):
        if int(getattr(result, name)) <= 0:
            raise ValueError(f"TM01 {name} must be positive")
    return result


def resolve_action_config(config: TaskConfig) -> ActionConfig:
    operations = config.operations
    return ActionConfig(
        knob_on_metadata=dict(_mapping(operations, "stove_knob_on")),
        knob_off_metadata=dict(_mapping(operations, "stove_knob_off")),
        drawer_open_metadata=dict(_mapping(operations, "drawer_open")),
        inspect_metadata=dict(_mapping(operations, "inspect")),
        drawer_close_metadata=dict(_mapping(operations, "drawer_close")),
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
        raise ValueError("TM01 route, drawer order, and prompt are seed/YAML-owned")
    return {}


def missing_calibration_resources(config: TaskConfig) -> tuple[str, ...]:
    missing: list[str] = []
    for name in CALIBRATION_RESOURCE_NAMES:
        path = config.path(config.resources.get(name))
        if path is None or not path.is_file():
            missing.append(name)
    return tuple(missing)


def require_production_calibration(config: TaskConfig) -> None:
    missing = missing_calibration_resources(config)
    if missing:
        raise TM01CalibrationError(
            "TM01 physical calibration is incomplete: " + ", ".join(missing)
        )


def _mapping(payload: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = payload.get(name, {})
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"TM01 {name} must be a mapping")
    return value


def _reject_unknown(
    payload: Mapping[str, Any],
    allowed: set[str],
    label: str,
) -> None:
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise ValueError(f"unknown TM01 {label}: " + ", ".join(unknown))


__all__ = [
    "CALIBRATION_RESOURCE_NAMES",
    "DEFAULT_TASK_CONFIG_FILE",
    "TM01CalibrationError",
    "load_task_config",
    "missing_calibration_resources",
    "require_production_calibration",
    "resolve_action_config",
    "resolve_observer_config",
    "resolve_run_context",
    "resolve_task_options",
]
