"""Strict TM04 YAML boundary and physical-calibration gate."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from .domain.contracts import (
    CANONICAL_PROMPT,
    CANONICAL_PROMPT_SHA256,
    ENV_NAME,
    SUGAR_NAMES,
    TARGET_SUGAR_COUNT,
    TASK_ID,
    TOTAL_SUGAR_COUNT,
    ActionConfig,
    ObserverConfig,
    RunContext,
    TaskConfig,
)


PACKAGE_ROOT = Path(__file__).resolve().parent
DEFAULT_TASK_CONFIG_FILE = PACKAGE_ROOT / "resources" / "task.yaml"
CALIBRATION_RESOURCE_NAMES = (
    "manual_base_path",
    "cabinet_station_poses",
    "cabinet_door_pose",
    "can_grasp_pose",
    "can_retrieval_prepose",
    "can_place_pose",
    "sugar_pick_poses",
    "sugar_delivery_poses",
    "sugar_scene_placements",
)


class TM04CalibrationError(RuntimeError):
    """Raised before physical execution while TM04 calibration is incomplete."""


def load_task_config(path: str | Path = DEFAULT_TASK_CONFIG_FILE) -> TaskConfig:
    source = Path(path)
    payload = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, Mapping):
        raise ValueError("TM04 config root must be a mapping")
    allowed = {
        "schema_version", "task", "scene", "episode", "initial_state", "semantics",
        "sensors", "resources", "navigation", "manipulation", "operations", "recording",
    }
    _reject_unknown(payload, allowed, "config sections")
    if int(payload.get("schema_version", 1)) != 1:
        raise ValueError("TM04 only supports schema_version=1")
    sections = {
        name: dict(_mapping(payload, name))
        for name in allowed
        if name != "schema_version"
    }
    task = sections["task"]
    if str(task.get("id")) != TASK_ID:
        raise ValueError(f"TM04 config must declare task.id={TASK_ID!r}")
    prompt = str(task.get("canonical_prompt", ""))
    prompt_hash = str(task.get("canonical_prompt_sha256", ""))
    if prompt != CANONICAL_PROMPT:
        raise ValueError("TM04 canonical_prompt differs from its contract")
    if prompt_hash != CANONICAL_PROMPT_SHA256 or hashlib.sha256(prompt.encode()).hexdigest() != prompt_hash:
        raise ValueError("TM04 canonical_prompt_sha256 mismatch")
    scene = sections["scene"]
    if str(scene.get("env_name")) != ENV_NAME:
        raise ValueError(f"TM04 config must declare scene.env_name={ENV_NAME!r}")
    if (int(scene.get("layout_id", -1)), int(scene.get("style_id", -1))) != (15, 34):
        raise ValueError("TM04 must preserve TM02 layout 15 / style 34")
    episode = sections["episode"]
    if (int(episode.get("target_sugar_count", -1)), int(episode.get("total_sugar_count", -1))) != (
        TARGET_SUGAR_COUNT,
        TOTAL_SUGAR_COUNT,
    ):
        raise ValueError("TM04 must freeze K=2, N=3")
    if tuple(scene.get("sugar_names", ())) != SUGAR_NAMES:
        raise ValueError(f"TM04 sugar_names must equal {SUGAR_NAMES}")
    config = TaskConfig(source_path=source, **sections)
    placements_path = config.path(config.resources.get("sugar_scene_placements"))
    if placements_path is not None and placements_path.is_file():
        placements = json.loads(placements_path.read_text(encoding="utf-8"))
        if str(placements.get("task_id")) != TASK_ID:
            raise ValueError("TM04 sugar scene placements declare the wrong task")
        for name in ("sugar_plate_world_xyz", "hot_chocolate_world_xyz"):
            declared = tuple(float(value) for value in scene.get(name, ()))
            calibrated = tuple(float(value) for value in placements.get(name, ()))
            if len(declared) != 3 or declared != calibrated:
                raise ValueError(
                    f"TM04 scene.{name} differs from sugar_scene_placements"
                )
    return config


def resolve_observer_config(config: TaskConfig) -> ObserverConfig:
    values = config.semantics
    result = ObserverConfig(
        cabinet_open_threshold=float(values["cabinet_open_threshold"]),
        cabinet_closed_threshold=float(values["cabinet_closed_threshold"]),
        button_pressed_threshold_m=float(values["button_pressed_threshold_m"]),
        button_released_threshold_m=float(values["button_released_threshold_m"]),
        button_release_dwell_steps=int(values["button_release_dwell_steps"]),
        can_place_xy_tolerance_m=float(values["can_place_xy_tolerance_m"]),
        pot_water_xy_threshold_m=float(values["pot_water_xy_threshold_m"]),
        pot_translation_guard_m=float(values["pot_translation_guard_m"]),
        pot_tilt_guard_deg=float(values["pot_tilt_guard_deg"]),
        pot_fill_required_steps=int(values["pot_fill_required_steps"]),
        pot_filled_threshold=float(values["pot_filled_threshold"]),
        saucepan_translation_guard_m=float(values["saucepan_translation_guard_m"]),
        saucepan_tilt_guard_deg=float(values["saucepan_tilt_guard_deg"]),
        saucepan_unstable_dwell_steps=int(values["saucepan_unstable_dwell_steps"]),
    )
    if not 0 <= result.cabinet_closed_threshold < result.cabinet_open_threshold <= 1:
        raise ValueError("TM04 cabinet thresholds are invalid")
    if not 0 < result.button_released_threshold_m < result.button_pressed_threshold_m:
        raise ValueError("TM04 button hysteresis is invalid")
    if (
        result.button_release_dwell_steps <= 0
        or result.pot_fill_required_steps <= 0
        or result.saucepan_unstable_dwell_steps <= 0
    ):
        raise ValueError("TM04 dwell and fill step counts must be positive")
    if not 0 < result.pot_filled_threshold <= 1:
        raise ValueError("TM04 pot_filled_threshold must be in (0, 1]")
    if any(
        value <= 0
        for value in (
            result.can_place_xy_tolerance_m,
            result.pot_water_xy_threshold_m,
            result.pot_translation_guard_m,
            result.pot_tilt_guard_deg,
            result.saucepan_translation_guard_m,
            result.saucepan_tilt_guard_deg,
        )
    ):
        raise ValueError("TM04 physical tolerances must be positive")
    return result


def resolve_action_config(config: TaskConfig) -> ActionConfig:
    operations = config.operations
    return ActionConfig(
        water_button_metadata=dict(_mapping(operations, "water_button")),
        cabinet_open_metadata=dict(_mapping(operations, "cabinet_open")),
        can_pick_metadata=dict(_mapping(operations, "can_pick")),
        can_place_metadata=dict(_mapping(operations, "can_place")),
        cabinet_close_metadata=dict(_mapping(operations, "cabinet_close")),
        sugar_pick_metadata=dict(_mapping(operations, "sugar_pick")),
        sugar_delivery_metadata=dict(_mapping(operations, "sugar_delivery")),
        completion_button_metadata=dict(_mapping(operations, "completion_button")),
        return_arm_metadata=dict(_mapping(operations, "return_arm")),
    )


def resolve_run_context(
    *, split: str, env_split: str, variant: str, execute: bool,
    extra_metadata: Mapping[str, Any] | None = None,
) -> RunContext:
    return RunContext(
        split=str(split), env_split=str(env_split), variant=str(variant),
        execute=bool(execute), extra_metadata=dict(extra_metadata or {}),
    )


def resolve_task_options(task_options: Mapping[str, Any] | None) -> dict[str, Any]:
    if task_options:
        raise ValueError("TM04 route, K/N, and prompt are seed/YAML-owned")
    return {}


def missing_calibration_resources(config: TaskConfig) -> tuple[str, ...]:
    return tuple(
        name
        for name in CALIBRATION_RESOURCE_NAMES
        if (path := config.path(config.resources.get(name))) is None or not path.is_file()
    )


def require_production_calibration(config: TaskConfig) -> None:
    missing = missing_calibration_resources(config)
    scene = config.scene
    if scene.get("sugar_plate_world_xyz") is None:
        missing += ("sugar_plate_world_xyz",)
    if scene.get("hot_chocolate_world_xyz") is None:
        missing += ("hot_chocolate_world_xyz",)
    if missing:
        raise TM04CalibrationError(
            "TM04 physical calibration is incomplete: " + ", ".join(missing)
        )


def _mapping(payload: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = payload.get(name, {})
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"TM04 {name} must be a mapping")
    return value


def _reject_unknown(payload: Mapping[str, Any], allowed: set[str], label: str) -> None:
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise ValueError(f"unknown TM04 {label}: " + ", ".join(unknown))


__all__ = [
    "CALIBRATION_RESOURCE_NAMES", "DEFAULT_TASK_CONFIG_FILE", "TM04CalibrationError",
    "load_task_config", "missing_calibration_resources", "require_production_calibration",
    "resolve_action_config", "resolve_observer_config", "resolve_run_context", "resolve_task_options",
]
