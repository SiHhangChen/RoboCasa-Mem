"""Calibrated physical timer-knob primitive for the TM03 toaster oven."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from bench.tasks.tm03.execution.appliance_manipulation import (
    TM03ManipulationController,
    _knob_rotation_pose,
    _resolve_anchor_pose_world,
    _without_obs,
    _world_target_in_current_base,
)


TURN_ON_OPERATION = "turn_native_toaster_oven_time_knob_on"
TURN_OFF_OPERATION = "turn_native_toaster_oven_time_knob_off"


@dataclass(frozen=True)
class ToasterOvenInteractionCalibration:
    source: Path
    expected_asset: str
    target_knob: str
    pose_files: Mapping[str, Path]
    turn_on_waypoints_deg: tuple[float, ...]
    turn_off_waypoints_deg: tuple[float, ...]
    open_steps: int
    close_steps: int
    release_steps: int
    off_qpos_max_rad: float
    on_qpos_min_rad: float


def load_toaster_oven_interaction_calibration(
    path: str | Path,
) -> ToasterOvenInteractionCalibration | None:
    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if (
        payload.get("config_kind")
        != "tm03_native_toaster_oven_time_knob_interaction"
    ):
        raise ValueError(f"invalid TM03 toaster-oven calibration: {source}")
    if payload.get("calibrated") is not True:
        return None
    values = dict(payload.get("pose_files") or {})
    required = {"pinch_off", "pinch_on", "pinch_on_for_off"}
    if set(values) != required:
        raise ValueError(
            "TM03 toaster-oven calibration requires exactly "
            + ", ".join(sorted(required))
        )
    files = {
        role: (source.parent / str(value)).resolve()
        for role, value in values.items()
    }
    if not all(path.is_file() for path in files.values()):
        raise FileNotFoundError("TM03 toaster-oven calibration pose is missing")
    waypoints = dict(payload.get("rotation_waypoints_deg") or {})
    turn_on = _finite_tuple(waypoints.get("turn_on"), "turn_on")
    turn_off = _finite_tuple(waypoints.get("turn_off"), "turn_off")
    if not turn_on or any(b <= a for a, b in zip(turn_on, turn_on[1:])):
        raise ValueError("TM03 toaster-oven turn_on waypoints must increase")
    if not turn_off or any(b >= a for a, b in zip(turn_off, turn_off[1:])):
        raise ValueError("TM03 toaster-oven turn_off waypoints must decrease")
    gripper = dict(payload.get("gripper") or {})
    steps = {
        key: int(gripper[key])
        for key in ("open_steps", "close_steps", "release_steps")
    }
    if any(value <= 0 for value in steps.values()):
        raise ValueError("TM03 toaster-oven gripper steps must be positive")
    validation = dict(payload.get("validation") or {})
    off_max = float(validation["off_qpos_max_rad"])
    on_min = float(validation["on_qpos_min_rad"])
    if not 0.0 <= off_max < on_min:
        raise ValueError("TM03 toaster-oven qpos thresholds are invalid")
    return ToasterOvenInteractionCalibration(
        source=source.resolve(),
        expected_asset=str(payload.get("expected_asset", "ToasterOven059")),
        target_knob=str(payload.get("target_knob", "time")),
        pose_files=files,
        turn_on_waypoints_deg=turn_on,
        turn_off_waypoints_deg=turn_off,
        open_steps=steps["open_steps"],
        close_steps=steps["close_steps"],
        release_steps=steps["release_steps"],
        off_qpos_max_rad=off_max,
        on_qpos_min_rad=on_min,
    )


def turn_native_toaster_oven_time_knob(
    controller: TM03ManipulationController,
    target: str,
    *,
    turn_on: bool,
    calibration: ToasterOvenInteractionCalibration | None,
    **_: Any,
) -> dict[str, Any]:
    operation = TURN_ON_OPERATION if turn_on else TURN_OFF_OPERATION
    if calibration is None:
        return _failure(operation, target, "tm03_toaster_oven_calibration_required")
    if str(target) != calibration.target_knob:
        return _failure(operation, target, "tm03_toaster_oven_knob_mismatch")
    if calibration.expected_asset != controller.env.expected_toaster_oven_asset:
        return _failure(operation, target, "tm03_toaster_oven_asset_mismatch")

    controller.refresh_obs()
    state = controller.env.semantic_state
    initial_qpos = float(controller.env.toaster_oven_time_qpos())
    preconditions = {
        "no_absorbing_failure": not state.absorbing_failure,
        "expected_semantic_state": (
            not state.c1_complete
            if turn_on
            else state.c1_complete and not state.c2_complete
        ),
        "expected_physical_state": (
            initial_qpos <= calibration.off_qpos_max_rad
            if turn_on
            else initial_qpos >= calibration.on_qpos_min_rad
        ),
        "door_closed": controller.env.toaster_oven_door_closed(),
        "food_inside": controller.env.toaster_oven_food_inside(),
    }
    if controller.runtime_config.execute and not all(preconditions.values()):
        return {
            **_failure(operation, target, "tm03_toaster_oven_preconditions_not_met"),
            "preconditions": preconditions,
            "initial_qpos_rad": initial_qpos,
        }

    pinch_off_world, pinch_off_info = _resolve_anchor_pose_world(
        controller.env, calibration.pose_files["pinch_off"]
    )
    approach_role = "pinch_off" if turn_on else "pinch_on_for_off"
    approach_world, approach_info = _resolve_anchor_pose_world(
        controller.env, calibration.pose_files[approach_role]
    )
    endpoint_role = "pinch_on" if turn_on else "pinch_off"
    endpoint_world, endpoint_info = _resolve_anchor_pose_world(
        controller.env, calibration.pose_files[endpoint_role]
    )
    waypoint_angles = (
        calibration.turn_on_waypoints_deg
        if turn_on
        else calibration.turn_off_waypoints_deg
    )

    commands: list[dict[str, Any]] = []
    stages: list[dict[str, Any]] = []
    commands.append(
        {
            "command": "open",
            **_without_obs(
                controller.run_gripper_command(-1.0, calibration.open_steps)
            ),
        }
    )
    approach = controller.run_planned_stage(
        _world_target_in_current_base(controller, approach_world),
        stage=f"{operation}_approach",
        gripper_action=-1.0,
    )
    stages.append(approach)
    if not bool(approach.get("pass", False)):
        return _motion_failure(
            operation,
            target,
            "tm03_toaster_oven_approach_failed",
            preconditions,
            commands,
            stages,
            approach_info,
            endpoint_info,
        )

    commands.append(
        {
            "command": "close",
            **_without_obs(
                controller.run_gripper_command(1.0, calibration.close_steps)
            ),
        }
    )
    for index, angle in enumerate(waypoint_angles):
        target_world = _knob_rotation_pose(pinch_off_world, angle_deg=float(angle))
        if index == len(waypoint_angles) - 1:
            target_world = endpoint_world
        stage = controller.run_planned_stage(
            _world_target_in_current_base(controller, target_world),
            stage=f"{operation}_rotate_{float(angle):g}deg",
            gripper_action=1.0,
        )
        stages.append(stage)
        if not bool(stage.get("pass", False)):
            return _motion_failure(
                operation,
                target,
                "tm03_toaster_oven_rotation_failed",
                preconditions,
                commands,
                stages,
                approach_info,
                endpoint_info,
            )

    commands.append(
        {
            "command": "release",
            **_without_obs(
                controller.run_gripper_command(-1.0, calibration.release_steps)
            ),
        }
    )
    controller.refresh_obs()
    final_qpos = float(controller.env.toaster_oven_time_qpos())
    diagnostics = controller.env.task_diagnostics()
    event_key = "c1_complete" if turn_on else "c2_complete"
    checks = {
        "all_motion_stages_passed": all(
            bool(stage.get("pass", False)) for stage in stages
        ),
        "target_qpos_reached": (
            final_qpos >= calibration.on_qpos_min_rad
            if turn_on
            else final_qpos <= calibration.off_qpos_max_rad
        ),
        "physical_state_reached": (
            controller.env.toaster_oven_on()
            if turn_on
            else not controller.env.toaster_oven_on()
        ),
        "semantic_event_committed": bool(diagnostics.get(event_key, False)),
        "door_closed": controller.env.toaster_oven_door_closed(),
        "food_inside": controller.env.toaster_oven_food_inside(),
        "no_absorbing_failure": not bool(diagnostics.get("absorbing_failure", False)),
    }
    passed = all(checks.values())
    return {
        "primitive": "pinch_and_rotate_native_toaster_oven_time_knob",
        "operation": operation,
        "target": str(target),
        "calibration_file": str(calibration.source),
        "preconditions": preconditions,
        "initial_qpos_rad": initial_qpos,
        "final_qpos_rad": final_qpos,
        "pinch_off_pose": pinch_off_info,
        "approach_pose": approach_info,
        "endpoint_pose": endpoint_info,
        "commands": commands,
        "stages": stages,
        "task_diagnostics": diagnostics,
        "checks": checks,
        "semantic_shortcut_used": False,
        "pass": passed,
        "reason": None if passed else "tm03_toaster_oven_time_knob_failed",
    }


def _motion_failure(
    operation: str,
    target: str,
    reason: str,
    preconditions: Mapping[str, bool],
    commands: list[dict[str, Any]],
    stages: list[dict[str, Any]],
    approach_info: Mapping[str, Any],
    endpoint_info: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        **_failure(operation, target, reason),
        "preconditions": dict(preconditions),
        "commands": commands,
        "stages": stages,
        "approach_pose": dict(approach_info),
        "endpoint_pose": dict(endpoint_info),
    }


def _failure(operation: str, target: str, reason: str) -> dict[str, Any]:
    return {
        "primitive": "pinch_and_rotate_native_toaster_oven_time_knob",
        "operation": operation,
        "target": str(target),
        "semantic_shortcut_used": False,
        "pass": False,
        "reason": str(reason),
    }


def _finite_tuple(value: Any, label: str) -> tuple[float, ...]:
    try:
        result = tuple(float(item) for item in value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"TM03 {label} waypoints must be a sequence") from exc
    if not all(np.isfinite(item) for item in result):
        raise ValueError(f"TM03 {label} waypoints must be finite")
    return result


__all__ = [
    "TURN_OFF_OPERATION",
    "TURN_ON_OPERATION",
    "ToasterOvenInteractionCalibration",
    "load_toaster_oven_interaction_calibration",
    "turn_native_toaster_oven_time_knob",
]
