"""TM03 manipulation assembly for stove, drawers, and toaster oven."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from bench.tasks.tm03.execution.appliance_manipulation import (
    ManipulationAssembly,
    build_manipulation_assembly as build_appliance_manipulation_assembly,
)
from bench.tasks.tm03.execution.toaster_oven import (
    TURN_OFF_OPERATION,
    TURN_ON_OPERATION,
    load_toaster_oven_interaction_calibration,
    turn_native_toaster_oven_time_knob,
)


def build_manipulation_assembly(
    env: Any,
    obs: dict[str, Any],
    *,
    settings: Mapping[str, Any],
    planner_script: Path,
    interaction_poses_file: Path,
    toaster_oven_interaction_poses_file: Path,
    execute: bool,
    missing_resources: tuple[str, ...],
) -> ManipulationAssembly:
    assembly = build_appliance_manipulation_assembly(
        env,
        obs,
        settings=settings,
        planner_script=planner_script,
        interaction_poses_file=interaction_poses_file,
        execute=execute,
        missing_resources=missing_resources,
    )
    controller = assembly.controller
    calibration = load_toaster_oven_interaction_calibration(
        toaster_oven_interaction_poses_file
    )
    original_completion = controller.operation_handlers["press_completion_button"]

    def press_completion(target: str, **kwargs: Any):
        state = env.semantic_state
        checks = {
            "c1_complete": bool(state.c1_complete),
            "c2_complete": bool(state.c2_complete),
            "toaster_oven_off": not bool(env.toaster_oven_on()),
            "toaster_oven_door_closed": bool(env.toaster_oven_door_closed()),
            "toaster_oven_food_inside": bool(env.toaster_oven_food_inside()),
        }
        if not all(checks.values()):
            return {
                "primitive": "physical_tm03_completion_button_press",
                "button": str(target),
                "preconditions": checks,
                "semantic_shortcut_used": False,
                "pass": False,
                "reason": "tm03_completion_button_preconditions_not_met",
            }
        return original_completion(target, **kwargs)

    controller.operation_handlers[TURN_ON_OPERATION] = (
        lambda target, **kwargs: turn_native_toaster_oven_time_knob(
            controller,
            str(target),
            turn_on=True,
            calibration=calibration,
            **kwargs,
        )
    )
    controller.operation_handlers[TURN_OFF_OPERATION] = (
        lambda target, **kwargs: turn_native_toaster_oven_time_knob(
            controller,
            str(target),
            turn_on=False,
            calibration=calibration,
            **kwargs,
        )
    )
    controller.operation_handlers["press_completion_button"] = press_completion
    return assembly


__all__ = ["build_manipulation_assembly"]
