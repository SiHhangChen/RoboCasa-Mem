"""Build the shared manipulation controller for WX-02."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from bench.execution.manipulation import (
    ManipulationController,
    ManipulationRuntimeConfig,
    get_object_hold_status,
)
from bench.execution.planning import MplibPlannerConfig, PickPlaceConfig, pose_from_matrix
from bench.execution.planning.mplib_planner import run_mplib_planner_subprocess


@dataclass(frozen=True)
class ManipulationAssembly:
    """Controller plus the effective planner configuration."""

    controller: ManipulationController
    planner_settings: Mapping[str, Any]


def build_manipulation_assembly(
    env: Any,
    obs: dict[str, Any],
    *,
    settings: Mapping[str, Any],
    planner_script: Path,
    execute: bool,
) -> ManipulationAssembly:
    """Create a strictly configured shared manipulation controller."""

    _reject_unknown(
        settings,
        {"pick_place", "planner", "runtime", "static_scene"},
        "manipulation sections",
    )
    planner = _section(settings, "planner")
    controller = ManipulationController(
        env,
        obs,
        planner=_make_subprocess_planner(
            script_path=planner_script,
            conda_env=str(planner.get("conda_env", "membench")),
            timeout_s=_optional_float(planner.get("subprocess_timeout_s", 60.0)),
        ),
        pick_place_config=_build_pick_place_config(_section(settings, "pick_place")),
        runtime_config=_build_runtime_config(
            execute=execute,
            runtime=_section(settings, "runtime"),
            planner=planner,
            static_scene=_section(settings, "static_scene"),
        ),
    )
    return ManipulationAssembly(controller=controller, planner_settings=planner)


def adopt_verified_held_object(
    controller: ManipulationController,
    object_name: str,
    *,
    max_end_effector_distance: float,
) -> dict[str, Any]:
    """Seed a fresh task controller from a physically verified setup grasp."""

    controller.refresh_obs()
    hold = get_object_hold_status(
        controller.env,
        object_name,
        max_end_effector_distance=float(max_end_effector_distance),
    )
    if not bool(hold.get("held", False)):
        raise RuntimeError(f"cannot adopt unheld WX-02 object {object_name!r}: {hold}")
    end_effector_world = controller.stage_executor.current_eef_site_world()
    controller._set_virtual_held_object(
        object_name,
        pose_from_matrix(end_effector_world),
    )
    transform = controller._virtual_object_from_eef
    if controller._virtual_held_object != str(object_name) or transform is None:
        raise RuntimeError(f"failed to seed WX-02 held state for {object_name!r}")
    return {
        "object_name": str(object_name),
        "physical_hold": hold,
        "controller_held_object": controller._virtual_held_object,
        "object_from_end_effector": np.asarray(transform, dtype=float).tolist(),
        "pass": True,
    }


def _build_pick_place_config(settings: Mapping[str, Any]) -> PickPlaceConfig:
    values = dict(settings)
    config = replace(
        PickPlaceConfig(),
        orientation_mode=str(values.pop("orientation_mode", "object-y-topdown")),
        mug_orientation_mode=str(
            values.pop("mug_orientation_mode", "object-y-topdown")
        ),
        pregrasp_height=float(values.pop("pregrasp_height", 0.20)),
        grasp_clearance=float(values.pop("grasp_clearance", -0.05)),
        lift_height=float(values.pop("lift_height", 0.12)),
        mug_lift_height=_optional_float(values.pop("mug_lift_height", 0.08)),
        grasp_yaw_offset_deg=float(values.pop("grasp_yaw_offset_deg", 0.0)),
        place_height=float(values.pop("place_height", 0.16)),
        retreat_height=float(values.pop("retreat_height", 0.30)),
        retreat_x=float(values.pop("retreat_x", -0.10)),
        retreat_y=float(values.pop("retreat_y", -0.18)),
        release_offset_x=float(values.pop("release_offset_x", 0.0)),
        release_offset_y=float(values.pop("release_offset_y", 0.0)),
        post_release_clearance_distance=float(
            values.pop("post_release_clearance_distance", 0.0)
        ),
        post_release_clearance_z_offset=float(
            values.pop("post_release_clearance_z_offset", 0.0)
        ),
    )
    if values:
        raise ValueError(
            "unknown WX-02 pick_place settings: " + ", ".join(sorted(values))
        )
    return config


def _build_runtime_config(
    *,
    execute: bool,
    runtime: Mapping[str, Any],
    planner: Mapping[str, Any],
    static_scene: Mapping[str, Any],
) -> ManipulationRuntimeConfig:
    planner_values = dict(planner)
    conda_env = planner_values.pop("conda_env", "membench")
    timeout = planner_values.pop("subprocess_timeout_s", 60.0)
    del conda_env, timeout
    planner_config = MplibPlannerConfig(
        planning_time=float(planner_values.pop("planning_time", 5.0)),
        time_step=float(planner_values.pop("time_step", 0.1)),
        qpos_step=float(planner_values.pop("qpos_step", 0.02)),
        rrt_range=float(planner_values.pop("rrt_range", 0.1)),
    )
    if planner_values:
        raise ValueError(
            "unknown WX-02 planner settings: " + ", ".join(sorted(planner_values))
        )
    scene = dict(static_scene)
    scene_options = {
        "enabled": bool(scene.pop("enabled", False)),
        "validate_collisions": bool(scene.pop("validate_collisions", False)),
        "fixture_keywords": str(
            scene.pop("fixture_keywords", "counter,dining,sink,cabinet")
        ),
        "exclude_keywords": str(scene.pop("exclude_keywords", "floor,wall")),
        "spacing": float(scene.pop("spacing", 0.04)),
        "inflate": float(scene.pop("inflate", 0.05)),
        "point_radius": float(scene.pop("point_radius", 0.01)),
        "faces": str(scene.pop("faces", "bottom")),
        "point_cloud_stages": str(scene.pop("point_cloud_stages", "lift,retreat")),
        "min_z": float(scene.pop("min_z", 1.0)),
        "max_points": int(scene.pop("max_points", 4000)),
    }
    if scene:
        raise ValueError(
            "unknown WX-02 static_scene settings: " + ", ".join(sorted(scene))
        )
    values = dict(runtime)
    config = ManipulationRuntimeConfig(
        execute=bool(execute),
        max_joint_step=float(values.pop("max_joint_step", 0.02)),
        settle_steps=int(values.pop("settle_steps", 10)),
        position_threshold=float(values.pop("position_threshold", 0.06)),
        orientation_threshold_deg=float(values.pop("orientation_threshold_deg", 10.0)),
        qpos_threshold=float(values.pop("qpos_threshold", 0.08)),
        open_steps=int(values.pop("open_steps", 10)),
        close_steps=int(values.pop("close_steps", 50)),
        release_steps=int(values.pop("release_steps", 20)),
        release_settle_steps=int(values.pop("release_settle_steps", 10)),
        planner_args=planner_config.planner_args(),
        static_scene_options=scene_options,
    )
    if values:
        raise ValueError("unknown WX-02 runtime settings: " + ", ".join(sorted(values)))
    return config


def _make_subprocess_planner(
    *, script_path: Path, conda_env: str, timeout_s: float | None
):
    def plan(payload: dict[str, Any]) -> dict[str, Any]:
        code, result = run_mplib_planner_subprocess(
            script_path=script_path,
            plan_request=payload,
            conda_env=conda_env,
            timeout_s=timeout_s,
        )
        if result is None:
            return {
                "subprocess_returncode": int(code),
                "plan": {"status": "planner_subprocess_no_json", "attempts": []},
            }
        result["subprocess_returncode"] = int(code)
        return result

    return plan


def _section(payload: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = payload.get(name, {})
    if not isinstance(value, Mapping):
        raise ValueError(f"WX-02 manipulation.{name} must be a mapping")
    return value


def _reject_unknown(payload: Mapping[str, Any], allowed: set[str], label: str) -> None:
    unknown = set(payload) - allowed
    if unknown:
        raise ValueError(f"unknown WX-02 {label}: " + ", ".join(sorted(unknown)))


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)


__all__ = [
    "ManipulationAssembly",
    "adopt_verified_held_object",
    "build_manipulation_assembly",
]
