"""Construction of the generic arm controller used by WR-01."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from bench.execution.manipulation import ManipulationController, ManipulationRuntimeConfig
from bench.execution.planning import MplibPlannerConfig, PickPlaceConfig
from bench.execution.planning.mplib_planner import run_mplib_planner_subprocess
from bench.tasks.wr01.execution.storage import install_held_item_follow_assist


@dataclass(frozen=True)
class ManipulationAssembly:
    controller: ManipulationController
    planner_settings: Mapping[str, Any]
    gripper_settings: Mapping[str, Any]


def build_manipulation_assembly(
    env: Any,
    obs: dict[str, Any],
    *,
    settings: Mapping[str, Any],
    planner_script: Path,
    execute: bool,
) -> ManipulationAssembly:
    planner_settings = _section(settings, "planner")
    gripper_settings = _section(settings, "gripper")
    controller = ManipulationController(
        env,
        obs,
        planner=_make_planner(
            script_path=planner_script,
            conda_env=str(planner_settings.get("conda_env", "membench")),
            timeout_s=float(planner_settings.get("subprocess_timeout_s", 60.0)),
        ),
        pick_place_config=_pick_place_config(_section(settings, "pick_place")),
        runtime_config=_runtime_config(
            execute=execute,
            runtime=_section(settings, "runtime"),
            planner=planner_settings,
            static_scene=_section(settings, "static_scene"),
        ),
    )
    force_limit = float(gripper_settings.get("force_limit", 10.0))
    if set(gripper_settings) - {"force_limit"}:
        unknown = sorted(set(gripper_settings) - {"force_limit"})
        raise ValueError("unknown WR-01 gripper settings: " + ", ".join(unknown))
    _configure_right_gripper_force_limit(env, force_limit)
    install_held_item_follow_assist(env, controller)
    return ManipulationAssembly(
        controller=controller,
        planner_settings=planner_settings,
        gripper_settings={"force_limit": force_limit},
    )


def _configure_right_gripper_force_limit(env: Any, force_limit: float) -> None:
    """Apply the validated per-finger force cap without changing shared robot assets."""

    if not np.isfinite(force_limit) or force_limit <= 0.0:
        raise ValueError("WR-01 gripper.force_limit must be positive and finite")
    gripper = env.robots[0].gripper["right"]
    for actuator_name in list(gripper.actuators):
        actuator_id = int(env.sim.model.actuator_name2id(actuator_name))
        env.sim.model.actuator_forcelimited[actuator_id] = 1
        env.sim.model.actuator_forcerange[actuator_id] = [
            -float(force_limit),
            float(force_limit),
        ]


def _pick_place_config(values: Mapping[str, Any]) -> PickPlaceConfig:
    remaining = dict(values)
    config = replace(
        PickPlaceConfig(),
        orientation_mode=str(remaining.pop("orientation_mode", "object-y-topdown")),
        mug_orientation_mode=str(remaining.pop("mug_orientation_mode", "mug-body-side")),
        pregrasp_height=float(remaining.pop("pregrasp_height", 0.15)),
        grasp_clearance=float(remaining.pop("grasp_clearance", -0.04)),
        lift_height=float(remaining.pop("lift_height", 0.12)),
        place_height=float(remaining.pop("place_height", 0.10)),
        retreat_height=float(remaining.pop("retreat_height", 0.18)),
        retreat_x=float(remaining.pop("retreat_x", 0.0)),
        retreat_y=float(remaining.pop("retreat_y", -0.10)),
    )
    if remaining:
        raise ValueError("unknown WR-01 pick_place settings: " + ", ".join(sorted(remaining)))
    return config


def _runtime_config(
    *,
    execute: bool,
    runtime: Mapping[str, Any],
    planner: Mapping[str, Any],
    static_scene: Mapping[str, Any],
) -> ManipulationRuntimeConfig:
    values = dict(runtime)
    planner_config = MplibPlannerConfig(
        planning_time=float(planner.get("planning_time", 5.0)),
        time_step=float(planner.get("time_step", 0.1)),
        qpos_step=float(planner.get("qpos_step", 0.02)),
        rrt_range=float(planner.get("rrt_range", 0.1)),
    )
    scene = dict(static_scene)
    options = {
        "enabled": bool(scene.pop("enabled", True)),
        "validate_collisions": bool(scene.pop("validate_collisions", False)),
        "fixture_keywords": str(scene.pop("fixture_keywords", "cabinet,counter,stove")),
        "exclude_keywords": str(scene.pop("exclude_keywords", "floor,wall")),
        "spacing": float(scene.pop("spacing", 0.04)),
        "inflate": float(scene.pop("inflate", 0.05)),
        "point_radius": float(scene.pop("point_radius", 0.01)),
        "faces": str(scene.pop("faces", "bottom")),
        "point_cloud_stages": str(
            scene.pop(
                "point_cloud_stages",
                "pregrasp,grasp,lift,place,clearance,retreat",
            )
        ),
        "min_z": float(scene.pop("min_z", 1.0)),
        "max_points": int(scene.pop("max_points", 4000)),
    }
    if scene:
        raise ValueError("unknown WR-01 static_scene settings: " + ", ".join(sorted(scene)))
    config = ManipulationRuntimeConfig(
        execute=bool(execute),
        max_joint_step=float(values.pop("max_joint_step", 0.025)),
        settle_steps=int(values.pop("settle_steps", 10)),
        position_threshold=float(values.pop("position_threshold", 0.12)),
        orientation_threshold_deg=float(values.pop("orientation_threshold_deg", 15.0)),
        qpos_threshold=float(values.pop("qpos_threshold", 0.16)),
        open_steps=int(values.pop("open_steps", 10)),
        close_steps=int(values.pop("close_steps", 50)),
        release_steps=int(values.pop("release_steps", 20)),
        release_settle_steps=int(values.pop("release_settle_steps", 5)),
        planner_args=planner_config.planner_args(),
        static_scene_options=options,
    )
    if values:
        raise ValueError("unknown WR-01 runtime settings: " + ", ".join(sorted(values)))
    return config


def _make_planner(*, script_path: Path, conda_env: str, timeout_s: float):
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
        raise ValueError(f"WR-01 manipulation.{name} must be a mapping")
    return value


__all__ = ["ManipulationAssembly", "build_manipulation_assembly"]
