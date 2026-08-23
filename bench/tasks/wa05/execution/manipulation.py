"""Build the WA-05 manipulation controller from task-owned settings."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from bench.execution.manipulation import ManipulationController, ManipulationRuntimeConfig
from bench.execution.planning import MplibPlannerConfig, PickPlaceConfig
from bench.execution.planning.mplib_planner import run_mplib_planner_subprocess
from bench.tasks.wa05.execution.geometry import build_pick_payload_selector


STATIC_SCENE_DEFAULT_STAGES = "pregrasp,grasp,lift,place,clearance,retreat"


@dataclass(frozen=True)
class ManipulationAssembly:
    """Controller plus the planner settings recorded for diagnostics."""

    controller: ManipulationController
    planner_settings: Mapping[str, Any]


def build_manipulation_assembly(
    env: Any,
    obs: dict[str, Any],
    *,
    settings: Mapping[str, Any],
    planner_script: Path,
    cup_grasp_candidates_file: Path,
    execute: bool,
) -> ManipulationAssembly:
    """Create a fresh task-local controller; no older WA2 helpers are reused."""

    planner = _section(settings, "planner")
    controller = ManipulationController(
        env,
        obs,
        planner=_make_subprocess_planner(
            script_path=planner_script,
            conda_env=str(planner.get("conda_env", "membench")),
            timeout_s=_optional_float(planner.get("subprocess_timeout_s", 60.0)),
        ),
        pick_place_config=_pick_place_config(_section(settings, "pick_place")),
        runtime_config=_runtime_config(
            execute=execute,
            runtime=_section(settings, "runtime"),
            planner=planner,
            static_scene=_section(settings, "static_scene"),
        ),
        pick_payload_selector=build_pick_payload_selector(
            manual_candidates_file=cup_grasp_candidates_file,
        ),
    )
    return ManipulationAssembly(controller=controller, planner_settings=planner)


def _pick_place_config(values: Mapping[str, Any]) -> PickPlaceConfig:
    settings = dict(values)
    defaults = PickPlaceConfig()
    allowed = set(defaults.__dataclass_fields__)
    unknown = sorted(set(settings) - allowed)
    if unknown:
        raise ValueError("unknown WA-05 pick_place settings: " + ", ".join(unknown))
    return replace(defaults, **settings)


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
    scene_options = {
        "enabled": bool(scene.pop("enabled", False)),
        "validate_collisions": bool(scene.pop("validate_collisions", False)),
        "fixture_keywords": str(scene.pop("fixture_keywords", "sink,counter")),
        "exclude_keywords": str(scene.pop("exclude_keywords", "floor,wall")),
        "spacing": float(scene.pop("spacing", 0.04)),
        "inflate": float(scene.pop("inflate", 0.05)),
        "point_radius": float(scene.pop("point_radius", 0.01)),
        "faces": str(scene.pop("faces", "bottom")),
        "point_cloud_stages": str(
            scene.pop("point_cloud_stages", STATIC_SCENE_DEFAULT_STAGES)
        ),
        "min_z": float(scene.pop("min_z", 0.8)),
        "max_points": int(scene.pop("max_points", 4000)),
    }
    if scene:
        raise ValueError("unknown WA-05 static_scene settings: " + ", ".join(sorted(scene)))
    config = ManipulationRuntimeConfig(
        execute=bool(execute),
        max_joint_step=float(values.pop("max_joint_step", 0.025)),
        settle_steps=int(values.pop("settle_steps", 20)),
        position_threshold=float(values.pop("position_threshold", 0.12)),
        orientation_threshold_deg=float(values.pop("orientation_threshold_deg", 15.0)),
        qpos_threshold=float(values.pop("qpos_threshold", 0.24)),
        open_steps=int(values.pop("open_steps", 20)),
        close_steps=int(values.pop("close_steps", 90)),
        release_steps=int(values.pop("release_steps", 45)),
        release_settle_steps=int(values.pop("release_settle_steps", 20)),
        planner_args=planner_config.planner_args(),
        static_scene_options=scene_options,
    )
    if values:
        raise ValueError("unknown WA-05 runtime settings: " + ", ".join(sorted(values)))
    return config


def _make_subprocess_planner(
    *,
    script_path: Path,
    conda_env: str,
    timeout_s: float | None,
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
        raise ValueError(f"WA-05 manipulation.{name} must be a mapping")
    return value


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)


__all__ = ["ManipulationAssembly", "build_manipulation_assembly"]
