"""Build the TS01 manipulation controller from YAML-owned settings."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from bench.execution.manipulation import ManipulationController, ManipulationRuntimeConfig
from bench.execution.planning import MplibPlannerConfig, PickPlaceConfig, pose_from_matrix
from bench.execution.planning.mplib_planner import run_mplib_planner_subprocess
from bench.tasks.ts03.execution.geometry import build_pick_payload_selector


@dataclass(frozen=True)
class ManipulationAssembly:
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
    planner = _section(settings, "planner")
    controller = ManipulationController(
        env,
        obs,
        planner=_make_planner(
            planner_script,
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
            descriptor.actions.grasp_yaw_offsets_deg,
            grasp_finger_offset_x=getattr(descriptor.actions, "grasp_finger_offset_x", 0.0),
            grasp_finger_offset_y=getattr(descriptor.actions, "grasp_finger_offset_y", 0.0),
        ),
    )
    _install_held_sugar_follow_assist(env, controller)
    return ManipulationAssembly(controller, planner)


def _pick_place_config(values: Mapping[str, Any]) -> PickPlaceConfig:
    settings = dict(values)
    defaults = PickPlaceConfig()
    allowed = set(defaults.__dataclass_fields__)
    unknown = sorted(set(settings) - allowed)
    if unknown:
        raise ValueError("unknown TS01 pick_place settings: " + ", ".join(unknown))
    return replace(defaults, **settings)


def _runtime_config(
    *,
    execute: bool,
    runtime: Mapping[str, Any],
    planner: Mapping[str, Any],
    static_scene: Mapping[str, Any],
) -> ManipulationRuntimeConfig:
    runtime_values = dict(runtime)
    planner_config = MplibPlannerConfig(
        planning_time=float(planner.get("planning_time", 5.0)),
        time_step=float(planner.get("time_step", 0.1)),
        qpos_step=float(planner.get("qpos_step", 0.02)),
        rrt_range=float(planner.get("rrt_range", 0.1)),
    )
    planner_args = planner_config.planner_args()
    preserve_raw = planner.get("preserve_ik_branch_stages", ("place",))
    preserve_stages = (
        [item.strip() for item in str(preserve_raw).replace(";", ",").split(",")]
        if isinstance(preserve_raw, str)
        else [str(item).strip() for item in preserve_raw]
    )
    preserve_stages = [item for item in preserve_stages if item]
    nearest_ik_samples = int(planner.get("nearest_ik_samples", 64))
    nearest_ik_threshold = float(planner.get("nearest_ik_threshold", 0.01))
    nearest_ik_max_joint_delta = float(
        planner.get("nearest_ik_max_joint_delta", 2.0)
    )
    if nearest_ik_samples < 1:
        raise ValueError("TS01 nearest_ik_samples must be positive")
    if nearest_ik_threshold <= 0.0 or nearest_ik_max_joint_delta <= 0.0:
        raise ValueError("TS01 nearest-IK thresholds must be positive")
    planner_args.update(
        {
            "preserve_ik_branch_stages": preserve_stages,
            "nearest_ik_samples": nearest_ik_samples,
            "nearest_ik_threshold": nearest_ik_threshold,
            "nearest_ik_max_joint_delta": nearest_ik_max_joint_delta,
        }
    )
    return ManipulationRuntimeConfig(
        execute=bool(execute),
        max_joint_step=float(runtime_values.get("max_joint_step", 0.025)),
        settle_steps=int(runtime_values.get("settle_steps", 80)),
        position_threshold=float(runtime_values.get("position_threshold", 0.04)),
        orientation_threshold_deg=float(runtime_values.get("orientation_threshold_deg", 10.0)),
        qpos_threshold=float(runtime_values.get("qpos_threshold", 0.50)),
        open_steps=int(runtime_values.get("open_steps", 20)),
        close_steps=int(runtime_values.get("close_steps", 90)),
        release_steps=int(runtime_values.get("release_steps", 40)),
        release_settle_steps=int(runtime_values.get("release_settle_steps", 20)),
        planner_args=planner_args,
        static_scene_options=dict(static_scene),
    )


def _make_planner(script: Path, *, conda_env: str, timeout_s: float | None):
    def plan(payload: dict[str, Any]) -> dict[str, Any]:
        code, result = run_mplib_planner_subprocess(
            script_path=script,
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


def _install_held_sugar_follow_assist(env: Any, controller: ManipulationController) -> None:
    original_step = env.step

    def step_with_follow(action: Any):
        obs, reward, done, info = original_step(action)
        held = controller._virtual_held_object
        transform = controller._virtual_object_from_eef
        values = np.asarray(action, dtype=float).reshape(-1)
        if held is None or transform is None or not (values.size and values[-1] > 0.0):
            return obs, reward, done, info
        if str(held) not in set(getattr(env, "sugar_names", ())):
            return obs, reward, done, info
        pose_world = controller.stage_executor.current_eef_site_world() @ transform
        obj = env.objects[str(held)]
        qpos = np.asarray(env.sim.data.get_joint_qpos(obj.joints[0]), dtype=float).reshape(-1)
        if qpos.size < 7:
            return obs, reward, done, info
        pose = pose_from_matrix(pose_world)
        quat_xyzw = np.asarray(pose["q_xyzw"], dtype=float)
        qpos[:3] = np.asarray(pose["p"], dtype=float)
        qpos[3:7] = quat_xyzw[[3, 0, 1, 2]]
        env.sim.data.set_joint_qpos(obj.joints[0], qpos)
        env.sim.forward()
        return env._get_observations(force_update=True), reward, done, info

    env.step = step_with_follow


def _section(payload: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = payload.get(name, {})
    if not isinstance(value, Mapping):
        raise ValueError(f"TS01 manipulation.{name} must be a mapping")
    return value


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)


__all__ = ["ManipulationAssembly", "build_manipulation_assembly"]
