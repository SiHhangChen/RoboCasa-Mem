"""构建 WA-01 实际使用的机械臂控制器。"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from bench.execution.manipulation import ManipulationController, ManipulationRuntimeConfig
from bench.execution.planning import MplibPlannerConfig, PickPlaceConfig, pose_from_matrix
from bench.execution.planning.mplib_planner import run_mplib_planner_subprocess


STATIC_SCENE_DEFAULT_STAGES = "pregrasp,grasp,lift,place,clearance,retreat"


@dataclass(frozen=True)
class ManipulationAssembly:
    """汇总已装配机械臂控制器和有效规划参数。"""

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
    """创建规划器、抓放配置、运行时配置和机械臂控制器。"""

    planner_settings = _section(settings, "planner")
    runtime = _build_runtime_config(
        execute=execute,
        runtime=_section(settings, "runtime"),
        planner=planner_settings,
        static_scene=_section(settings, "static_scene"),
    )
    controller = ManipulationController(
        env,
        obs,
        planner=_make_subprocess_planner(
            script_path=planner_script,
            conda_env=str(planner_settings.get("conda_env", "membench")),
            timeout_s=_optional_float(
                planner_settings.get("subprocess_timeout_s", 60.0)
            ),
        ),
        pick_place_config=_build_pick_place_config(
            _section(settings, "pick_place")
        ),
        runtime_config=runtime,
    )
    _install_held_mug_follow_assist(env, controller)
    return ManipulationAssembly(controller=controller, planner_settings=planner_settings)


def _build_pick_place_config(settings: Mapping[str, Any]) -> PickPlaceConfig:
    """将 YAML 抓放参数严格转换为通用 PickPlaceConfig。"""

    values = dict(settings)
    resolved = replace(
        PickPlaceConfig(),
        orientation_mode=str(values.pop("orientation_mode", "object-y-topdown")),
        mug_orientation_mode=str(values.pop("mug_orientation_mode", "mug-body-side")),
        pregrasp_height=float(values.pop("pregrasp_height", 0.20)),
        grasp_clearance=float(values.pop("grasp_clearance", -0.05)),
        lift_height=float(values.pop("lift_height", 0.12)),
        mug_lift_height=_optional_float(values.pop("mug_lift_height", 0.035)),
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
            "unknown WA-01 pick_place settings: " + ", ".join(sorted(values))
        )
    return resolved


def _build_runtime_config(
    *,
    execute: bool,
    runtime: Mapping[str, Any],
    planner: Mapping[str, Any],
    static_scene: Mapping[str, Any],
) -> ManipulationRuntimeConfig:
    """构建机械臂步进阈值、MPLib 参数和静态碰撞场景配置。"""

    runtime_values = dict(runtime)
    planner_config = MplibPlannerConfig(
        planning_time=float(planner.get("planning_time", 5.0)),
        time_step=float(planner.get("time_step", 0.1)),
        qpos_step=float(planner.get("qpos_step", 0.02)),
        rrt_range=float(planner.get("rrt_range", 0.1)),
    )
    scene = dict(static_scene)
    scene_options = {
        "enabled": bool(scene.pop("enabled", True)),
        "validate_collisions": bool(scene.pop("validate_collisions", False)),
        "fixture_keywords": str(scene.pop("fixture_keywords", "cabinet,coffee_machine")),
        "exclude_keywords": str(scene.pop("exclude_keywords", "floor,wall")),
        "spacing": float(scene.pop("spacing", 0.04)),
        "inflate": float(scene.pop("inflate", 0.05)),
        "point_radius": float(scene.pop("point_radius", 0.01)),
        "faces": str(scene.pop("faces", "bottom")),
        "point_cloud_stages": str(
            scene.pop("point_cloud_stages", STATIC_SCENE_DEFAULT_STAGES)
        ),
        "min_z": float(scene.pop("min_z", 1.0)),
        "max_points": int(scene.pop("max_points", 4000)),
    }
    if scene:
        raise ValueError("unknown WA-01 static_scene settings: " + ", ".join(sorted(scene)))
    config = ManipulationRuntimeConfig(
        execute=bool(execute),
        max_joint_step=float(runtime_values.pop("max_joint_step", 0.025)),
        settle_steps=int(runtime_values.pop("settle_steps", 10)),
        position_threshold=float(runtime_values.pop("position_threshold", 0.12)),
        orientation_threshold_deg=float(
            runtime_values.pop("orientation_threshold_deg", 15.0)
        ),
        qpos_threshold=float(runtime_values.pop("qpos_threshold", 0.16)),
        open_steps=int(runtime_values.pop("open_steps", 10)),
        close_steps=int(runtime_values.pop("close_steps", 50)),
        release_steps=int(runtime_values.pop("release_steps", 20)),
        release_settle_steps=int(runtime_values.pop("release_settle_steps", 5)),
        planner_args=planner_config.planner_args(),
        static_scene_options=scene_options,
    )
    if runtime_values:
        raise ValueError("unknown WA-01 runtime settings: " + ", ".join(sorted(runtime_values)))
    return config


def _make_subprocess_planner(
    *,
    script_path: Path,
    conda_env: str,
    timeout_s: float | None,
):
    """创建通过指定 conda 环境调用 MPLib 子进程的规划函数。"""

    def plan(payload: dict[str, Any]) -> dict[str, Any]:
        """执行一次规划请求并将子进程返回码写入结果。"""

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


def _install_held_mug_follow_assist(
    env: Any,
    controller: ManipulationController,
) -> None:
    """夹爪保持闭合时，使已确认抓住的杯子持续跟随末端位姿。"""
    original_step = env.step

    def step_with_held_mug_follow(action: Any):
        """包装 env.step，并在闭合动作后同步虚拟持杯位姿。"""

        obs, reward, done, info = original_step(action)
        held = controller._virtual_held_object
        transform = controller._virtual_object_from_eef
        values = np.asarray(action, dtype=float).reshape(-1)
        if held is None or transform is None or not (values.size and values[-1] > 0.0):
            return obs, reward, done, info
        pose_world = controller.stage_executor.current_eef_site_world() @ transform
        obj = env.objects[str(held)]
        qpos = np.asarray(
            env.sim.data.get_joint_qpos(obj.joints[0]),
            dtype=float,
        ).reshape(-1)
        if qpos.size < 7:
            return obs, reward, done, info
        pose = pose_from_matrix(pose_world)
        quat_xyzw = np.asarray(pose["q_xyzw"], dtype=float)
        updated = qpos.copy()
        updated[:3] = np.asarray(pose["p"], dtype=float)
        updated[3:7] = quat_xyzw[[3, 0, 1, 2]]
        env.sim.data.set_joint_qpos(obj.joints[0], updated)
        env.sim.forward()
        return env._get_observations(force_update=True), reward, done, info

    env.step = step_with_held_mug_follow


def _section(payload: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    """读取必须为 mapping 的机械臂配置子分区。"""

    value = payload.get(name, {})
    if not isinstance(value, Mapping):
        raise ValueError(f"WA-01 manipulation.{name} must be a mapping")
    return value


def _optional_float(value: Any) -> float | None:
    """将可选数值规范为 float 或 None。"""

    return None if value is None else float(value)


__all__ = ["ManipulationAssembly", "build_manipulation_assembly"]
