"""Build the WA-02 manipulation controller from task-owned settings."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from bench.execution.manipulation import ManipulationController, ManipulationRuntimeConfig
from bench.execution.planning import MplibPlannerConfig, PickPlaceConfig, pose_from_matrix
from bench.execution.planning.mplib_planner import run_mplib_planner_subprocess
from bench.tasks.wa02.domain.contracts import CUP_IDS
from bench.tasks.wa02.execution.geometry import build_pick_payload_selector


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
    _install_held_cup_follow_assist(env, controller)
    return ManipulationAssembly(controller=controller, planner_settings=planner)


def _pick_place_config(values: Mapping[str, Any]) -> PickPlaceConfig:
    settings = dict(values)
    defaults = PickPlaceConfig()
    allowed = set(defaults.__dataclass_fields__)
    unknown = sorted(set(settings) - allowed)
    if unknown:
        raise ValueError("unknown WA-02 pick_place settings: " + ", ".join(unknown))
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
        raise ValueError("unknown WA-02 static_scene settings: " + ", ".join(sorted(scene)))
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
        raise ValueError("unknown WA-02 runtime settings: " + ", ".join(sorted(values)))
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


def _install_held_cup_follow_assist(env: Any, controller: ManipulationController) -> None:
    """Keep a verified held WA-02 cup physically synchronized with the EEF.

    This is a MuJoCo transport assist, not a task-semantic transition: it only
    mirrors a controller-confirmed grasp and never records a fill, service, or
    final-selection event.
    """

    if bool(getattr(env, "_wa02_held_cup_follow_installed", False)):
        return
    original_step = env.step
    release_anchor_object: str | None = None
    release_anchor_qpos: np.ndarray | None = None
    release_separated_object: str | None = None

    def clear_release_anchor(*, clear_separated: bool = False) -> None:
        nonlocal release_anchor_object, release_anchor_qpos, release_separated_object
        release_anchor_object = None
        release_anchor_qpos = None
        if clear_separated:
            release_separated_object = None

    def step_with_follow(action: Any):
        nonlocal release_anchor_object, release_anchor_qpos, release_separated_object

        robot = env.robots[0]
        command = _right_gripper_command(robot, action)
        held_before_step = controller._virtual_held_object
        separation_distance = float(
            getattr(env, "_wa02_release_separation_distance", 0.0) or 0.0
        )
        release_assist_enabled = bool(
            held_before_step in CUP_IDS
            and command is not None
            and command < 0.0
            and separation_distance > 0.0
            and release_separated_object != held_before_step
        )
        if release_assist_enabled and (
            release_anchor_object != held_before_step or release_anchor_qpos is None
        ):
            obj = env.objects[str(held_before_step)]
            qpos = np.asarray(
                env.sim.data.get_joint_qpos(obj.joints[0]), dtype=float
            ).reshape(-1)
            if qpos.size >= 7 and np.all(np.isfinite(qpos[:7])):
                release_anchor_object = str(held_before_step)
                release_anchor_qpos = qpos.copy()
                trace = _release_separation_trace(env)
                trace.update(
                    {
                        "captured": True,
                        "cup_id": str(held_before_step),
                        "distance_threshold_m": float(separation_distance),
                        "anchor_position_world": qpos[:3].astype(float).tolist(),
                    }
                )

        obs, reward, done, info = original_step(action)
        held = controller._virtual_held_object
        transform = controller._virtual_object_from_eef
        if held not in CUP_IDS or transform is None or command is None:
            clear_release_anchor(clear_separated=True)
            return obs, reward, done, info

        if command < 0.0 and separation_distance > 0.0:
            if (
                release_separated_object == held
                or release_anchor_object != held
                or release_anchor_qpos is None
            ):
                return obs, reward, done, info
            eef_world = controller.stage_executor.current_eef_site_world()
            separation = _horizontal_release_separation(
                release_anchor_qpos,
                eef_world,
            )
            trace = _release_separation_trace(env)
            trace["last_horizontal_separation_m"] = float(separation)
            trace["max_horizontal_separation_m"] = max(
                float(trace.get("max_horizontal_separation_m", 0.0)),
                float(separation),
            )
            if separation >= separation_distance:
                release_separated_object = str(held)
                trace["separated"] = True
                trace["separation_distance_m"] = float(separation)
                clear_release_anchor()
                return obs, reward, done, info

            obj = env.objects[str(held)]
            env.sim.data.set_joint_qpos(obj.joints[0], release_anchor_qpos.copy())
            try:
                qvel = np.asarray(
                    env.sim.data.get_joint_qvel(obj.joints[0]), dtype=float
                ).reshape(-1)
                env.sim.data.set_joint_qvel(obj.joints[0], np.zeros_like(qvel))
            except (AttributeError, KeyError, TypeError, ValueError):
                pass
            env.sim.forward()
            trace["pinned_steps"] = int(trace.get("pinned_steps", 0)) + 1
            return env._get_observations(force_update=True), reward, done, info

        if command <= 0.0:
            clear_release_anchor()
            return obs, reward, done, info

        clear_release_anchor(clear_separated=True)
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
    env._wa02_held_cup_follow_installed = True


def _right_gripper_command_is_closing(robot: Any, action: Any) -> bool:
    """Read the named gripper slice instead of assuming it is action[-1]."""

    command = _right_gripper_command(robot, action)
    return bool(command is not None and command > 0.0)


def _right_gripper_command(robot: Any, action: Any) -> float | None:
    """Return the scalar Panda gripper command from the named action slice."""

    values = np.asarray(action, dtype=float).reshape(-1)
    indexes = getattr(robot.composite_controller, "_action_split_indexes", {})
    bounds = indexes.get("right_gripper")
    if not isinstance(bounds, tuple) or len(bounds) != 2:
        return None
    start, end = (int(bounds[0]), int(bounds[1]))
    if start < 0 or end <= start or end > values.size:
        return None
    command = values[start:end]
    if command.size != 1 or not np.isfinite(command[0]):
        return None
    return float(command[0])


def _horizontal_release_separation(
    anchor_qpos: Any,
    eef_world: Any,
) -> float:
    """Measure tabletop-plane clearance between a released cup and the TCP."""

    anchor = np.asarray(anchor_qpos, dtype=float).reshape(-1)
    eef = np.asarray(eef_world, dtype=float)
    if anchor.size < 3 or eef.shape != (4, 4):
        return 0.0
    return float(np.linalg.norm(eef[:2, 3] - anchor[:2]))


def _release_separation_trace(env: Any) -> dict[str, Any]:
    trace = getattr(env, "_wa02_release_separation_trace", None)
    if not isinstance(trace, dict):
        trace = {}
        env._wa02_release_separation_trace = trace
    return trace


def _section(payload: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = payload.get(name, {})
    if not isinstance(value, Mapping):
        raise ValueError(f"WA-02 manipulation.{name} must be a mapping")
    return value


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)


__all__ = ["ManipulationAssembly", "build_manipulation_assembly"]
