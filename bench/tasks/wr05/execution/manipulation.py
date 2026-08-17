"""WR-05 manipulation assembly.

The plate re-pick uses a WA-02-style low side-grasp selector: it builds
  ``mug-body-side`` candidates at several yaw offsets, dry-runs each full
  pregrasp/grasp/lift chain, and executes the shortest reachable candidate.
  Object motion is always produced by MuJoCo contacts; this module never
  writes a condiment body's free-joint pose.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from bench.execution.manipulation import ManipulationController, ManipulationRuntimeConfig
from bench.execution.planning import (
    MplibPlannerConfig,
    PickPlaceConfig,
    make_object_pick_target_poses,
    matrix_from_json_pose,
    matrix_from_pos_quat_wxyz,
    pose_from_matrix,
)
from bench.execution.planning.mplib_planner import run_mplib_planner_subprocess


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
            script_path=planner_script,
            conda_env=str(planner.get("conda_env", "membench")),
            timeout_s=float(planner.get("subprocess_timeout_s", 60.0)),
        ),
        pick_place_config=_pick_place_config(_section(settings, "pick_place")),
        runtime_config=_runtime_config(
            execute=execute,
            runtime=_section(settings, "runtime"),
            planner=planner,
            static_scene=_section(settings, "static_scene"),
        ),
    )
    _install_plate_side_pick_selector(controller)
    return ManipulationAssembly(controller=controller, planner_settings=planner)


def _install_plate_side_pick_selector(controller: Any) -> None:
    """Select a reachable WA-02-style low side grasp for staged bottles."""

    inherited_selector = controller.pick_payload_selector

    def selector(request: dict[str, Any]) -> dict[str, Any] | None:
        kwargs = dict(request.get("kwargs") or {})
        raw_offsets = kwargs.get("plate_side_grasp_yaw_offsets_deg")
        if raw_offsets is None:
            return inherited_selector(request) if callable(inherited_selector) else None

        offsets = [float(value) for value in raw_offsets]
        # -90 deg approaches the staged bottle across the plate and cannot
        # reach the precise grasp pose physically (dry-run only checks plan
        # reachability). Exclude it unconditionally so a candidate list that
        # re-introduces it can never select it.
        offsets = [
            value for value in offsets if not np.isclose(value, -90.0)
        ]
        records: list[dict[str, Any]] = []
        preferred_yaw = _preferred_plate_side_grasp_yaw(
            request["env"],
            object_name=str(request["object_name"]),
        )
        passing: list[
            tuple[tuple[float, float, float], dict[str, Any], dict[str, Any]]
        ] = []
        for index, yaw_offset in enumerate(offsets):
            payload = make_object_pick_target_poses(
                request["env"],
                request["obs"],
                object_name=str(request["object_name"]),
                orientation_mode=str(kwargs.get("orientation_mode", "mug-body-side")),
                pregrasp_height=float(kwargs.get("pregrasp_height", 0.18)),
                grasp_clearance=float(kwargs.get("grasp_clearance", -0.02)),
                lift_height=float(kwargs.get("lift_height", 0.12)),
                grasp_yaw_offset_deg=yaw_offset,
                object_pos_world=kwargs.get("object_pos_world"),
                grasp_offset_x=float(kwargs.get("grasp_offset_x", 0.0)),
                grasp_offset_y=float(kwargs.get("grasp_offset_y", 0.0)),
                grasp_offset_z=float(kwargs.get("grasp_offset_z", 0.0)),
            )
            validation = request["dry_run_pick_sequence"](
                payload,
                object_name=str(request["object_name"]),
            )
            score = _pick_plan_score(
                validation,
                index=index,
                yaw_offset_deg=yaw_offset,
                preferred_yaw_deg=preferred_yaw,
            )
            record = {
                "index": int(index),
                "yaw_offset_deg": yaw_offset,
                "pass": bool(validation.get("pass", False)),
                "score": list(score),
            }
            records.append(record)
            if record["pass"]:
                passing.append((score, payload, record))
        if not passing:
            return {
                "payload": None,
                "selection": {
                    "pass": False,
                    "mode": "wr05_wa02_style_side_grasp",
                    "candidates": records,
                    "reason": "no_reachable_plate_side_grasp_candidate",
                },
            }
        score, payload, selected = min(passing, key=lambda item: item[0])
        # Generated side grasps normally do not expose the object-relative
        # attachment carried by manual grasp files. Persist the selected live
        # relation so navigation and the later shelf placement can reuse the
        # actual selected yaw instead of falling back to a stale shelf grasp.
        payload["manual_eef_site_in_object"] = _eef_in_object_for_pick_payload(
            request["env"],
            object_name=str(request["object_name"]),
            payload=payload,
        )
        return {
            "payload": payload,
            "selection": {
                "pass": True,
                "mode": "wr05_wa02_style_side_grasp",
                "preferred_yaw_offset_deg": preferred_yaw,
                "selected_yaw_offset_deg": selected["yaw_offset_deg"],
                "selected_score": list(score),
                "candidates": records,
            },
        }

    controller.pick_payload_selector = selector


def _preferred_plate_side_grasp_yaw(env: Any, *, object_name: str) -> float:
    """Use the physically validated side grasp for each bottle/plate slot."""
    object_name = str(object_name)
    use_assignment = dict(getattr(env, "use_assignment", {}) or {})
    if object_name == "mustard" and use_assignment.get(object_name) == "plate_left":
        return 45.0
    return 90.0


def _eef_in_object_for_pick_payload(
    env: Any,
    *,
    object_name: str,
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    body_id = env.obj_body_id[str(object_name)]
    object_world = matrix_from_pos_quat_wxyz(
        np.asarray(env.sim.data.body_xpos[body_id], dtype=float),
        np.asarray(env.sim.data.body_xquat[body_id], dtype=float),
    )
    grasp_world = matrix_from_json_pose(payload["poses"]["grasp"]["world"])
    return pose_from_matrix(np.linalg.inv(object_world) @ grasp_world)


def _pick_plan_score(
    validation: dict[str, Any],
    *,
    index: int,
    yaw_offset_deg: float | None = None,
    preferred_yaw_deg: float = 90.0,
) -> tuple[float, float, float]:
    if not bool(validation.get("pass", False)):
        return (1.0e12, 1.0e12, float(index))
    steps = 0.0
    for stage in validation.get("stages", ()):
        plan = stage.get("plan") or {}
        length = plan.get("trajectory_len")
        if length is not None:
            steps += float(length)
    # Physical grasp stability is more important than dry-run path length.
    # Prefer the bottle-calibrated side first, then use rounded trajectory
    # length and input order to rank the remaining candidates.
    preferred_yaw_rank = (
        0.0
        if yaw_offset_deg is not None
        and np.isclose(yaw_offset_deg, float(preferred_yaw_deg))
        else 1.0
    )
    return (preferred_yaw_rank, float(round(steps / 5.0) * 5.0), float(index))


def _pick_place_config(values: Mapping[str, Any]) -> PickPlaceConfig:
    settings = dict(values)
    config = replace(
        PickPlaceConfig(),
        orientation_mode=str(settings.pop("orientation_mode", "object-y-topdown")),
        mug_orientation_mode=str(settings.pop("mug_orientation_mode", "mug-body-side")),
        pregrasp_height=float(settings.pop("pregrasp_height", 0.15)),
        grasp_clearance=float(settings.pop("grasp_clearance", -0.04)),
        lift_height=float(settings.pop("lift_height", 0.12)),
        place_height=float(settings.pop("place_height", 0.10)),
        retreat_height=float(settings.pop("retreat_height", 0.18)),
        retreat_x=float(settings.pop("retreat_x", 0.0)),
        retreat_y=float(settings.pop("retreat_y", -0.10)),
    )
    if settings:
        raise ValueError("unknown WR-05 pick_place settings: " + ", ".join(sorted(settings)))
    return config


def _runtime_config(
    *, execute: bool, runtime: Mapping[str, Any], planner: Mapping[str, Any],
    static_scene: Mapping[str, Any],
) -> ManipulationRuntimeConfig:
    values = dict(runtime)
    planner_known = {
        "planning_time", "time_step", "qpos_step", "rrt_range", "conda_env",
        "subprocess_timeout_s", "preserve_ik_branch_stages", "nearest_ik_samples",
        "nearest_ik_threshold", "nearest_ik_max_joint_delta",
    }
    unknown_planner = sorted(set(planner) - planner_known)
    if unknown_planner:
        raise ValueError("unknown WR-05 planner settings: " + ", ".join(unknown_planner))
    planner_config = MplibPlannerConfig(
        planning_time=float(planner.get("planning_time", 5.0)),
        time_step=float(planner.get("time_step", 0.1)),
        qpos_step=float(planner.get("qpos_step", 0.02)),
        rrt_range=float(planner.get("rrt_range", 0.1)),
    )
    planner_args = planner_config.planner_args()
    planner_args.update(
        {
            "preserve_ik_branch_stages": list(planner.get("preserve_ik_branch_stages", ["place"])),
            "nearest_ik_samples": int(planner.get("nearest_ik_samples", 64)),
            "nearest_ik_threshold": float(planner.get("nearest_ik_threshold", 0.01)),
            "nearest_ik_max_joint_delta": float(planner.get("nearest_ik_max_joint_delta", 2.15)),
        }
    )
    scene = dict(static_scene)
    scene_options = {
        "enabled": bool(scene.pop("enabled", False)),
        "validate_collisions": bool(scene.pop("validate_collisions", False)),
        "fixture_keywords": str(scene.pop("fixture_keywords", "counter,sink,cabinet")),
        "exclude_keywords": str(scene.pop("exclude_keywords", "floor,wall")),
        "spacing": float(scene.pop("spacing", 0.04)),
        "inflate": float(scene.pop("inflate", 0.05)),
        "point_radius": float(scene.pop("point_radius", 0.01)),
        "faces": str(scene.pop("faces", "bottom")),
        "point_cloud_stages": str(scene.pop("point_cloud_stages", "pregrasp,grasp,lift,place,clearance,retreat")),
        "min_z": float(scene.pop("min_z", 1.0)),
        "max_points": int(scene.pop("max_points", 4000)),
    }
    if scene:
        raise ValueError("unknown WR-05 static_scene settings: " + ", ".join(sorted(scene)))
    config = ManipulationRuntimeConfig(
        execute=bool(execute),
        max_joint_step=float(values.pop("max_joint_step", 0.012)),
        settle_steps=int(values.pop("settle_steps", 80)),
        position_threshold=float(values.pop("position_threshold", 0.04)),
        orientation_threshold_deg=float(values.pop("orientation_threshold_deg", 15.0)),
        qpos_threshold=float(values.pop("qpos_threshold", 0.16)),
        open_steps=int(values.pop("open_steps", 20)),
        close_steps=int(values.pop("close_steps", 90)),
        release_steps=int(values.pop("release_steps", 45)),
        release_settle_steps=int(values.pop("release_settle_steps", 80)),
        planner_args=planner_args,
        static_scene_options=scene_options,
    )
    if values:
        raise ValueError("unknown WR-05 runtime settings: " + ", ".join(sorted(values)))
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
        raise ValueError(f"WR-05 manipulation.{name} must be a mapping")
    return value


__all__ = ["ManipulationAssembly", "build_manipulation_assembly"]
