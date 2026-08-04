#!/usr/bin/env python3
"""Instantiate TM01 and audit its frozen layout-39/style-24 scene contract."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bench.env.robocasa_runtime import (
    bootstrap_local_robocasa_sources,
    make_joint_position_controller_config,
    make_robocasa_env,
    reset_env_with_initial_arm_qpos,
)
from bench.tasks.tm01.config import load_task_config, resolve_observer_config
from bench.tasks.tm01.domain.contracts import (
    CANONICAL_PROMPT,
    COMPLETION_BUTTON_NAMES,
    DRAWER_SIDE_COMPLETION_BUTTON,
    STOVE_SIDE_COMPLETION_BUTTON,
)
from bench.tasks.tm01.domain.episode import resolve_scene_config
from bench.tasks.wa02_button_asset import wa02_button_geom_names


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON report path.",
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=None,
        help="Optionally render the three policy cameras as PNG files.",
    )
    parser.add_argument(
        "--station-poses",
        type=Path,
        default=None,
        help=(
            "Optional station-pose JSON. Each station's first valid sample is "
            "rendered from the robot policy cameras."
        ),
    )
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    bootstrap_local_robocasa_sources()
    # Import after source bootstrapping so the class registers with robosuite.
    from bench.tasks.tm01.environment import (  # noqa: PLC0415
        COMPLETION_BUTTON_GEOM_NAMES,
        COMPLETION_BUTTON_JOINT_NAMES,
        DIAGNOSTIC_DRAWER_CAMERA,
        DIAGNOSTIC_OVERVIEW_CAMERA,
        TM01BoilWaterAndInspectTwoDrawers,
    )

    del TM01BoilWaterAndInspectTwoDrawers
    config = load_task_config()
    scene = resolve_scene_config(env_split="pretrain", **dict(config.scene))
    observer = resolve_observer_config(config)
    env = make_robocasa_env(
        env_name=scene.env_name,
        split=scene.env_split,
        seed=scene.scene_seed,
        controller_config=make_joint_position_controller_config(),
        env_kwargs={
            "drawer_names": scene.drawer_names,
            "drawer_world_xyzs": scene.drawer_world_xyzs,
            "drawer_world_yaw_deg": scene.drawer_world_yaw_deg,
            "completion_button_world_xyzs": dict(scene.completion_button_world_xyzs),
            "expected_stove_asset": scene.expected_stove_asset,
            "knob_id": scene.knob_id,
            "burner_id": scene.burner_id,
            "pot_name": scene.pot_name,
            "pot_xml": scene.pot_xml,
            "observer_config": asdict(observer),
        },
        camera_names=(
            "robot0_agentview_right",
            "robot0_agentview_left",
            "robot0_eye_in_hand",
        ),
        camera_widths=int(args.width),
        camera_heights=int(args.height),
        layout_id=scene.layout_id,
        style_id=scene.style_id,
        has_renderer=False,
        has_offscreen_renderer=args.image_dir is not None,
        use_camera_obs=False,
        camera_depths=False,
    )
    try:
        reset_env_with_initial_arm_qpos(env)
        report = _scene_report(
            env,
            completion_button_geom_names=COMPLETION_BUTTON_GEOM_NAMES,
            completion_button_joint_names=COMPLETION_BUTTON_JOINT_NAMES,
        )
        if args.image_dir is not None:
            report["rendered_images"] = _render_diagnostic_and_policy_cameras(
                env,
                output_dir=args.image_dir,
                width=int(args.width),
                height=int(args.height),
                diagnostic_cameras=(
                    DIAGNOSTIC_OVERVIEW_CAMERA,
                    DIAGNOSTIC_DRAWER_CAMERA,
                ),
            )
            if args.station_poses is not None:
                report["station_views"] = _render_station_views(
                    env,
                    station_pose_path=args.station_poses,
                    output_dir=args.image_dir / "stations",
                    width=int(args.width),
                    height=int(args.height),
                    overview_camera=DIAGNOSTIC_DRAWER_CAMERA,
                )
    finally:
        close = getattr(env, "close", None)
        if callable(close):
            close()

    encoded = json.dumps(report, indent=2, sort_keys=True)
    print(encoded)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")
    return 0 if report["contract_pass"] else 1


def _scene_report(
    env: Any,
    *,
    completion_button_geom_names: dict[str, str],
    completion_button_joint_names: dict[str, str],
) -> dict[str, Any]:
    joint_name = env.native_knob_joint_name()
    knob_geom_name = env.native_knob_geom_name()
    burner_site = str(env.stove.burner_sites[env.burner].get("name"))
    joint_id = int(env.sim.model.joint_name2id(joint_name))
    geom_id = int(env.sim.model.geom_name2id(knob_geom_name))
    site_id = int(env.sim.model.site_name2id(burner_site))
    button_geom_ids = {
        name: int(env.sim.model.geom_name2id(completion_button_geom_names[name]))
        for name in COMPLETION_BUTTON_NAMES
    }
    button_joint_ids = {
        name: int(env.sim.model.joint_name2id(completion_button_joint_names[name]))
        for name in COMPLETION_BUTTON_NAMES
    }
    pot_position = env.pot_world_position()
    burner_position = env.burner_world_position()
    drawer_report = {
        name: {
            "fixture_name": env.cabinet_for_drawer(name).name,
            "open_fraction": env.drawer_open_fraction(name),
            "pose": env.cabinet_for_drawer(name).drawer_pose().tolist(),
        }
        for name in env.drawer_names
    }
    all_geom_names = [
        env.sim.model.geom_id2name(index) or ""
        for index in range(int(env.sim.model.ngeom))
    ]
    task_button_geoms = sorted(
        name for name in all_geom_names if name.startswith("bench_tm01_")
    )
    forbidden_button_geoms = sorted(
        name
        for name in all_geom_names
        if "red_button" in name.lower() or "burner_button" in name.lower()
    )
    metadata = env.get_ep_meta()
    counter_bbox = np.asarray(env.counter.get_bbox_points(), dtype=float)
    counter_min = np.min(counter_bbox, axis=0)
    counter_max = np.max(counter_bbox, axis=0)
    stove_bbox = np.asarray(env.stove.get_bbox_points(), dtype=float)
    stove_min = np.min(stove_bbox, axis=0)
    stove_max = np.max(stove_bbox, axis=0)
    button_half_width = 0.060
    button_report: dict[str, dict[str, Any]] = {}
    for name in COMPLETION_BUTTON_NAMES:
        button_base = np.asarray(env.completion_button_world_xyzs[name], dtype=float)
        geom_id_for_button = button_geom_ids[name]
        on_countertop = bool(
            counter_min[0] + button_half_width
            <= button_base[0]
            <= counter_max[0] - button_half_width
            and counter_min[1] + button_half_width
            <= button_base[1]
            <= counter_max[1] - button_half_width
            and abs(float(button_base[2] - counter_max[2])) <= 0.015
        )
        button_report[name] = {
            "geom_name": completion_button_geom_names[name],
            "joint_name": completion_button_joint_names[name],
            "world_xyz": env.completion_button_position(name).tolist(),
            "base_world_xyz": button_base.tolist(),
            "on_countertop": on_countertop,
            "press_depth_m": env.completion_button_press_depth(name),
            "rgba": np.asarray(
                env.sim.model.geom_rgba[geom_id_for_button],
                dtype=float,
            ).tolist(),
        }
    button_geoms = set(completion_button_geom_names.values())
    expected_task_button_geoms = {
        component
        for geom_name in button_geoms
        for component in wa02_button_geom_names(geom_name)
    }
    checks = {
        "environment_class": type(env).__name__ == "TM01BoilWaterAndInspectTwoDrawers",
        "layout_id": int(env.layout_id) == 39,
        "style_id": int(env.style_id) == 24,
        "target_knob_joint_exists": joint_id >= 0,
        "target_knob_geom_exists": geom_id >= 0,
        "target_burner_site_exists": site_id >= 0,
        "joint_range_is_zero_to_pi": np.allclose(
            np.asarray(env.sim.model.jnt_range[joint_id], dtype=float),
            [0.0, np.pi],
        ),
        "knob_initially_off": env.native_knob_qpos() <= 0.10,
        "target_burner_initially_off": not bool(
            env.bound_burner_on()
        ),
        "other_burners_initially_off": all(
            not bool(env.stove.is_burner_on(env=env, burner_loc=location))
            for location in env.stove.get_knobs_state(env)
            if location != env.knob
        ),
        "two_drawers_exist": tuple(env.drawer_names) == ("drawer_left", "drawer_right"),
        "drawers_initially_closed": all(
            value["open_fraction"] <= 0.08 for value in drawer_report.values()
        ),
        "two_completion_button_geoms_exist": all(
            button_geom_ids[name] >= 0 for name in COMPLETION_BUTTON_NAMES
        ),
        "two_completion_button_joints_exist": all(
            button_joint_ids[name] >= 0 for name in COMPLETION_BUTTON_NAMES
        ),
        "both_completion_buttons_are_green": all(
            float(env.sim.model.geom_rgba[button_geom_ids[name]][1])
            > float(env.sim.model.geom_rgba[button_geom_ids[name]][0])
            and float(env.sim.model.geom_rgba[button_geom_ids[name]][1])
            > float(env.sim.model.geom_rgba[button_geom_ids[name]][2])
            for name in COMPLETION_BUTTON_NAMES
        ),
        "both_completion_buttons_released": all(
            env.completion_button_press_depth(name) <= 0.001
            for name in COMPLETION_BUTTON_NAMES
        ),
        "both_completion_buttons_on_countertop": all(
            bool(button_report[name]["on_countertop"])
            for name in COMPLETION_BUTTON_NAMES
        ),
        "exactly_two_task_button_plungers": all(
            name in task_button_geoms for name in button_geoms
        )
        and len(button_geoms) == 2,
        "completion_buttons_match_full_wa02_asset": (
            set(task_button_geoms) == expected_task_button_geoms
        ),
        "drawer_side_button_is_north_of_stove": bool(
            env.completion_button_world_xyzs[DRAWER_SIDE_COMPLETION_BUTTON][1]
            > stove_max[1] + button_half_width
        ),
        "stove_side_button_is_right_of_stove": bool(
            env.completion_button_world_xyzs[STOVE_SIDE_COMPLETION_BUTTON][0]
            > stove_max[0] + button_half_width
        ),
        "no_red_or_burner_button": not forbidden_button_geoms,
        "pot_within_burner_region": float(
            np.linalg.norm(pot_position[:2] - burner_position[:2])
        )
        <= 0.08,
        "pot_stable": env.pot_stable_on_burner(),
        "observer_initial_state_valid": not (
            env.physical_observer.invalid_environment_labels
        ),
        "fixed_prompt": metadata.get("lang") == CANONICAL_PROMPT
        and metadata.get("initial_prompt") == CANONICAL_PROMPT,
        "knob_actuator_not_commanded": not env._knob_actuator_commanded(),
    }
    return {
        "contract_pass": all(checks.values()),
        "checks": checks,
        "environment": {
            "class": type(env).__name__,
            "layout_id": int(env.layout_id),
            "style_id": int(env.style_id),
            "stove_name": str(env.stove.name),
            "stove_asset": Path(env.stove.file).parent.name,
            "stove_naming_prefix": str(env.stove.naming_prefix),
        },
        "native_stove_control": {
            "knob_id": str(env.knob),
            "bound_burner_id": str(env.burner),
            "joint_name": joint_name,
            "joint_range_rad": np.asarray(
                env.sim.model.jnt_range[joint_id],
                dtype=float,
            ).tolist(),
            "joint_qpos_rad": env.native_knob_qpos(),
            "joint_qvel_rad_s": env.native_knob_qvel(),
            "collision_geom": knob_geom_name,
            "burner_site": burner_site,
            "burner_world_xyz": burner_position.tolist(),
            "burner_on": bool(env.bound_burner_on()),
            "actuator_commanded": env._knob_actuator_commanded(),
        },
        "pot": {
            "name": str(env.pot_name),
            "world_xyz": pot_position.tolist(),
            "burner_xy_distance_m": float(
                np.linalg.norm(pot_position[:2] - burner_position[:2])
            ),
            "stable": env.pot_stable_on_burner(),
        },
        "drawers": drawer_report,
        "completion_buttons": {
            **button_report,
            "support_counter_bbox_min": counter_min.tolist(),
            "support_counter_bbox_max": counter_max.tolist(),
            "stove_bbox_min": stove_min.tolist(),
            "stove_bbox_max": stove_max.tolist(),
            "task_button_geoms": task_button_geoms,
            "forbidden_button_geoms": forbidden_button_geoms,
        },
        "observer": {
            "invalid_environment_labels": list(
                env.physical_observer.invalid_environment_labels
            ),
            "diagnostics": env.task_diagnostics(),
        },
    }


def _render_diagnostic_and_policy_cameras(
    env: Any,
    *,
    output_dir: Path,
    width: int,
    height: int,
    diagnostic_cameras: tuple[str, ...],
) -> dict[str, str]:
    import imageio.v2 as imageio  # noqa: PLC0415

    output_dir.mkdir(parents=True, exist_ok=True)
    result: dict[str, str] = {}
    for camera in (
        *diagnostic_cameras,
        "robot0_agentview_right",
        "robot0_agentview_left",
        "robot0_eye_in_hand",
    ):
        image = np.asarray(
            env.sim.render(
                width=int(width),
                height=int(height),
                camera_name=camera,
                depth=False,
            ),
            dtype=np.uint8,
        )[::-1]
        path = output_dir / f"{camera}.png"
        imageio.imwrite(path, image)
        result[camera] = str(path.resolve())
    return result


def _render_station_views(
    env: Any,
    *,
    station_pose_path: Path,
    output_dir: Path,
    width: int,
    height: int,
    overview_camera: str,
) -> dict[str, Any]:
    import imageio.v2 as imageio  # noqa: PLC0415

    from bench.execution.navigation.base_motion import set_base_pose_kinematic  # noqa: PLC0415
    from bench.execution.navigation.station_poses import (  # noqa: PLC0415
        load_station_pose_payload,
        station_sample_records,
    )

    payload = load_station_pose_payload(station_pose_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    result: dict[str, Any] = {}
    cameras = (
        overview_camera,
        "robot0_agentview_right",
        "robot0_agentview_left",
        "robot0_eye_in_hand",
    )
    for station_id, station in (payload.get("stations", {}) or {}).items():
        records = station_sample_records(list(station.get("samples", [])))
        if not records:
            result[str(station_id)] = {"valid": False, "reason": "no_valid_samples"}
            continue
        record = records[0]
        target_xy = np.asarray(record["target_xy"], dtype=float)
        target_yaw = float(record["target_yaw"])
        set_base_pose_kinematic(
            env,
            target_xy=target_xy,
            target_yaw=target_yaw,
        )
        rendered: dict[str, str] = {}
        images: list[np.ndarray] = []
        for camera in cameras:
            rgb = np.asarray(
                env.sim.render(
                    width=int(width),
                    height=int(height),
                    camera_name=camera,
                    depth=False,
                ),
                dtype=np.uint8,
            )[::-1]
            path = output_dir / f"{station_id}__{camera}.png"
            imageio.imwrite(path, rgb)
            rendered[camera] = str(path.resolve())
            images.append(rgb)
        montage = np.concatenate(
            (
                np.concatenate((images[0], images[1]), axis=1),
                np.concatenate((images[2], images[3]), axis=1),
            ),
            axis=0,
        )
        montage_path = output_dir / f"{station_id}__montage.png"
        imageio.imwrite(montage_path, montage)
        result[str(station_id)] = {
            "valid": True,
            "base_xy": target_xy.tolist(),
            "base_yaw": target_yaw,
            "torso_height": record["torso_height"],
            "rendered_images": rendered,
            "montage": str(montage_path.resolve()),
        }
    return result


if __name__ == "__main__":
    raise SystemExit(main())
