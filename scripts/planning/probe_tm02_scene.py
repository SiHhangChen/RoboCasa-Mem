#!/usr/bin/env python3
"""Instantiate TM02 and audit its WA02-derived fixed scene contract."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bench.env.robocasa_runtime import (  # noqa: E402
    bootstrap_local_robocasa_sources,
    make_joint_position_controller_config,
    make_robocasa_env,
    reset_env_with_initial_arm_qpos,
)
from bench.tasks.tm02.config import (  # noqa: E402
    load_task_config,
    resolve_observer_config,
)
from bench.tasks.tm02.domain.contracts import (  # noqa: E402
    BUTTON_NAMES,
    CANONICAL_PROMPT,
    COMPLETION_BUTTON,
    WATER_BUTTON,
)
from bench.tasks.tm02.domain.episode import resolve_scene_config  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--image-dir", type=Path, default=None)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--settle-steps", type=int, default=20)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    bootstrap_local_robocasa_sources()
    from bench.tasks.tm02.environment import (  # noqa: PLC0415
        BUTTON_GEOM_NAMES,
        BUTTON_JOINT_NAMES,
        CABINET_SUPPORT_COUNTER_NAME,
        CAN_PLACE_REGION_GEOM,
        DIAGNOSTIC_OVERVIEW_CAMERA,
        DIAGNOSTIC_TOP_CAMERA,
        RED_CAN_XML,
        TM02FillPotAndRetrieveCan,
        WR01_SLIDE_CABINET_ASSET,
    )

    del TM02FillPotAndRetrieveCan
    config = load_task_config()
    scene = resolve_scene_config(env_split="pretrain", **dict(config.scene))
    observer = resolve_observer_config(config)
    env = make_robocasa_env(
        env_name=scene.env_name,
        split=scene.env_split,
        seed=scene.scene_seed,
        controller_config=make_joint_position_controller_config(),
        env_kwargs={
            "water_button_world_xyz": scene.water_button_world_xyz,
            "completion_button_world_xyz": scene.completion_button_world_xyz,
            "cabinet_world_xyz": scene.cabinet_world_xyz,
            "cabinet_yaw_deg": scene.cabinet_yaw_deg,
            "can_place_region_world_xyz": scene.can_place_region_world_xyz,
            "can_place_region_size_xy": scene.can_place_region_size_xy,
            "pot_name": scene.pot_name,
            "pot_xml": scene.pot_xml,
            "can_name": scene.can_name,
            "can_xml": scene.can_xml,
            "can_scale": scene.can_scale,
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
        for _ in range(max(0, int(args.settle_steps))):
            env.sim.step()
        env.sim.forward()
        report = _scene_report(
            env,
            button_geom_names=BUTTON_GEOM_NAMES,
            button_joint_names=BUTTON_JOINT_NAMES,
            place_region_geom=CAN_PLACE_REGION_GEOM,
            cabinet_support_counter_name=CABINET_SUPPORT_COUNTER_NAME,
            wr01_cabinet_asset=WR01_SLIDE_CABINET_ASSET,
            wr01_can_xml=RED_CAN_XML,
        )
        if args.image_dir is not None:
            report["rendered_images"] = _render_cameras(
                env,
                output_dir=args.image_dir,
                width=int(args.width),
                height=int(args.height),
                cameras=(
                    DIAGNOSTIC_OVERVIEW_CAMERA,
                    DIAGNOSTIC_TOP_CAMERA,
                    "robot0_agentview_right",
                    "robot0_agentview_left",
                    "robot0_eye_in_hand",
                ),
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
    button_geom_names: dict[str, str],
    button_joint_names: dict[str, str],
    place_region_geom: str,
    cabinet_support_counter_name: str,
    wr01_cabinet_asset: Path,
    wr01_can_xml: str,
) -> dict[str, Any]:
    button_geom_ids = {
        name: int(env.sim.model.geom_name2id(geom))
        for name, geom in button_geom_names.items()
    }
    button_joint_ids = {
        name: int(env.sim.model.joint_name2id(joint))
        for name, joint in button_joint_names.items()
    }
    marker_id = int(env.sim.model.geom_name2id(place_region_geom))
    cabinet_bbox = np.asarray(env.storage_cabinet.get_bbox_points(), dtype=float)
    cabinet_min = np.min(cabinet_bbox, axis=0)
    cabinet_max = np.max(cabinet_bbox, axis=0)
    support_bbox = np.asarray(env.cabinet_counter.get_bbox_points(), dtype=float)
    support_min = np.min(support_bbox, axis=0)
    support_max = np.max(support_bbox, axis=0)
    sink_bbox = np.asarray(env.sink.get_bbox_points(), dtype=float)
    sink_min = np.min(sink_bbox, axis=0)
    sink_max = np.max(sink_bbox, axis=0)
    coffee_machine_bbox = np.asarray(
        env.fixtures["coffee_machine_left_group_1"].get_bbox_points(),
        dtype=float,
    )
    coffee_machine_min = np.min(coffee_machine_bbox, axis=0)
    coffee_machine_max = np.max(coffee_machine_bbox, axis=0)
    stove_bbox = np.asarray(
        env.fixtures["stovetop_1_left_group_1"].get_bbox_points(),
        dtype=float,
    )
    stove_min = np.min(stove_bbox, axis=0)
    stove_max = np.max(stove_bbox, axis=0)
    upper_cabinet_bbox = np.asarray(
        env.fixtures["cab_2_left_group_1"].get_bbox_points(),
        dtype=float,
    )
    upper_cabinet_min = np.min(upper_cabinet_bbox, axis=0)
    horizontal_gap = _rectangle_gap(
        cabinet_min[:2],
        cabinet_max[:2],
        sink_min[:2],
        sink_max[:2],
    )
    stove_gap = _rectangle_gap(
        cabinet_min[:2],
        cabinet_max[:2],
        stove_min[:2],
        stove_max[:2],
    )
    upper_cabinet_vertical_gap = float(upper_cabinet_min[2] - cabinet_max[2])
    place_center = np.asarray(env.can_place_region_world_xyz, dtype=float)
    place_half = np.asarray(env.can_place_region_size_xy, dtype=float) * 0.5
    place_min = place_center[:2] - place_half
    place_max = place_center[:2] + place_half
    place_stove_gap = _rectangle_gap(
        place_min,
        place_max,
        stove_min[:2],
        stove_max[:2],
    )
    place_coffee_machine_gap = _rectangle_gap(
        place_min,
        place_max,
        coffee_machine_min[:2],
        coffee_machine_max[:2],
    )
    place_cabinet_gap = _rectangle_gap(
        place_min,
        place_max,
        cabinet_min[:2],
        cabinet_max[:2],
    )
    calibrated_place_station_xy = np.asarray([0.880, -2.518], dtype=float)
    place_station_target_distance = float(
        np.linalg.norm(calibrated_place_station_xy - place_center[:2])
    )
    metadata = env.get_ep_meta()
    diagnostics = env.task_diagnostics()
    button_report = {
        name: {
            "geom": button_geom_names[name],
            "joint": button_joint_names[name],
            "position": env.button_position(name).tolist(),
            "press_depth_m": env.button_press_depth(name),
            "rgba": np.asarray(
                env.sim.model.geom_rgba[button_geom_ids[name]],
                dtype=float,
            ).tolist(),
        }
        for name in BUTTON_NAMES
    }
    checks = {
        "environment_class": type(env).__name__ == "TM02FillPotAndRetrieveCan",
        "wa02_layout_id": int(env.layout_id) == 15,
        "wa02_style_id": int(env.style_id) == 34,
        "sink_exists": env.sink is not None,
        "water_site_exists": env.water_site_position() is not None,
        "faucet_initially_off": not env.faucet_is_on(),
        "pot_in_sink": env.pot_in_sink(),
        "pot_under_water": env.pot_under_water(),
        "pot_stable": env.pot_stable(),
        "single_cabinet_exists": env.storage_cabinet is not None,
        "cabinet_directly_reuses_wr01_class": (
            type(env.storage_cabinet).__name__ == "WR01SlideStorageCabinet"
        ),
        "cabinet_directly_reuses_wr01_asset": (
            Path(env.storage_cabinet.file).resolve()
            == Path(wr01_cabinet_asset).resolve()
        ),
        "can_directly_reuses_wr01_asset": (
            Path(env.can_xml).resolve() == Path(wr01_can_xml).resolve()
        ),
        "cabinet_initially_closed": env.cabinet_open_fraction() <= 0.08,
        "cabinet_support_counter_is_frozen": (
            env.cabinet_counter.name == cabinet_support_counter_name
        ),
        "cabinet_footprint_on_support_counter": bool(
            np.all(cabinet_min[:2] >= support_min[:2] - 0.05)
            and np.all(cabinet_max[:2] <= support_max[:2] + 0.05)
        ),
        "cabinet_supported_at_counter_height": bool(
            abs(float(cabinet_min[2] - support_max[2])) <= 0.03
        ),
        "cabinet_clear_of_sink": horizontal_gap >= 1.0,
        "cabinet_clear_of_stove": stove_gap >= 0.03,
        "cabinet_clear_of_upper_cabinet": upper_cabinet_vertical_gap >= 0.05,
        "can_initially_inside_cabinet": env.can_inside_cabinet(),
        "can_not_initially_in_place_region": not env.can_in_place_region(),
        "water_button_exists": button_geom_ids[WATER_BUTTON] >= 0
        and button_joint_ids[WATER_BUTTON] >= 0,
        "exactly_two_task_buttons": set(button_report) == set(BUTTON_NAMES),
        "water_button_is_blue": bool(
            button_report[WATER_BUTTON]["rgba"][2]
            > button_report[WATER_BUTTON]["rgba"][0]
        ),
        "water_button_initially_released": (
            env.button_press_depth(WATER_BUTTON) <= 0.001
        ),
        "completion_button_exists": (
            button_geom_ids[COMPLETION_BUTTON] >= 0
            and button_joint_ids[COMPLETION_BUTTON] >= 0
        ),
        "completion_button_is_red": bool(
            button_report[COMPLETION_BUTTON]["rgba"][0]
            > button_report[COMPLETION_BUTTON]["rgba"][2]
        ),
        "completion_button_initially_released": (
            env.button_press_depth(COMPLETION_BUTTON) <= 0.001
        ),
        "completion_button_is_left_of_water_button": bool(
            np.allclose(
                np.asarray(env.completion_button_world_xyz, dtype=float),
                np.asarray(env.button_world_xyzs[WATER_BUTTON], dtype=float)
                + np.asarray([0.0, 0.14, 0.0], dtype=float),
                atol=1e-9,
            )
        ),
        "completion_button_base_clearance_m": bool(
            np.linalg.norm(
                np.asarray(env.completion_button_world_xyz, dtype=float)[:2]
                - np.asarray(env.button_world_xyzs[WATER_BUTTON], dtype=float)[:2]
            )
            >= 0.12
        ),
        "place_region_marker_exists": marker_id >= 0,
        "place_region_on_stove_counter": bool(
            np.all(place_min >= support_min[:2])
            and np.all(place_max <= support_max[:2] + 0.005)
        ),
        "place_region_clear_of_stove": place_stove_gap >= 0.20,
        "place_region_clear_of_coffee_machine": place_coffee_machine_gap >= 0.20,
        "place_region_clear_of_cabinet": place_cabinet_gap >= 0.80,
        "place_region_matches_calibrated_station": bool(
            0.45 <= place_station_target_distance <= 0.75
        ),
        "observer_initial_state_valid": not (
            env.physical_observer.invalid_environment_labels
        ),
        "fixed_prompt": metadata.get("lang") == CANONICAL_PROMPT
        and metadata.get("initial_prompt") == CANONICAL_PROMPT,
        "initial_task_not_already_success": not diagnostics["success"],
    }
    return {
        "contract_pass": all(checks.values()),
        "checks": checks,
        "environment": {
            "class": type(env).__name__,
            "layout_id": int(env.layout_id),
            "style_id": int(env.style_id),
            "sink_name": str(env.sink.name),
            "counter_name": str(env.counter.name),
            "cabinet_support_counter_name": str(env.cabinet_counter.name),
            "cabinet_support_bbox_min": support_min.tolist(),
            "cabinet_support_bbox_max": support_max.tolist(),
        },
        "buttons": button_report,
        "pot": {
            "world_xyz": env.object_world_position(env.pot_name).tolist(),
            "water_site_world_xyz": env.water_site_position().tolist(),
        },
        "cabinet": {
            "world_xyz": list(env.cabinet_world_xyz),
            "yaw_deg": float(env.cabinet_yaw_deg),
            "bbox_min": cabinet_min.tolist(),
            "bbox_max": cabinet_max.tolist(),
            "sink_horizontal_gap_m": horizontal_gap,
            "stove_horizontal_gap_m": stove_gap,
            "upper_cabinet_vertical_gap_m": upper_cabinet_vertical_gap,
            "fixture_class": type(env.storage_cabinet).__name__,
            "asset_xml": str(Path(env.storage_cabinet.file).resolve()),
            "door_open_fraction": env.cabinet_open_fraction(),
        },
        "can": {
            "world_xyz": env.object_world_position(env.can_name).tolist(),
            "inside_cabinet": env.can_inside_cabinet(),
            "asset_xml": env.can_xml,
            "place_region_world_xyz": list(env.can_place_region_world_xyz),
            "place_region_size_xy": list(env.can_place_region_size_xy),
            "place_region_stove_gap_m": place_stove_gap,
            "place_region_coffee_machine_gap_m": place_coffee_machine_gap,
            "place_region_cabinet_gap_m": place_cabinet_gap,
            "calibrated_place_station_xy": calibrated_place_station_xy.tolist(),
            "place_station_target_distance_m": place_station_target_distance,
        },
        "diagnostics": diagnostics,
    }


def _rectangle_gap(
    a_min: np.ndarray,
    a_max: np.ndarray,
    b_min: np.ndarray,
    b_max: np.ndarray,
) -> float:
    delta = np.maximum(np.maximum(a_min - b_max, b_min - a_max), 0.0)
    return float(np.linalg.norm(delta))


def _render_cameras(
    env: Any,
    *,
    output_dir: Path,
    width: int,
    height: int,
    cameras: tuple[str, ...],
) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    for camera in cameras:
        rgb = env.sim.render(
            camera_name=camera,
            width=width,
            height=height,
            depth=False,
        )
        path = output_dir / f"{camera}.png"
        imageio.imwrite(path, rgb[::-1])
        paths.append(str(path))
    return paths


if __name__ == "__main__":
    raise SystemExit(main())
