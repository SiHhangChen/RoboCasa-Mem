#!/usr/bin/env python3
"""Instantiate TM03 and audit its frozen stove/drawer/toaster-oven scene."""

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

from bench.env.robocasa_runtime import (  # noqa: E402
    bootstrap_local_robocasa_sources,
    make_joint_position_controller_config,
    make_robocasa_env,
    reset_env_with_initial_arm_qpos,
)
from bench.tasks.tm03.config import (  # noqa: E402
    load_task_config,
    missing_calibration_resources,
    resolve_observer_config,
)
from bench.tasks.tm03.domain.contracts import (  # noqa: E402
    CANONICAL_PROMPT,
    DRAWER_NAMES,
    EXPECTED_TOASTER_OVEN_ASSET,
)
from bench.tasks.tm03.domain.episode import resolve_scene_config  # noqa: E402


POLICY_CAMERAS = (
    "robot0_agentview_right",
    "robot0_agentview_left",
    "robot0_eye_in_hand",
)


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
    if args.width <= 0 or args.height <= 0:
        raise ValueError("TM03 probe image dimensions must be positive")
    if args.settle_steps < 0:
        raise ValueError("TM03 probe settle steps must be non-negative")

    bootstrap_local_robocasa_sources()
    # Import after source bootstrapping so the environment is registered.
    from bench.tasks.tm03.environment import (  # noqa: PLC0415
        TM03BoilWaterInspectDrawersAndRunToasterOven,
    )

    del TM03BoilWaterInspectDrawersAndRunToasterOven
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
            "completion_button_world_xyzs": dict(
                scene.completion_button_world_xyzs
            ),
            "expected_stove_asset": scene.expected_stove_asset,
            "knob_id": scene.knob_id,
            "burner_id": scene.burner_id,
            "pot_name": scene.pot_name,
            "pot_xml": scene.pot_xml,
            "expected_toaster_oven_asset": scene.expected_toaster_oven_asset,
            "toaster_oven_food_name": scene.toaster_oven_food_name,
            "toaster_oven_food_xml": scene.toaster_oven_food_xml,
            "toaster_oven_time_joint": scene.toaster_oven_time_joint,
            "toaster_oven_time_collision_geom": (
                scene.toaster_oven_time_collision_geom
            ),
            "observer_config": asdict(observer),
        },
        camera_names=POLICY_CAMERAS,
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
        for _ in range(int(args.settle_steps)):
            env.sim.step()
        env.sim.forward()
        report = scene_report(
            env,
            config=config,
            expected_layout_id=scene.layout_id,
            expected_style_id=scene.style_id,
            food_name=scene.toaster_oven_food_name,
        )
        if args.image_dir is not None:
            report["rendered_images"] = render_cameras(
                env,
                output_dir=args.image_dir,
                width=int(args.width),
                height=int(args.height),
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


def scene_report(
    env: Any,
    *,
    config: Any,
    expected_layout_id: int,
    expected_style_id: int,
    food_name: str,
) -> dict[str, Any]:
    """Return a JSON-safe report without changing any episode state."""

    toaster_oven_asset = Path(env.toaster_oven.file).parent.name
    time_knob_position = env.toaster_oven_time_position()
    metadata = env.get_ep_meta()
    diagnostics = env.task_diagnostics()
    drawer_fractions = {
        name: float(env.drawer_open_fraction(name)) for name in DRAWER_NAMES
    }
    food_body_id = (getattr(env, "obj_body_id", {}) or {}).get(food_name)
    food_position = (
        None
        if food_body_id is None
        else np.asarray(env.sim.data.body_xpos[int(food_body_id)], dtype=float)
    )
    missing = missing_calibration_resources(config)
    checks = {
        "environment_class": (
            type(env).__name__
            == "TM03BoilWaterInspectDrawersAndRunToasterOven"
        ),
        "layout_id": int(env.layout_id) == int(expected_layout_id),
        "style_id": int(env.style_id) == int(expected_style_id),
        "stove_asset": Path(env.stove.file).parent.name
        == str(config.scene["expected_stove_asset"]),
        "knob_initially_off": float(env.native_knob_qpos())
        <= float(env.physical_observer.config.off_commit_max_rad),
        "target_burner_initially_off": not bool(env.bound_burner_on()),
        "pot_stable_on_burner": bool(env.pot_stable_on_burner()),
        "two_drawers_exist": tuple(env.drawer_names) == tuple(DRAWER_NAMES),
        "drawers_initially_closed": all(
            fraction <= float(env.physical_observer.config.drawer_closed_threshold)
            for fraction in drawer_fractions.values()
        ),
        "toaster_oven_asset": toaster_oven_asset == EXPECTED_TOASTER_OVEN_ASSET,
        "time_knob_position_finite": bool(
            np.all(np.isfinite(time_knob_position))
        ),
        "toaster_oven_initially_off": not bool(env.toaster_oven_on()),
        "toaster_oven_door_initially_closed": bool(env.toaster_oven_door_closed()),
        "toaster_oven_food_object_exists": food_body_id is not None,
        "toaster_oven_food_initially_inside": bool(env.toaster_oven_food_inside()),
        "observer_initial_state_valid": not bool(
            env.physical_observer.invalid_environment_labels
        ),
        "fixed_prompt": metadata.get("lang") == CANONICAL_PROMPT
        and metadata.get("initial_prompt") == CANONICAL_PROMPT,
    }
    return {
        "contract_pass": all(checks.values()),
        "production_ready": not missing,
        "checks": checks,
        "environment": {
            "class": type(env).__name__,
            "layout_id": int(env.layout_id),
            "style_id": int(env.style_id),
            "stove_asset": Path(env.stove.file).parent.name,
            "toaster_oven_fixture": str(env.toaster_oven.name),
            "toaster_oven_asset": toaster_oven_asset,
        },
        "stove_and_drawers": {
            "knob_qpos_rad": float(env.native_knob_qpos()),
            "target_burner_on": bool(env.bound_burner_on()),
            "pot_stable": bool(env.pot_stable_on_burner()),
            "drawer_open_fraction": drawer_fractions,
        },
        "toaster_oven": {
            "on": bool(env.toaster_oven_on()),
            "door_closed": bool(env.toaster_oven_door_closed()),
            "food_name": str(food_name),
            "food_world_xyz": (
                None if food_position is None else food_position.tolist()
            ),
            "food_inside": bool(env.toaster_oven_food_inside()),
            "time_knob_joint": env.toaster_oven_time_joint_name(),
            "time_knob_geom": env.toaster_oven_time_geom_name(),
            "time_knob_world_xyz": time_knob_position.tolist(),
            "time_knob_qpos_rad": env.toaster_oven_time_qpos(),
        },
        "observer": {
            "invalid_environment_labels": list(
                env.physical_observer.invalid_environment_labels
            ),
            "diagnostics": diagnostics,
        },
        "calibration": {
            "missing_resources": list(missing),
            "note": (
                "Scene contract can pass before navigation and toaster-oven "
                "interaction poses are production-calibrated."
            ),
        },
    }


def render_cameras(
    env: Any, *, output_dir: Path, width: int, height: int
) -> dict[str, str]:
    import imageio.v2 as imageio  # noqa: PLC0415

    output_dir.mkdir(parents=True, exist_ok=True)
    result: dict[str, str] = {}
    for camera in POLICY_CAMERAS:
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


if __name__ == "__main__":
    raise SystemExit(main())
