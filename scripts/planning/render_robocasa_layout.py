#!/usr/bin/env python3
"""Render policy-camera previews for a RoboCasa layout/style pair."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import imageio.v2 as imageio

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bench.env.robocasa_runtime import (  # noqa: E402
    bootstrap_local_robocasa_sources,
    make_joint_position_controller_config,
    make_robocasa_env,
    reset_env_with_initial_arm_qpos,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layout-id", type=int, required=True)
    parser.add_argument("--style-id", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=720)
    args = parser.parse_args()

    bootstrap_local_robocasa_sources()
    from robocasa.models.fixtures.fixture import FixtureType

    camera_names = (
        "robot0_agentview_right",
        "robot0_agentview_left",
        "robot0_eye_in_hand",
    )
    env = make_robocasa_env(
        env_name="Kitchen",
        split="pretrain",
        seed=0,
        controller_config=make_joint_position_controller_config(),
        env_kwargs={"init_robot_base_ref": FixtureType.STOVE},
        camera_names=camera_names,
        camera_widths=args.width,
        camera_heights=args.height,
        layout_id=args.layout_id,
        style_id=args.style_id,
        has_renderer=False,
        has_offscreen_renderer=True,
        use_camera_obs=False,
    )
    try:
        reset_env_with_initial_arm_qpos(env)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        for camera_name in camera_names:
            rgb = env.sim.render(
                camera_name=camera_name,
                width=args.width,
                height=args.height,
                depth=False,
            )
            path = args.output_dir / f"{camera_name}.png"
            imageio.imwrite(path, rgb[::-1])
            print(path)
        print(f"layout={env.layout_id} style={env.style_id}")
        for name, fixture in env.fixtures.items():
            if "stovetop" in name.lower():
                print(f"stovetop fixture: {name} ({type(fixture).__name__})")
    finally:
        env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
