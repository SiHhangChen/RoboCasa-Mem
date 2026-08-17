#!/usr/bin/env python3
"""Render TS02's completion button from the robot policy cameras."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bench.execution.probe import reset_base_to_station, resolve_station_pose  # noqa: E402
from bench.runtime import RoboCasaRuntimeFactory  # noqa: E402
from bench.tasks import TaskCatalog, resolve_builtin_task_plugin  # noqa: E402
from bench.tasks.contracts import EpisodeRequest  # noqa: E402
from bench.tasks.ts02.button_asset import ts02_button_geom_names  # noqa: E402


CAMERAS = ("robot0_agentview_left", "robot0_agentview_right")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=720)
    args = parser.parse_args()

    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    episode = TaskCatalog.resolve(EpisodeRequest(task_id="TS02", seed=args.seed))
    api = resolve_builtin_task_plugin("TS02").load_runtime_api()
    runtime = RoboCasaRuntimeFactory(api).create(episode, execute=False, expert=True)
    try:
        station = resolve_station_pose(
            runtime,
            station_id="completion_button_station",
        )
        reset_base_to_station(runtime, station)
        runtime.env.sim.forward()
        args.output_dir.mkdir(parents=True, exist_ok=True)
        geom_name = str(runtime.env.COMPLETION_BUTTON_GEOM_NAME)
        joint_name = str(runtime.env.COMPLETION_BUTTON_JOINT_NAME)

        frames: list[Image.Image] = []
        images: dict[str, str] = {}
        for camera in CAMERAS:
            rgb = np.asarray(
                runtime.env.sim.render(
                    camera_name=camera,
                    width=args.width,
                    height=args.height,
                    depth=False,
                ),
                dtype=np.uint8,
            )[::-1]
            path = args.output_dir / f"ts02_button_{camera}.png"
            imageio.imwrite(path, rgb)
            images[camera] = str(path.resolve())
            panel = Image.fromarray(rgb)
            ImageDraw.Draw(panel).rectangle((0, 0, args.width, 36), fill=(20, 20, 20))
            ImageDraw.Draw(panel).text((12, 10), camera, fill=(255, 255, 255))
            frames.append(panel)

        sheet = Image.new("RGB", (args.width * len(frames), args.height))
        for index, frame in enumerate(frames):
            sheet.paste(frame, (index * args.width, 0))
        collage = args.output_dir / "ts02_tm01_style_button_robot_views.png"
        sheet.save(collage)

        geom_names = ts02_button_geom_names(geom_name)
        geom_ids = {
            name: int(runtime.env.sim.model.geom_name2id(name)) for name in geom_names
        }
        joint_range = np.asarray(
            runtime.env.sim.model.jnt_range[
                int(runtime.env.sim.model.joint_name2id(joint_name))
            ],
            dtype=float,
        )
        print(
            json.dumps(
                {
                    "collage": str(collage.resolve()),
                    "images": images,
                    "button_position_world": runtime.env.completion_button_position().tolist(),
                    "button_geom_count": len(geom_ids),
                    "button_geom_ids": geom_ids,
                    "joint_name": joint_name,
                    "joint_range_m": joint_range.tolist(),
                },
                indent=2,
                sort_keys=True,
            )
        )
    finally:
        runtime.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
