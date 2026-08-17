#!/usr/bin/env python3
"""Render WX01's scene (current blocking cup + cleaning tool) from the policy cameras."""

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

from bench.runtime import RoboCasaRuntimeFactory  # noqa: E402
from bench.tasks import TaskCatalog, resolve_builtin_task_plugin  # noqa: E402
from bench.tasks.contracts import EpisodeRequest  # noqa: E402

CAMERAS = (
    "robot0_agentview_left",
    "robot0_agentview_right",
    "robot0_eye_in_hand",
)


def _render_free_camera(
    env: Any,
    output_dir: Path,
    *,
    label: str,
    lookat: list[float],
    distance: float,
    azimuth: float,
    elevation: float,
    width: int,
    height: int,
) -> Path:
    """Render a free MuJoCo camera aimed at a world point (not a fixed policy camera)."""

    import mujoco

    model = env.sim.model._model
    data = env.sim.data._data
    renderer = mujoco.Renderer(model, width=width, height=height)
    try:
        camera = mujoco.MjvCamera()
        camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        camera.lookat[:] = lookat
        camera.distance = distance
        camera.azimuth = azimuth
        camera.elevation = elevation
        renderer.update_scene(data, camera=camera)
        rgb = renderer.render().copy()
    finally:
        renderer.close()
    path = output_dir / f"wx01_{label}.png"
    imageio.imwrite(path, rgb[::-1])
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=720)
    args = parser.parse_args()

    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    from bench.env.robocasa_runtime import bootstrap_local_robocasa_sources

    bootstrap_local_robocasa_sources()
    from bench.tasks.wx01.environment import (  # noqa: E402
        BLOCKING_OBJECT_XML,
        CLEANING_TOOL_XML,
    )

    episode = TaskCatalog.resolve(EpisodeRequest(task_id="WX01", seed=args.seed))
    api = resolve_builtin_task_plugin("WX01").load_runtime_api()
    runtime = RoboCasaRuntimeFactory(api).create(episode, execute=False, expert=True)
    try:
        env = runtime.env
        env.sim.forward()
        args.output_dir.mkdir(parents=True, exist_ok=True)

        frames: list[Image.Image] = []
        images: dict[str, str] = {}
        for camera in CAMERAS:
            rgb = np.asarray(
                env.sim.render(
                    camera_name=camera,
                    width=args.width,
                    height=args.height,
                    depth=False,
                ),
                dtype=np.uint8,
            )[::-1]
            path = args.output_dir / f"wx01_{camera}.png"
            imageio.imwrite(path, rgb)
            images[camera] = str(path.resolve())
            panel = Image.fromarray(rgb)
            ImageDraw.Draw(panel).rectangle((0, 0, args.width, 36), fill=(20, 20, 20))
            ImageDraw.Draw(panel).text((12, 10), camera, fill=(255, 255, 255))
            frames.append(panel)

        sheet = Image.new("RGB", (args.width * len(frames), args.height))
        for index, frame in enumerate(frames):
            sheet.paste(frame, (index * args.width, 0))
        collage = args.output_dir / "wx01_robot_views.png"
        sheet.save(collage)

        object_poses = {
            name: env.get_object_world_position(name).tolist()
            for name in ("blocking_object", "cleaning_tool")
        }
        free_camera = {
            "cup_focus": str(
                _render_free_camera(
                    env,
                    args.output_dir,
                    label="cup_focus",
                    lookat=object_poses["blocking_object"],
                    distance=0.55,
                    azimuth=-60.0,
                    elevation=-25.0,
                    width=args.width,
                    height=args.height,
                )
            ),
            "scene_overview": str(
                _render_free_camera(
                    env,
                    args.output_dir,
                    label="scene_overview",
                    lookat=[2.4, -4.3, 1.2],
                    distance=3.0,
                    azimuth=-90.0,
                    elevation=-45.0,
                    width=args.width,
                    height=args.height,
                )
            ),
        }
        print(
            json.dumps(
                {
                    "collage": str(collage.resolve()),
                    "images": images,
                    "free_camera": free_camera,
                    "object_poses_world": object_poses,
                    "blocking_object_xml": str(BLOCKING_OBJECT_XML),
                    "cleaning_tool_xml": str(CLEANING_TOOL_XML),
                    "scene": {
                        "layout_id": int(env.layout_id),
                        "style_id": int(env.style_id),
                        "seed": int(env.seed),
                    },
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
