#!/usr/bin/env python3
"""Render WA-02's installed hot/cold sensor faucets from policy cameras."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bench.runtime import RoboCasaRuntimeFactory  # noqa: E402
from bench.tasks import TaskCatalog, resolve_builtin_task_plugin  # noqa: E402
from bench.tasks.contracts import EpisodeRequest  # noqa: E402
from bench.tasks.wa02.sensor_faucet_asset import (  # noqa: E402
    wa02_sensor_faucet_body_name,
    wa02_sensor_faucet_stream_name,
)


CAMERAS = ("robot0_agentview_left", "robot0_agentview_right")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    args = parser.parse_args()

    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    episode = TaskCatalog.resolve(
        EpisodeRequest(task_id="WA-02", seed=args.seed, allow_partial_calibration=True)
    )
    api = resolve_builtin_task_plugin("WA-02").load_runtime_api()
    runtime = RoboCasaRuntimeFactory(api).create(episode, execute=False, expert=True)
    try:
        env = runtime.env
        env.sim.forward()
        args.output_dir.mkdir(parents=True, exist_ok=True)
        images: dict[str, str] = {}
        for camera in CAMERAS:
            rgb = env.sim.render(
                camera_name=camera,
                width=args.width,
                height=args.height,
                depth=False,
            )
            path = args.output_dir / f"{camera}.png"
            imageio.imwrite(path, np.asarray(rgb)[::-1])
            images[camera] = str(path.resolve())

        faucets: dict[str, dict[str, object]] = {}
        for temperature in ("hot", "cold"):
            body_name = wa02_sensor_faucet_body_name(temperature)
            stream_name = wa02_sensor_faucet_stream_name(temperature)
            body_id = int(env.sim.model.body_name2id(body_name))
            site_id = int(env.sim.model.site_name2id(stream_name))
            faucets[temperature] = {
                "body_name": body_name,
                "body_world_xyz": env.sim.data.body_xpos[body_id].tolist(),
                "stream_name": stream_name,
                "stream_world_xyz": env.sim.data.site_xpos[site_id].tolist(),
                "stream_alpha": float(env.sim.model.site_rgba[site_id][3]),
            }
        legacy_control_geoms: list[dict[str, object]] = []
        sink_prefix = str(env.sink.naming_prefix)
        for geom_id, name in enumerate(env.sim.model.geom_names):
            if not str(name).startswith(sink_prefix):
                continue
            body_id = int(env.sim.model.geom_bodyid[geom_id])
            body_name = str(env.sim.model.body_id2name(body_id))
            if body_name not in {f"{sink_prefix}handle", f"{sink_prefix}spout"}:
                continue
            legacy_control_geoms.append(
                {
                    "name": str(name),
                    "body": body_name,
                    "alpha": float(env.sim.model.geom_rgba[geom_id][3]),
                }
            )
        print(
            json.dumps(
                {
                    "images": images,
                    "faucets": faucets,
                    "legacy_control_geom_count": len(legacy_control_geoms),
                    "visible_legacy_control_geoms": [
                        geom["name"]
                        for geom in legacy_control_geoms
                        if float(geom["alpha"]) > 0.0
                    ],
                },
                indent=2,
            )
        )
    finally:
        runtime.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
