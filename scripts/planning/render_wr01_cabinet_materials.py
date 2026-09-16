#!/usr/bin/env python3
"""Render all seeded WR-01 cabinet finishes from one fixed camera."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

from PIL import Image, ImageDraw
import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bench.runtime import RoboCasaRuntimeFactory  # noqa: E402
from bench.tasks import TaskCatalog, resolve_builtin_task_plugin  # noqa: E402
from bench.tasks.contracts import EpisodeRequest  # noqa: E402
from bench.tasks.wr01.cabinet_materials import CABINET_MATERIAL_TEXTURES  # noqa: E402
from bench.tasks.wr01.config import DEFAULT_TASK_CONFIG_FILE  # noqa: E402
from scripts.planning.render_wr_scene import _render_free_camera  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    config = yaml.safe_load(DEFAULT_TASK_CONFIG_FILE.read_text(encoding="utf-8"))
    config["resources"] = {
        key: str((DEFAULT_TASK_CONFIG_FILE.parent / value).resolve())
        for key, value in config["resources"].items()
    }
    api = resolve_builtin_task_plugin("WR-01").load_runtime_api()
    frames = []
    records = []
    for material in CABINET_MATERIAL_TEXTURES:
        config["episode"]["cabinet_material_candidates"] = [material]
        config_path = output / f"{material}_preview.yaml"
        config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        episode = TaskCatalog.resolve(
            EpisodeRequest(task_id="WR-01", seed=args.seed, task_config_file=config_path)
        )
        runtime = RoboCasaRuntimeFactory(api).create(episode, execute=False, expert=True)
        try:
            env = runtime.env
            env.sim.forward()
            path = _render_free_camera(
                env,
                output,
                label=f"wr01_{material}",
                lookat=[3.10, -0.72, 1.30],
                distance=1.65,
                azimuth=90.0,
                elevation=-8.0,
                width=640,
                height=480,
            )
            panel = Image.open(path).convert("RGB")
            draw = ImageDraw.Draw(panel)
            draw.rectangle((0, 0, 640, 34), fill=(20, 20, 20))
            draw.text((12, 10), material, fill=(255, 255, 255))
            frames.append(panel)
            records.append(
                {
                    "material": material,
                    "image": str(path.resolve()),
                    "episode_material": episode.facts.get("cabinet_material_id"),
                }
            )
            print(f"Rendered {material}", flush=True)
        finally:
            runtime.close()
    collage = Image.new("RGB", (640 * len(frames), 480), "white")
    for index, frame in enumerate(frames):
        collage.paste(frame, (index * 640, 0))
    collage_path = output / "wr01_cabinet_materials.png"
    collage.save(collage_path)
    (output / "manifest.json").write_text(
        json.dumps({"seed": args.seed, "variants": records}, indent=2), encoding="utf-8"
    )
    print(f"Saved comparison to {collage_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
