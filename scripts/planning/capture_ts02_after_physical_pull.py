#!/usr/bin/env python3
"""Capture TS02 policy cameras after a real expert drawer pull."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bench.execution.contracts import ExpertPlan  # noqa: E402
from bench.runtime import RoboCasaRuntimeFactory  # noqa: E402
from bench.tasks import TaskCatalog, resolve_builtin_task_plugin  # noqa: E402
from bench.tasks.contracts import EpisodeRequest, ExpertPlanInputs  # noqa: E402
from bench.tasks.ts02.target_object import TARGET_OBJECT_GEOM_NAME  # noqa: E402


CAMERAS = (
    "robot0_agentview_left",
    "robot0_agentview_right",
    "robot0_eye_in_hand",
)


def _render_rgb(env: Any, camera: str, *, width: int, height: int) -> np.ndarray:
    return np.asarray(
        env.sim.render(
            camera_name=camera,
            width=int(width),
            height=int(height),
            depth=False,
        ),
        dtype=np.uint8,
    )[::-1].copy()


def _target_mask(
    env: Any,
    camera: str,
    *,
    target_geom_id: int,
    width: int,
    height: int,
) -> np.ndarray:
    segmentation = np.asarray(
        env.sim.render(
            camera_name=camera,
            width=int(width),
            height=int(height),
            depth=False,
            segmentation=True,
        )
    )[::-1]
    if segmentation.ndim != 3 or segmentation.shape[-1] != 2:
        raise RuntimeError(
            f"unexpected segmentation shape for {camera}: {segmentation.shape}"
        )
    return np.asarray(segmentation[..., 1] == int(target_geom_id), dtype=bool)


def _bbox(mask: np.ndarray) -> list[int] | None:
    ys, xs = np.nonzero(mask)
    if not len(xs):
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]


def _font(size: int) -> ImageFont.ImageFont:
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf",
    ):
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def _compose_diagnostic(
    frames: dict[str, np.ndarray],
    visibility: dict[str, dict[str, Any]],
    *,
    output_path: Path,
) -> None:
    first = frames[CAMERAS[0]]
    height, width = first.shape[:2]
    header_height = 64
    gap = 12
    sheet = Image.new(
        "RGB",
        (width * len(CAMERAS) + gap * (len(CAMERAS) - 1), height + header_height),
        (20, 20, 20),
    )
    draw = ImageDraw.Draw(sheet)
    title_font = _font(24)
    info_font = _font(18)
    for index, camera in enumerate(CAMERAS):
        x0 = index * (width + gap)
        panel = Image.fromarray(frames[camera])
        bbox = visibility[camera]["bbox_xyxy"]
        if bbox is not None:
            panel_draw = ImageDraw.Draw(panel)
            x1, y1, x2, y2 = bbox
            margin = 8
            panel_draw.rectangle(
                (
                    max(0, x1 - margin),
                    max(0, y1 - margin),
                    min(width - 1, x2 + margin),
                    min(height - 1, y2 + margin),
                ),
                outline=(0, 255, 255),
                width=4,
            )
        sheet.paste(panel, (x0, header_height))
        draw.text((x0 + 10, 7), camera, font=title_font, fill=(245, 245, 245))
        visible = bool(visibility[camera]["target_visible"])
        color = (80, 255, 120) if visible else (255, 100, 100)
        label = (
            f"red cube visible: {'YES' if visible else 'NO'}"
            f"  |  exact pixels: {visibility[camera]['target_pixel_count']}"
        )
        draw.text((x0 + 10, 37), label, font=info_font, fill=color)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=720)
    args = parser.parse_args()

    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

    print(f"Resolving TS02 seed {args.seed}...", flush=True)
    episode = TaskCatalog.resolve(EpisodeRequest(task_id="TS02", seed=args.seed))
    api = resolve_builtin_task_plugin("TS02").load_runtime_api()
    descriptor = api.build_descriptor(episode, True)
    production_plan = TaskCatalog.get("TS02").build_expert_plan(
        ExpertPlanInputs.from_descriptor(descriptor)
    )
    if len(production_plan.stages) < 2:
        raise RuntimeError("TS02 production plan does not contain navigate + open")
    target_fixture = str(episode.facts["target_fixture"])
    first_fixture = str(production_plan.metadata["fixture_order"][0])
    if first_fixture != target_fixture:
        raise RuntimeError(
            f"seed {args.seed} first fixture {first_fixture!r} is not target "
            f"{target_fixture!r}; choose a seed that exposes the target after one pull"
        )
    partial_plan = ExpertPlan(
        task_id="TS02",
        stages=tuple(production_plan.stages[:2]),
        version=production_plan.version,
        metadata={"purpose": "capture_after_real_physical_pull"},
    )

    print(
        f"Creating RoboCasa runtime; target and first drawer are {target_fixture}...",
        flush=True,
    )
    runtime = RoboCasaRuntimeFactory(api).create_from_descriptor(
        descriptor,
        expert=True,
    )
    try:
        runner = runtime.initialized.runner
        if runner is None:
            raise RuntimeError("TS02 runtime did not expose an expert runner")
        print("Executing real navigation and contact drawer pull...", flush=True)
        result = runner.run(partial_plan)
        stage_results = [
            {
                "index": int(stage.index),
                "kind": stage.kind.value,
                "success": bool(stage.success),
                "failure_code": stage.failure_code,
                "details": dict(stage.details),
            }
            for stage in result.stages
        ]
        open_fraction = float(runtime.env.slot_open_fraction(target_fixture))
        print(
            f"Physical pull finished: success={result.success}, "
            f"open_fraction={open_fraction:.6f}",
            flush=True,
        )
        if not result.success:
            raise RuntimeError(f"physical expert stages failed: {stage_results}")
        if open_fraction < float(runtime.env.open_threshold):
            raise RuntimeError(
                f"drawer fraction {open_fraction:.6f} is below open threshold "
                f"{float(runtime.env.open_threshold):.6f}"
            )

        runtime.env.sim.forward()
        target_geom_id = int(
            runtime.env.sim.model.geom_name2id(TARGET_OBJECT_GEOM_NAME)
        )
        args.output_dir.mkdir(parents=True, exist_ok=True)
        frames: dict[str, np.ndarray] = {}
        visibility: dict[str, dict[str, Any]] = {}
        for camera in CAMERAS:
            frame = _render_rgb(
                runtime.env,
                camera,
                width=args.width,
                height=args.height,
            )
            mask = _target_mask(
                runtime.env,
                camera,
                target_geom_id=target_geom_id,
                width=args.width,
                height=args.height,
            )
            frames[camera] = frame
            pixel_count = int(mask.sum())
            visibility[camera] = {
                "target_visible": pixel_count > 0,
                "target_pixel_count": pixel_count,
                "bbox_xyxy": _bbox(mask),
            }
            imageio.imwrite(args.output_dir / f"ts02_{camera}_after_pull.png", frame)
            imageio.imwrite(
                args.output_dir / f"ts02_{camera}_target_mask.png",
                np.asarray(mask, dtype=np.uint8) * 255,
            )

        collage_path = (
            args.output_dir / "ts02_robot_three_views_after_physical_pull.png"
        )
        _compose_diagnostic(frames, visibility, output_path=collage_path)
        report = {
            "seed": int(args.seed),
            "target_fixture": target_fixture,
            "executed_stages": [stage.name for stage in partial_plan.stages],
            "physical_stage_success": bool(result.success),
            "drawer_open_fraction": open_fraction,
            "drawer_open_threshold": float(runtime.env.open_threshold),
            "target_geom_id": target_geom_id,
            "visibility": visibility,
            "images": {
                camera: str(
                    (args.output_dir / f"ts02_{camera}_after_pull.png").resolve()
                )
                for camera in CAMERAS
            },
            "collage": str(collage_path.resolve()),
            "stage_results": stage_results,
        }
        report_path = args.output_dir / "ts02_after_physical_pull_report.json"
        report_path.write_text(
            json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
        )
        print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    finally:
        runtime.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
