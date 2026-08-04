#!/usr/bin/env python3
"""Render RoboCasa bar-soap and sponge assets in flat and edge poses."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[2]
for source_root in (ROOT / "robosuite", ROOT / "robocasa"):
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))

import mujoco  # noqa: E402
from robosuite.models import MujocoWorldBase  # noqa: E402
from robosuite.models.arenas import TableArena  # noqa: E402
from robosuite.models.objects import MujocoXMLObject  # noqa: E402
from robosuite.utils.mjcf_utils import array_to_string  # noqa: E402


ASSETS = {
    "bar_soap": (
        ROOT
        / "robocasa/robocasa/models/assets/objects/objaverse/bar_soap"
        / "bar_soap_0/model.xml",
        (0.95, 0.95, 1.05),
    ),
    "sponge": (
        ROOT
        / "robocasa/robocasa/models/assets/objects/objaverse/sponge"
        / "sponge_0/model.xml",
        (0.90, 0.90, 0.90),
    ),
}


def _render_asset(
    *,
    asset_name: str,
    pose_name: str,
    width: int,
    height: int,
) -> tuple[np.ndarray, dict[str, object]]:
    asset_file, scale = ASSETS[asset_name]
    world = MujocoWorldBase()
    world.merge(
        TableArena(
            table_full_size=(0.48, 0.48, 0.05),
            table_offset=(0.0, 0.0, 0.80),
            has_legs=False,
        )
    )

    obj = MujocoXMLObject(
        str(asset_file),
        name=asset_name,
        joints="default",
        obj_type="all",
        duplicate_collision_geoms=False,
        scale=scale,
    )
    body = obj.get_obj()
    body.set("pos", array_to_string((0.0, 0.0, 0.90)))
    if pose_name == "edge":
        body.set("quat", array_to_string((np.sqrt(0.5), np.sqrt(0.5), 0.0, 0.0)))
    elif pose_name != "flat":
        raise ValueError(f"unknown pose {pose_name!r}")
    world.merge_assets(obj)
    world.worldbody.append(body)

    model = world.get_model(mode="mujoco")
    data = mujoco.MjData(model)
    for _ in range(800):
        mujoco.mj_step(model, data)

    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = (0.0, 0.0, 0.80)
    camera.distance = 0.42
    camera.azimuth = 135.0
    camera.elevation = -32.0

    renderer = mujoco.Renderer(model, height=height, width=width)
    try:
        renderer.update_scene(data, camera=camera)
        rgb = renderer.render().copy()
    finally:
        renderer.close()

    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, obj.root_body)
    return rgb, {
        "asset": asset_name,
        "asset_file": str(asset_file.relative_to(ROOT)),
        "pose": pose_name,
        "scale": list(scale),
        "settled_position": data.xpos[body_id].astype(float).tolist(),
        "settled_quaternion_wxyz": data.xquat[body_id].astype(float).tolist(),
    }


def _make_contact_sheet(images: dict[str, np.ndarray]) -> Image.Image:
    labels = (
        ("bar_soap_flat", "BAR SOAP - flat"),
        ("bar_soap_edge", "BAR SOAP - edge"),
        ("sponge_flat", "SPONGE - flat"),
        ("sponge_edge", "SPONGE - edge"),
    )
    tile_height, tile_width = next(iter(images.values())).shape[:2]
    label_height = 42
    sheet = Image.new(
        "RGB", (tile_width * 2, (tile_height + label_height) * 2), "white"
    )
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default(size=20)
    for index, (key, label) in enumerate(labels):
        x = (index % 2) * tile_width
        y = (index // 2) * (tile_height + label_height)
        sheet.paste(Image.fromarray(images[key]), (x, y + label_height))
        draw.text((x + 14, y + 10), label, fill=(25, 25, 25), font=font)
    return sheet


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/asset_previews/wr02_soap_sponge"),
    )
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=420)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    images: dict[str, np.ndarray] = {}
    metadata: list[dict[str, object]] = []
    for asset_name in ASSETS:
        for pose_name in ("flat", "edge"):
            key = f"{asset_name}_{pose_name}"
            rgb, details = _render_asset(
                asset_name=asset_name,
                pose_name=pose_name,
                width=args.width,
                height=args.height,
            )
            images[key] = rgb
            metadata.append(details)
            imageio.imwrite(args.output_dir / f"{key}.png", rgb)

    sheet_path = args.output_dir / "soap_sponge_pose_comparison.png"
    _make_contact_sheet(images).save(sheet_path)
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(sheet_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
