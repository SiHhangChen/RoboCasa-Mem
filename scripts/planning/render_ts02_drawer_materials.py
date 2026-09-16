#!/usr/bin/env python3
"""Render TS02 exterior finish candidates on the drawer's actual MJCF asset."""

from __future__ import annotations

import argparse
import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bench.tasks.ts02.config import PACKAGE_ROOT, load_task_config  # noqa: E402
from bench.tasks.ts02.drawer_materials import (  # noqa: E402
    DRAWER_MATERIAL_TEXTURES,
    apply_drawer_material,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all", action="store_true", help="Include all available finishes")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    os.environ.setdefault("MUJOCO_GL", "egl")
    import mujoco

    materials = (
        tuple(DRAWER_MATERIAL_TEXTURES)
        if args.all
        else ("original", *load_task_config().episode["drawer_material_candidates"])
    )
    width, height, label_height, columns = 320, 240, 44, 4
    rows = (len(materials) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * width, rows * (height + label_height)), (242, 244, 246))
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 15)
    fixture = "ts02_preview_fixture"
    for index, material_id in enumerate(materials):
        root = ET.parse(PACKAGE_ROOT / "resources/assets/single_tabletop_drawer.xml").getroot()
        for material in root.findall("./asset/material"):
            material.set("name", f"{fixture}_{material.get('name')}")
        for geom in root.iter("geom"):
            if geom.get("material") is not None:
                geom.set("material", f"{fixture}_{geom.get('material')}")
        ET.SubElement(
            root.find("asset"),
            "texture",
            name="preview_sky",
            type="skybox",
            builtin="flat",
            rgb1="0.82 0.84 0.86",
            width="128",
            height="128",
        )
        ET.SubElement(root, "visual")
        ET.SubElement(
            root.find("visual"), "headlight", ambient="0.4 0.4 0.4", diffuse="0.7 0.7 0.7"
        )
        apply_drawer_material(
            root,
            fixture_name=fixture,
            material_id=material_id,
            texture_root=ROOT / "robocasa/robocasa/models/assets/textures",
        )
        model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
        data = mujoco.MjData(model)
        # Open the drawer so the preview verifies the front, side walls,
        # bottom, and back share one finish while the handle stays unchanged.
        if model.njnt:
            data.qpos[model.jnt_qposadr[0]] = 0.18
        mujoco.mj_forward(model, data)
        camera = mujoco.MjvCamera()
        camera.lookat[:] = [0.0, -0.12, -0.025]
        camera.distance = 0.72
        camera.azimuth = 110.0
        camera.elevation = -28.0
        with mujoco.Renderer(model, height=height, width=width) as renderer:
            renderer.update_scene(data, camera=camera)
            frame = renderer.render().copy()
            renderer.enable_segmentation_rendering()
            visible_pixels = int(np.count_nonzero(renderer.render()[..., 0] >= 0))
        if visible_pixels < width * height * 0.02:
            raise RuntimeError(f"blank TS02 material render: {material_id}")
        x = (index % columns) * width
        y = (index // columns) * (height + label_height)
        sheet.paste(Image.fromarray(frame), (x, y))
        category, _, name = material_id.partition("/")
        label = f"{category}\n{name}" if name else category
        draw.multiline_text(
            (x + 10, y + height + 4), label, font=font, fill=(35, 40, 45), spacing=1
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(args.output)
    print(f"Rendered {len(materials)} finishes: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
