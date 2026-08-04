#!/usr/bin/env python3
"""Render RoboCasa sponge or bar-soap assets for visual comparison."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[2]
for source_root in (ROOT / "robosuite", ROOT / "robocasa"):
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))

from robosuite.models import MujocoWorldBase  # noqa: E402
from robosuite.models.arenas import TableArena  # noqa: E402
from robosuite.models.objects import MujocoXMLObject  # noqa: E402
from robosuite.utils.mjcf_utils import array_to_string  # noqa: E402


OBJECT_ASSET_ROOT = ROOT / "robocasa/robocasa/models/assets/objects"
ASSET_CATEGORIES = {
    "sponge": {
        "aigen": {
            "directory": "aigen_objs",
            "display_name": "AIGen",
            "scale": 1.20,
            "indices": tuple(range(10)),
        },
        "objaverse": {
            "directory": "objaverse",
            "display_name": "Objaverse",
            "scale": 0.90,
            "indices": (0, 1, 2, 3, 4, 5, 6, 8),
        },
    },
    "bar_soap": {
        "aigen": {
            "directory": "aigen_objs",
            "display_name": "AIGen",
            "scale": (1.25, 1.25, 1.40),
            "indices": tuple(range(8)),
        },
        "objaverse": {
            "directory": "objaverse",
            "display_name": "Objaverse",
            "scale": (0.95, 0.95, 1.05),
            "indices": (0, 1, 3, 4),
        },
    },
}


def _sanitized_model(asset_file: Path, temp_dir: Path) -> Path:
    """Remove stale texture references and make remaining file paths absolute."""
    tree = ET.parse(asset_file)
    root = tree.getroot()
    asset = root.find("asset")
    if asset is None:
        raise ValueError(f"missing <asset> in {asset_file}")

    removed_textures: set[str] = set()
    for texture in list(asset.findall("texture")):
        file_value = texture.get("file")
        if file_value and Path(file_value).is_absolute() and not Path(file_value).exists():
            if texture.get("name"):
                removed_textures.add(texture.get("name", ""))
            asset.remove(texture)

    for material in list(asset.findall("material")):
        if material.get("texture") in removed_textures:
            asset.remove(material)

    for element in asset:
        file_value = element.get("file")
        if file_value and not Path(file_value).is_absolute():
            element.set("file", str((asset_file.parent / file_value).resolve()))

    source_name = asset_file.parents[2].name
    output = temp_dir / f"{source_name}_{asset_file.parent.name}_model.xml"
    tree.write(output, encoding="unicode")
    return output


def _bbox_size(asset_file: Path) -> np.ndarray:
    root = ET.parse(asset_file).getroot()
    bbox = root.find(".//geom[@name='reg_bbox']")
    if bbox is None or not bbox.get("size"):
        raise ValueError(f"missing reg_bbox size in {asset_file}")
    return 2.0 * np.fromstring(bbox.get("size", ""), sep=" ")


def _render(
    *,
    category: str,
    source: str,
    candidate: int,
    width: int,
    height: int,
    temp_dir: Path,
) -> tuple[np.ndarray, dict[str, object]]:
    source_config = ASSET_CATEGORIES[category][source]
    source_directory = str(source_config["directory"])
    scale_value = np.asarray(source_config["scale"], dtype=float)
    render_scale = np.broadcast_to(scale_value, (3,)).copy()
    asset_file = (
        OBJECT_ASSET_ROOT
        / source_directory
        / category
        / f"{category}_{candidate}"
        / "model.xml"
    )
    bbox = _bbox_size(asset_file)
    sanitized_file = _sanitized_model(asset_file, temp_dir)

    world = MujocoWorldBase()
    world.merge(
        TableArena(
            table_full_size=(0.48, 0.48, 0.05),
            table_offset=(0.0, 0.0, 0.80),
            has_legs=False,
        )
    )
    obj = MujocoXMLObject(
        str(sanitized_file),
        name=f"{source}_{category}_{candidate}",
        joints="default",
        obj_type="all",
        duplicate_collision_geoms=False,
        scale=tuple(render_scale),
    )
    body = obj.get_obj()
    body.set("pos", array_to_string((0.0, 0.0, 0.90)))
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
    planar_ratio = float(max(bbox[:2]) / min(bbox[:2]))
    return rgb, {
        "source": source,
        "source_display_name": source_config["display_name"],
        "category": category,
        "candidate_index": candidate,
        "candidate": f"{source_directory}/{category}/{category}_{candidate}",
        "asset_file": str(asset_file.relative_to(ROOT)),
        "render_scale": render_scale.astype(float).tolist(),
        "native_bbox_m": bbox.astype(float).tolist(),
        "configured_bbox_m": (bbox * render_scale).astype(float).tolist(),
        "planar_aspect_ratio": planar_ratio,
        "settled_position": data.xpos[body_id].astype(float).tolist(),
        "settled_quaternion_wxyz": data.xquat[body_id].astype(float).tolist(),
    }


def _contact_sheet(
    images: dict[str, np.ndarray], metadata: dict[str, dict[str, object]]
) -> Image.Image:
    tile_height, tile_width = next(iter(images.values())).shape[:2]
    columns = 4
    rows = (len(images) + columns - 1) // columns
    label_height = 66
    sheet = Image.new(
        "RGB", (columns * tile_width, rows * (tile_height + label_height)), "white"
    )
    draw = ImageDraw.Draw(sheet)
    title_font = ImageFont.load_default(size=20)
    detail_font = ImageFont.load_default(size=16)

    for index, key in enumerate(images):
        x = (index % columns) * tile_width
        y = (index // columns) * (tile_height + label_height)
        sheet.paste(Image.fromarray(images[key]), (x, y + label_height))
        details = metadata[key]
        bbox_cm = 100.0 * np.asarray(details["configured_bbox_m"])
        ratio = details["planar_aspect_ratio"]
        display_name = details["source_display_name"]
        candidate = details["candidate_index"]
        draw.text(
            (x + 14, y + 8),
            f"{display_name} {details['category']}_{candidate}",
            fill=(25, 25, 25),
            font=title_font,
        )
        draw.text(
            (x + 14, y + 37),
            f"scaled {bbox_cm[0]:.1f} x {bbox_cm[1]:.1f} x {bbox_cm[2]:.1f} cm | XY ratio {ratio:.2f}",
            fill=(70, 70, 70),
            font=detail_font,
        )
    return sheet


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--category",
        choices=tuple(ASSET_CATEGORIES),
        default="sponge",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--aigen-candidates",
        type=int,
        nargs="*",
        default=None,
        help="AIGen asset indices to render (default: all in category)",
    )
    parser.add_argument(
        "--objaverse-candidates",
        type=int,
        nargs="*",
        default=None,
        help="Objaverse asset indices to render (default: all in category)",
    )
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=420)
    args = parser.parse_args()
    category_config = ASSET_CATEGORIES[args.category]
    if args.output_dir is None:
        category_plural = "sponges" if args.category == "sponge" else "bar_soaps"
        args.output_dir = Path(f"outputs/asset_previews/wr02_all_{category_plural}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    selections = {
        "aigen": (
            args.aigen_candidates
            if args.aigen_candidates is not None
            else category_config["aigen"]["indices"]
        ),
        "objaverse": (
            args.objaverse_candidates
            if args.objaverse_candidates is not None
            else category_config["objaverse"]["indices"]
        ),
    }
    images: dict[str, np.ndarray] = {}
    metadata: dict[str, dict[str, object]] = {}
    with tempfile.TemporaryDirectory(prefix=f"robocasa_{args.category}_") as temp_dir:
        for source, candidates in selections.items():
            for candidate in candidates:
                key = f"{source}_{args.category}_{candidate}"
                rgb, details = _render(
                    category=args.category,
                    source=source,
                    candidate=candidate,
                    width=args.width,
                    height=args.height,
                    temp_dir=Path(temp_dir),
                )
                images[key] = rgb
                metadata[key] = details
                imageio.imwrite(args.output_dir / f"{key}_flat.png", rgb)

    sheet_path = args.output_dir / f"all_{len(images)}_{args.category}_comparison.png"
    _contact_sheet(images, metadata).save(sheet_path)
    (args.output_dir / "metadata.json").write_text(
        json.dumps(list(metadata.values()), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(sheet_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
