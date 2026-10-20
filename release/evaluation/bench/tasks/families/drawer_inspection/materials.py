"""Shared appearance-only finishes for task-owned tabletop drawers."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from collections.abc import Sequence
from pathlib import Path
from typing import Any


WOOD_TEXTURE_NAMES = (
    "bamboo", "dark_wood_parquet", "dark_wood_planks", "dark_wood_planks_2",
    "gray_wood_grain", "gray_wood_planks", "greenheart_wood_grain",
    "light_wood_parquet", "light_wood_planks", "light_wood_planks_2",
    "light_wood_planks_long", "red_wood_planks", "varnished_wood_planks",
    "walnut_wood_grain", "warm_wood_grain", "warm_wood_grain_2",
    "warm_wood_parquet", "warm_wood_planks", "wood_grain_1", "wood_grain_2",
    "wood_grain_3", "wood_planks", "wood_planks_2", "wood_planks_3",
    "wood_planks_wall",
)
PAINT_TEXTURE_NAMES = (
    "black", "blue_gray", "cream", "dark_blue", "dark_gray", "gray", "green",
    "light_blue", "light_gray", "light_green", "lighter_gray", "plaster",
    "warm_white", "warm_white_2", "white",
)
METAL_TEXTURE_NAMES = ("brass", "bright_metal", "brighter_metal", "metal")
DRAWER_MATERIAL_TEXTURES = {
    "original": None,
    **{f"wood/{name}": f"wood/{name}.png" for name in WOOD_TEXTURE_NAMES},
    **{f"flat/{name}": f"flat/{name}.png" for name in PAINT_TEXTURE_NAMES},
    **{f"metals/{name}": f"metals/{name}.png" for name in METAL_TEXTURE_NAMES},
    "steel-brushed": "steel-brushed.png",
}


def normalize_drawer_material_candidates(value: Any) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError("drawer_material_candidates must be a non-empty sequence")
    candidates = tuple(str(item) for item in value)
    if not candidates or len(set(candidates)) != len(candidates):
        raise ValueError("drawer_material_candidates must be non-empty and unique")
    unknown = sorted(set(candidates) - DRAWER_MATERIAL_TEXTURES.keys())
    if unknown:
        raise ValueError("unknown drawer materials: " + ", ".join(unknown))
    return candidates


def apply_drawer_material(
    root: ET.Element,
    *,
    fixture_name: str,
    material_id: str,
    texture_root: Path,
) -> None:
    """Apply one finish to the cabinet and drawer without touching the handle."""

    if material_id not in DRAWER_MATERIAL_TEXTURES:
        raise ValueError(f"unknown drawer material: {material_id!r}")
    relative_path = DRAWER_MATERIAL_TEXTURES[material_id]
    if relative_path is None:
        return
    texture_path = (Path(texture_root) / relative_path).resolve()
    if not texture_path.is_file():
        raise FileNotFoundError(f"missing drawer texture: {texture_path}")
    asset = root.find("asset")
    if asset is None:
        raise ValueError("drawer material requires an asset section")
    materials = []
    for surface in ("cabinet_body", "drawer_front", "drawer_inside"):
        name = f"{fixture_name}_{surface}"
        material = asset.find(f"material[@name='{name}']")
        if material is None:
            raise ValueError(f"missing drawer material: {name}")
        materials.append(material)
    texture_name = f"{fixture_name}_finish_texture"
    texture = asset.find(f"texture[@name='{texture_name}']")
    if texture is None:
        texture = ET.SubElement(asset, "texture", name=texture_name)
    texture.attrib.update(type="cube", file=str(texture_path))
    is_metal = material_id.startswith("metals/") or material_id == "steel-brushed"
    for material in materials:
        material.attrib.update(
            texture=texture_name,
            rgba="1 1 1 1",
            texuniform="true",
            texrepeat="1 1",
            specular="0.5" if is_metal else "0.1",
            shininess="0.3" if is_metal else "0.05",
        )


__all__ = [
    "DRAWER_MATERIAL_TEXTURES", "METAL_TEXTURE_NAMES", "PAINT_TEXTURE_NAMES",
    "WOOD_TEXTURE_NAMES", "apply_drawer_material",
    "normalize_drawer_material_candidates",
]
