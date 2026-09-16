"""Seeded appearance-only cabinet finishes for WR-01."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any
import xml.etree.ElementTree as ET

import numpy as np


CABINET_MATERIAL_TEXTURES: dict[str, str | None] = {
    "original": None,
    "light_wood": "wood/light_wood_planks.png",
    "walnut": "wood/walnut_wood_grain.png",
    "gray_wood": "wood/gray_wood_planks.png",
    "warm_white": "flat/warm_white.png",
}
_CABINET_MATERIAL_SALT = 0x5752314D


def normalize_cabinet_material_candidates(value: Any) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError("WR-01 cabinet_material_candidates must be a non-empty sequence")
    candidates = tuple(str(item) for item in value)
    if not candidates or len(set(candidates)) != len(candidates):
        raise ValueError("WR-01 cabinet material candidates must be non-empty and unique")
    unknown = set(candidates) - CABINET_MATERIAL_TEXTURES.keys()
    if unknown:
        raise ValueError(
            "unknown WR-01 cabinet material candidates: " + ", ".join(sorted(unknown))
        )
    return candidates


def sample_cabinet_material(seed: int, candidates: Sequence[str]) -> str:
    normalized = normalize_cabinet_material_candidates(candidates)
    rng = np.random.default_rng(
        np.random.SeedSequence([int(seed), _CABINET_MATERIAL_SALT])
    )
    return normalized[int(rng.integers(0, len(normalized)))]


def apply_cabinet_material(
    root: ET.Element,
    *,
    fixture_names: Sequence[str],
    material_id: str,
    texture_root: Path,
) -> None:
    """Apply one finish to cabinet wood while preserving handles and physics."""

    if material_id not in CABINET_MATERIAL_TEXTURES:
        raise ValueError(f"unknown WR-01 cabinet material: {material_id!r}")
    relative_path = CABINET_MATERIAL_TEXTURES[material_id]
    if relative_path is None:
        return
    texture_path = (Path(texture_root) / relative_path).resolve()
    if not texture_path.is_file():
        raise FileNotFoundError(f"missing WR-01 cabinet texture: {texture_path}")
    asset = root.find("asset")
    if asset is None:
        raise ValueError("WR-01 cabinet material requires an asset section")
    for fixture_name in fixture_names:
        material_name = f"{fixture_name}_wr01_wood"
        material = asset.find(f"material[@name='{material_name}']")
        if material is None:
            raise ValueError(f"missing WR-01 cabinet material: {material_name}")
        texture_name = f"{fixture_name}_finish_texture"
        texture = asset.find(f"texture[@name='{texture_name}']")
        if texture is None:
            texture = ET.SubElement(asset, "texture", name=texture_name)
        texture.attrib.update(type="cube", file=str(texture_path))
        material.attrib.update(
            texture=texture_name,
            rgba="1 1 1 1",
            texrepeat="3 3",
            texuniform="false",
            reflectance="0.15",
            shininess="0.12",
        )


__all__ = [
    "CABINET_MATERIAL_TEXTURES",
    "apply_cabinet_material",
    "normalize_cabinet_material_candidates",
    "sample_cabinet_material",
]
