"""Seeded appearance-only cabinet finishes for TM02."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any
import xml.etree.ElementTree as ET

import numpy as np

from bench.tasks.wr01.cabinet_materials import CABINET_MATERIAL_TEXTURES

_SALT = 0x544D30324D


def normalize_cabinet_material_candidates(value: Any) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError("TM02 cabinet_material_candidates must be a non-empty sequence")
    result = tuple(str(item) for item in value)
    if not result or len(set(result)) != len(result):
        raise ValueError("TM02 cabinet material candidates must be non-empty and unique")
    unknown = set(result) - set(CABINET_MATERIAL_TEXTURES)
    if unknown:
        raise ValueError("unknown TM02 cabinet materials: " + ", ".join(sorted(unknown)))
    return result


def sample_cabinet_material(seed: int, candidates: Sequence[str]) -> str:
    values = normalize_cabinet_material_candidates(candidates)
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), _SALT]))
    return values[int(rng.integers(0, len(values)))]


def apply_cabinet_material(root: ET.Element, *, material_id: str, texture_root: Path) -> None:
    if material_id not in CABINET_MATERIAL_TEXTURES:
        raise ValueError(f"unknown TM02 cabinet material: {material_id!r}")
    relative = CABINET_MATERIAL_TEXTURES[material_id]
    if relative is None:
        return
    texture_path = (Path(texture_root) / relative).resolve()
    if not texture_path.is_file():
        raise FileNotFoundError(f"missing TM02 cabinet texture: {texture_path}")
    asset = root.find("asset")
    if asset is None:
        raise ValueError("TM02 cabinet material requires an asset section")
    material = asset.find("material[@name='tm02_wood']")
    if material is None:
        # RoboCasa prefixes fixture-local asset names when merging XML.
        material = next(
            (item for item in asset.findall("material")
             if str(item.get("name", "")).endswith("_tm02_wood")),
            None,
        )
    if material is None:
        raise ValueError("missing TM02 cabinet material: tm02_wood")
    fixture_prefix = str(material.get("name", "tm02")).rsplit("_tm02_wood", 1)[0]
    texture_name = f"{fixture_prefix}_finish_texture"
    texture = asset.find(f"texture[@name='{texture_name}']")
    if texture is None:
        texture = ET.SubElement(asset, "texture", name=texture_name)
    texture.attrib.update(type="cube", file=str(texture_path))
    material.attrib.update(texture=texture_name, rgba="1 1 1 1", texrepeat="3 3", texuniform="false", reflectance="0.15", shininess="0.12")


__all__ = ["CABINET_MATERIAL_TEXTURES", "apply_cabinet_material", "normalize_cabinet_material_candidates", "sample_cabinet_material"]
