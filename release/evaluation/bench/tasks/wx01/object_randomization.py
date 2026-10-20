"""WX-01 appearance profiles and repository-backed sponge assets."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np


CUP_MATERIAL_PROFILES = {
    "ceramic_white": {
        "body_rgba": (0.96, 0.96, 0.94, 1.0),
        "accent_rgba": (0.90, 0.90, 0.87, 1.0),
        "specular": 0.35,
        "shininess": 0.30,
        "reflectance": 0.02,
    },
    "ceramic_blue": {
        "body_rgba": (0.14, 0.36, 0.72, 1.0),
        "accent_rgba": (0.07, 0.20, 0.48, 1.0),
        "specular": 0.42,
        "shininess": 0.38,
        "reflectance": 0.03,
    },
    "ceramic_green": {
        "body_rgba": (0.20, 0.58, 0.36, 1.0),
        "accent_rgba": (0.08, 0.34, 0.20, 1.0),
        "specular": 0.38,
        "shininess": 0.34,
        "reflectance": 0.02,
    },
    "ceramic_yellow": {
        "body_rgba": (0.94, 0.72, 0.12, 1.0),
        "accent_rgba": (0.76, 0.46, 0.04, 1.0),
        "specular": 0.42,
        "shininess": 0.40,
        "reflectance": 0.03,
    },
    "matte_black": {
        "body_rgba": (0.055, 0.060, 0.065, 1.0),
        "accent_rgba": (0.018, 0.020, 0.022, 1.0),
        "specular": 0.10,
        "shininess": 0.06,
        "reflectance": 0.0,
    },
}
CUP_MATERIAL_IDS = tuple(CUP_MATERIAL_PROFILES)

SPONGE_ASSET_BY_TYPE = {
    sponge_type: f"objects/objaverse/sponge/{sponge_type}/model.xml"
    for sponge_type in (
        "sponge_0",
        "sponge_1",
        "sponge_2",
        "sponge_3",
        "sponge_4",
        "sponge_5",
        "sponge_6",
        "sponge_8",
    )
}
SPONGE_TYPES = tuple(SPONGE_ASSET_BY_TYPE)


def validate_cup_material(value: Any) -> str:
    material_id = str(value)
    if material_id not in CUP_MATERIAL_PROFILES:
        raise ValueError(f"invalid WX-01 cup material: {material_id!r}")
    return material_id


def validate_sponge_type(value: Any) -> str:
    sponge_type = str(value)
    if sponge_type not in SPONGE_ASSET_BY_TYPE:
        raise ValueError(f"invalid WX-01 sponge type: {sponge_type!r}")
    return sponge_type


def resolve_sponge_xml(*, sponge_type: Any, asset_root: str | Path) -> Path:
    resolved_type = validate_sponge_type(sponge_type)
    path = Path(asset_root) / SPONGE_ASSET_BY_TYPE[resolved_type]
    if not path.is_file():
        raise FileNotFoundError(f"missing WX-01 sponge asset: {path}")
    return path


def apply_cup_material(
    model: Any,
    *,
    root_body_id: int,
    material_id: Any,
) -> tuple[int, ...]:
    """Apply a finish to visible cup geoms without changing collision data."""

    resolved_material = validate_cup_material(material_id)
    profile: Mapping[str, Any] = CUP_MATERIAL_PROFILES[resolved_material]
    parents = np.asarray(model.body_parentid, dtype=int)
    visual_geom_ids: list[int] = []
    visual_material_ids: list[int] = []
    for geom_id, body_id in enumerate(np.asarray(model.geom_bodyid, dtype=int)):
        if int(model.geom_group[geom_id]) != 1:
            continue
        current = int(body_id)
        while current > 0 and current != int(root_body_id):
            current = int(parents[current])
        if current != int(root_body_id):
            continue
        visual_geom_ids.append(geom_id)
        geom_material_id = int(model.geom_matid[geom_id])
        if geom_material_id >= 0 and geom_material_id not in visual_material_ids:
            visual_material_ids.append(geom_material_id)
    if not visual_geom_ids or not visual_material_ids:
        raise RuntimeError("WX-01 cup has no material-backed group-1 visual geoms")

    colors = (profile["body_rgba"], profile["accent_rgba"])
    for index, resolved_id in enumerate(visual_material_ids):
        model.mat_rgba[resolved_id, :] = np.asarray(
            colors[min(index, len(colors) - 1)], dtype=float
        )
        model.mat_specular[resolved_id] = float(profile["specular"])
        model.mat_shininess[resolved_id] = float(profile["shininess"])
        model.mat_reflectance[resolved_id] = float(profile["reflectance"])
    return tuple(visual_geom_ids)


__all__ = [
    "CUP_MATERIAL_IDS",
    "CUP_MATERIAL_PROFILES",
    "SPONGE_ASSET_BY_TYPE",
    "SPONGE_TYPES",
    "apply_cup_material",
    "resolve_sponge_xml",
    "validate_cup_material",
    "validate_sponge_type",
]
