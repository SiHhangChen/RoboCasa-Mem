"""Seeded appearance-only materials for the three WA-03 cups and sugar cube."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from bench.tasks.wa03.domain.contracts import COFFEE_IDS

COFFEE_MATERIAL_SEED_SALT = 0x4D55474D
COFFEE_MATERIAL_PROFILES: Mapping[str, Mapping[str, Any]] = {
    "dark_blue": {
        "texture": "textures/flat/dark_blue.png",
        "specular": "0.22",
        "shininess": "0.10",
    },
    "green": {
        "texture": "textures/flat/green.png",
        "specular": "0.20",
        "shininess": "0.08",
    },
    "brass": {
        "texture": "textures/metals/brass.png",
        "specular": "0.55",
        "shininess": "0.28",
    },
    "red_wood": {
        "texture": "textures/red-wood.png",
        "specular": "0.22",
        "shininess": "0.10",
    },
    "steel": {
        "texture": "textures/steel-brushed.png",
        "specular": "0.65",
        "shininess": "0.35",
    },
}
DEFAULT_COFFEE_MATERIAL_CANDIDATES = tuple(COFFEE_MATERIAL_PROFILES)
SUGAR_CUBE_MATERIAL_SEED_SALT = 0x53504355
SUGAR_CUBE_MATERIAL_PROFILES: Mapping[str, Mapping[str, Any] | None] = {
    "original": None,
    "sugar_white": {
        "rgba": (1.0, 1.0, 1.0, 1.0),
        "specular": 0.18,
        "shininess": 0.20,
        "reflectance": 0.0,
    },
    "sugar_cream": {
        "rgba": (1.0, 0.89, 0.67, 1.0),
        "specular": 0.16,
        "shininess": 0.18,
        "reflectance": 0.0,
    },
    "sugar_pink": {
        "rgba": (1.0, 0.55, 0.68, 1.0),
        "specular": 0.16,
        "shininess": 0.18,
        "reflectance": 0.0,
    },
    "sugar_blue": {
        "rgba": (0.46, 0.72, 1.0, 1.0),
        "specular": 0.16,
        "shininess": 0.18,
        "reflectance": 0.0,
    },
}
DEFAULT_SUGAR_CUBE_MATERIAL_CANDIDATES = tuple(SUGAR_CUBE_MATERIAL_PROFILES)
_MESH_GEOM_TYPE = 7
_SKIP_NAME_SUFFIXES = ("liquid", "reg_int", "reg_bbox")


def _normalize(
    value: Any, profiles: Mapping[str, Any], label: str, default: tuple[str, ...]
) -> tuple[str, ...]:
    if value is None:
        return default
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"WA-03 {label} must be a non-empty sequence")
    result = tuple(str(item) for item in value)
    if not result or len(set(result)) != len(result):
        raise ValueError(f"WA-03 {label} must be non-empty and unique")
    unknown = sorted(set(result) - set(profiles))
    if unknown:
        raise ValueError(f"unknown WA-03 {label}: {', '.join(unknown)}")
    return result


def normalize_coffee_material_candidates(value: Any) -> tuple[str, ...]:
    return _normalize(
        value,
        COFFEE_MATERIAL_PROFILES,
        "coffee_material_candidates",
        DEFAULT_COFFEE_MATERIAL_CANDIDATES,
    )


def normalize_sugar_cube_material_candidates(value: Any) -> tuple[str, ...]:
    return _normalize(
        value,
        SUGAR_CUBE_MATERIAL_PROFILES,
        "sugar_cube_material_candidates",
        DEFAULT_SUGAR_CUBE_MATERIAL_CANDIDATES,
    )


def sample_coffee_materials(seed: int, candidates: Sequence[str]) -> dict[str, str]:
    normalized = normalize_coffee_material_candidates(candidates)
    material = str(
        np.random.default_rng(
            np.random.SeedSequence([int(seed), COFFEE_MATERIAL_SEED_SALT])
        ).choice(normalized)
    )
    return {coffee_id: material for coffee_id in COFFEE_IDS}


def sample_sugar_cube_material(seed: int, candidates: Sequence[str]) -> str:
    normalized = normalize_sugar_cube_material_candidates(candidates)
    return str(
        np.random.default_rng(
            np.random.SeedSequence([int(seed), SUGAR_CUBE_MATERIAL_SEED_SALT])
        ).choice(normalized)
    )


def apply_coffee_material(
    root: ET.Element, *, coffee_id: str, material_id: str, assets_root: Path
) -> None:
    if coffee_id not in COFFEE_IDS or material_id not in COFFEE_MATERIAL_PROFILES:
        raise ValueError(
            f"unknown WA-03 coffee material binding: {coffee_id!r}/{material_id!r}"
        )
    profile = COFFEE_MATERIAL_PROFILES[material_id]
    texture_path = (Path(assets_root) / str(profile["texture"])).resolve()
    if not texture_path.is_file():
        raise FileNotFoundError(f"missing WA-03 coffee texture: {texture_path}")
    asset = root.find("asset")
    if asset is None:
        raise ValueError("WA-03 coffee material requires an asset section")
    material_name = f"{coffee_id}_lambert3SG"
    material = asset.find(f"material[@name='{material_name}']")
    if material is None:
        raise ValueError(f"missing WA-03 coffee material binding: {material_name}")
    texture_name = f"{coffee_id}_wa03_material_texture"
    texture = asset.find(f"texture[@name='{texture_name}']")
    if texture is None:
        texture = ET.SubElement(asset, "texture", name=texture_name)
    texture.attrib.update(type="2d", file=str(texture_path))
    material.attrib.update(
        texture=texture_name,
        rgba="1 1 1 1",
        specular=str(profile["specular"]),
        shininess=str(profile["shininess"]),
    )


def _geom_name(model: Any, geom_id: int) -> str:
    getter = getattr(model, "geom_id2name", None)
    name = getter(int(geom_id)) if callable(getter) else None
    return "" if name is None else str(name)


def apply_object_material(
    env: Any,
    object_name: str,
    material_id: str,
    profiles: Mapping[str, Mapping[str, Any] | None],
) -> dict[str, Any]:
    if material_id not in profiles:
        raise ValueError(f"unknown material {material_id!r}")
    profile = profiles[material_id]
    if profile is None:
        return {"material": material_id, "applied": False, "visual_geom_count": 0}
    root = int(env.obj_body_id[str(object_name)])
    model = env.sim.model
    parents = np.asarray(model.body_parentid, dtype=int)
    material_ids: set[int] = set()
    count = 0
    for geom_id, body_id in enumerate(np.asarray(model.geom_bodyid, dtype=int)):
        current = int(body_id)
        belongs = False
        while current > 0:
            if current == root:
                belongs = True
                break
            current = int(parents[current])
        if (
            not belongs
            or int(model.geom_group[geom_id]) != 1
            or float(model.geom_rgba[geom_id, 3]) <= 0.0
        ):
            continue
        name = _geom_name(model, geom_id)
        if any(name.endswith(suffix) for suffix in _SKIP_NAME_SUFFIXES):
            continue
        if (
            getattr(model, "geom_type", None) is not None
            and int(model.geom_type[geom_id]) != _MESH_GEOM_TYPE
        ):
            continue
        model.geom_rgba[geom_id, :] = np.asarray(profile["rgba"], dtype=float)
        material_idx = int(model.geom_matid[geom_id])
        if material_idx >= 0:
            material_ids.add(material_idx)
        count += 1
    for material_idx in material_ids:
        model.mat_specular[material_idx] = float(profile["specular"])
        model.mat_shininess[material_idx] = float(profile["shininess"])
        model.mat_reflectance[material_idx] = float(profile["reflectance"])
    return {"material": material_id, "applied": True, "visual_geom_count": count}


__all__ = [
    "COFFEE_MATERIAL_PROFILES",
    "DEFAULT_COFFEE_MATERIAL_CANDIDATES",
    "DEFAULT_SUGAR_CUBE_MATERIAL_CANDIDATES",
    "SUGAR_CUBE_MATERIAL_PROFILES",
    "apply_coffee_material",
    "apply_object_material",
    "normalize_coffee_material_candidates",
    "normalize_sugar_cube_material_candidates",
    "sample_coffee_materials",
    "sample_sugar_cube_material",
]
