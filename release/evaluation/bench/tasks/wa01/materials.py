"""Seeded, appearance-only coffee-cup materials for WA-01.

The catalog and XML rewrite match WA-06: same cup mesh, independent per-cup
texture draws, no mass or collision changes.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from bench.tasks.wa01.domain.contracts import COFFEE_IDS


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
_MESH_GEOM_TYPE = 7
_SKIP_NAME_SUFFIXES = ("liquid", "reg_int", "reg_bbox")


def _rgba_profile(
    rgba: tuple[float, float, float, float],
    *,
    specular: float,
    shininess: float,
    reflectance: float = 0.0,
) -> dict[str, Any]:
    return {
        "rgba": rgba,
        "specular": float(specular),
        "shininess": float(shininess),
        "reflectance": float(reflectance),
    }


SUGAR_CUBE_MATERIAL_PROFILES: Mapping[str, Mapping[str, Any] | None] = {
    "original": None,
    "sugar_white": _rgba_profile((1.0, 1.0, 1.0, 1.0), specular=0.18, shininess=0.20),
    "sugar_cream": _rgba_profile((1.0, 0.89, 0.67, 1.0), specular=0.16, shininess=0.18),
    "sugar_pink": _rgba_profile((1.0, 0.55, 0.68, 1.0), specular=0.16, shininess=0.18),
    "sugar_blue": _rgba_profile((0.46, 0.72, 1.0, 1.0), specular=0.16, shininess=0.18),
}
DEFAULT_SUGAR_CUBE_MATERIAL_CANDIDATES = tuple(SUGAR_CUBE_MATERIAL_PROFILES)


def normalize_coffee_material_candidates(value: Any) -> tuple[str, ...]:
    """Validate the finite material catalog exposed by task configuration."""

    if value is None:
        return DEFAULT_COFFEE_MATERIAL_CANDIDATES
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError("WA-01 coffee_material_candidates must be a non-empty sequence")
    candidates = tuple(str(item) for item in value)
    if not candidates or len(set(candidates)) != len(candidates):
        raise ValueError("WA-01 coffee material candidates must be non-empty and unique")
    unknown = sorted(set(candidates) - COFFEE_MATERIAL_PROFILES.keys())
    if unknown:
        raise ValueError("unknown WA-01 coffee materials: " + ", ".join(unknown))
    return candidates


def sample_coffee_materials(seed: int, candidates: Sequence[str]) -> dict[str, str]:
    """Sample one shared, reproducible visual material for both cups."""

    normalized = normalize_coffee_material_candidates(candidates)
    material = str(
        np.random.default_rng(
            np.random.SeedSequence([int(seed), COFFEE_MATERIAL_SEED_SALT])
        ).choice(normalized)
    )
    return {coffee_id: material for coffee_id in COFFEE_IDS}


def apply_coffee_material(
    root: ET.Element,
    *,
    coffee_id: str,
    material_id: str,
    assets_root: Path,
) -> None:
    """Replace only a cup's visual texture and reflectance properties."""

    if coffee_id not in COFFEE_IDS:
        raise ValueError(f"invalid WA-01 coffee id: {coffee_id!r}")
    profile = COFFEE_MATERIAL_PROFILES.get(str(material_id))
    if profile is None:
        raise ValueError(f"unknown WA-01 coffee material: {material_id!r}")
    texture_path = (Path(assets_root) / str(profile["texture"])).resolve()
    if not texture_path.is_file():
        raise FileNotFoundError(f"missing WA-01 coffee texture: {texture_path}")
    asset = root.find("asset")
    if asset is None:
        raise ValueError("WA-01 coffee material requires an asset section")

    material_name = f"{coffee_id}_lambert3SG"
    material = asset.find(f"material[@name='{material_name}']")
    if material is None:
        raise ValueError(f"missing WA-01 coffee material binding: {material_name}")
    texture_name = f"{coffee_id}_wa01_material_texture"
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


def normalize_sugar_cube_material_candidates(value: Any) -> tuple[str, ...]:
    """Validate the TS03-matching sugar-cube appearance catalog."""

    if value is None:
        return DEFAULT_SUGAR_CUBE_MATERIAL_CANDIDATES
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError("WA-01 sugar_cube_material_candidates must be a non-empty sequence")
    candidates = tuple(str(item) for item in value)
    if not candidates or len(set(candidates)) != len(candidates):
        raise ValueError("WA-01 sugar cube material candidates must be non-empty and unique")
    unknown = sorted(set(candidates) - SUGAR_CUBE_MATERIAL_PROFILES.keys())
    if unknown:
        raise ValueError("unknown WA-01 sugar cube materials: " + ", ".join(unknown))
    return candidates


def sample_sugar_cube_material(seed: int, candidates: Sequence[str]) -> str:
    """Sample one reproducible visual material for the single sugar cube."""

    normalized = normalize_sugar_cube_material_candidates(candidates)
    return str(
        np.random.default_rng(
            np.random.SeedSequence([int(seed), SUGAR_CUBE_MATERIAL_SEED_SALT])
        ).choice(normalized)
    )


def _geom_name(model: Any, geom_id: int) -> str:
    getter = getattr(model, "geom_id2name", None)
    if not callable(getter):
        return ""
    name = getter(int(geom_id))
    return "" if name is None else str(name)


def _is_appearance_mesh(model: Any, geom_id: int) -> bool:
    name = _geom_name(model, geom_id)
    if any(name.endswith(suffix) for suffix in _SKIP_NAME_SUFFIXES):
        return False
    geom_types = getattr(model, "geom_type", None)
    if geom_types is None:
        return True
    return int(geom_types[geom_id]) == _MESH_GEOM_TYPE


def apply_object_material(
    env: Any,
    object_name: str,
    material_id: str,
    profiles: Mapping[str, Mapping[str, Any] | None],
) -> dict[str, Any]:
    """Apply optical parameters to visual meshes of one object.

    Collision meshes, region geoms, masses and friction stay untouched.
    """

    if material_id not in profiles:
        raise ValueError(f"unknown material {material_id!r}")
    profile = profiles[material_id]
    if profile is None:
        return {"material": material_id, "applied": False, "visual_geom_count": 0}
    root = int(env.obj_body_id[str(object_name)])
    model = env.sim.model
    parents = np.asarray(model.body_parentid, dtype=int)
    count = 0
    material_ids: set[int] = set()
    for geom_id, body_id in enumerate(np.asarray(model.geom_bodyid, dtype=int)):
        current = int(body_id)
        belongs = False
        while current > 0:
            if current == root:
                belongs = True
                break
            current = int(parents[current])
        if not belongs or int(model.geom_group[geom_id]) != 1:
            continue
        if float(model.geom_rgba[geom_id, 3]) <= 0.0:
            continue
        if not _is_appearance_mesh(model, geom_id):
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
