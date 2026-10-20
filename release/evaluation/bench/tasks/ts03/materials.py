"""TS03 appearance catalogs and appearance-only material application."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np


# Relative to RoboCasa's ``objects`` asset root.  Keeping the catalog relative
# makes episode facts portable across machines while the environment expands
# the selected entry against the local asset root.
SAUCEPAN_ASSET_BY_TYPE = {
    f"Saucepan{index:03d}": f"lightwheel/saucepan/Saucepan{index:03d}/model.xml"
    for index in range(2, 14)
}
SAUCEPAN_TYPES = tuple(SAUCEPAN_ASSET_BY_TYPE)


def _profile(
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


SAUCEPAN_MATERIAL_PROFILES = {
    "original": None,
    "steel": _profile((0.86, 0.89, 0.92, 1.0), specular=0.72, shininess=0.82, reflectance=0.12),
    "dark_metal": _profile((0.25, 0.28, 0.32, 1.0), specular=0.58, shininess=0.62, reflectance=0.08),
    "copper": _profile((0.76, 0.34, 0.16, 1.0), specular=0.68, shininess=0.76, reflectance=0.10),
}
SUGAR_PLATE_MATERIAL_PROFILES = {
    "original": None,
    "ceramic_white": _profile((1.0, 1.0, 1.0, 1.0), specular=0.28, shininess=0.32),
    "ceramic_blue": _profile((0.25, 0.48, 0.86, 1.0), specular=0.30, shininess=0.34),
    "ceramic_green": _profile((0.25, 0.70, 0.46, 1.0), specular=0.30, shininess=0.34),
    "metal": _profile((0.72, 0.76, 0.82, 1.0), specular=0.62, shininess=0.70, reflectance=0.08),
}
SUGAR_CUBE_MATERIAL_PROFILES = {
    "original": None,
    "sugar_white": _profile((1.0, 1.0, 1.0, 1.0), specular=0.18, shininess=0.20),
    "sugar_cream": _profile((1.0, 0.89, 0.67, 1.0), specular=0.16, shininess=0.18),
    "sugar_pink": _profile((1.0, 0.55, 0.68, 1.0), specular=0.16, shininess=0.18),
    "sugar_blue": _profile((0.46, 0.72, 1.0, 1.0), specular=0.16, shininess=0.18),
}

SAUCEPAN_MATERIALS = tuple(SAUCEPAN_MATERIAL_PROFILES)
SUGAR_PLATE_MATERIALS = tuple(SUGAR_PLATE_MATERIAL_PROFILES)
SUGAR_CUBE_MATERIALS = tuple(SUGAR_CUBE_MATERIAL_PROFILES)
# MuJoCo mesh geoms. Region volumes for liquid/containment are cylinders or boxes.
_MESH_GEOM_TYPE = 7
_SKIP_NAME_SUFFIXES = ("liquid", "reg_int", "reg_bbox")


def _geom_name(model: Any, geom_id: int) -> str:
    getter = getattr(model, "geom_id2name", None)
    if not callable(getter):
        return ""
    name = getter(int(geom_id))
    return "" if name is None else str(name)


def _is_appearance_mesh(model: Any, geom_id: int) -> bool:
    """Keep saucepan liquid and other region volumes out of appearance edits."""

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
    """Apply optical parameters to visual meshes belonging to one object.

    Collision meshes, region geoms (including hot-chocolate liquid), body
    masses, friction and inertial parameters are deliberately untouched.
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
    "SAUCEPAN_ASSET_BY_TYPE",
    "SAUCEPAN_MATERIAL_PROFILES",
    "SAUCEPAN_MATERIALS",
    "SAUCEPAN_TYPES",
    "SUGAR_CUBE_MATERIAL_PROFILES",
    "SUGAR_CUBE_MATERIALS",
    "SUGAR_PLATE_MATERIAL_PROFILES",
    "SUGAR_PLATE_MATERIALS",
    "apply_object_material",
]
