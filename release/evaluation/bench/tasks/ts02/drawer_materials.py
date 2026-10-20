"""Compatibility exports for the shared tabletop-drawer finish catalog."""

from bench.tasks.families.drawer_inspection.materials import (
    DRAWER_MATERIAL_TEXTURES,
    METAL_TEXTURE_NAMES,
    PAINT_TEXTURE_NAMES,
    WOOD_TEXTURE_NAMES,
    apply_drawer_material,
    normalize_drawer_material_candidates,
)

__all__ = [
    "DRAWER_MATERIAL_TEXTURES",
    "METAL_TEXTURE_NAMES",
    "PAINT_TEXTURE_NAMES",
    "WOOD_TEXTURE_NAMES",
    "apply_drawer_material",
    "normalize_drawer_material_candidates",
]
