"""Fixed shelf poses for the WR-02 sponge and bar-soap assets."""

from __future__ import annotations

from math import pi, sqrt


SHELF_UPRIGHT_ROTATION_AXIS = "x"
SHELF_UPRIGHT_ROTATION_RAD = pi / 2.0
SHELF_UPRIGHT_QUAT_XYZW = (sqrt(0.5), 0.0, 0.0, sqrt(0.5))

# RoboCasa computes the sampled root Z from the unrotated bottom_offset. These
# per-asset corrections put the rotated local-Y edge directly on the shelf.
SHELF_UPRIGHT_PLACEMENT_Z_OFFSETS_M = {
    "ketchup": 0.02125942,  # objaverse/sponge/sponge_3
    "mustard": 0.02145193,  # aigen_objs/bar_soap/bar_soap_4
}


__all__ = [
    "SHELF_UPRIGHT_PLACEMENT_Z_OFFSETS_M",
    "SHELF_UPRIGHT_QUAT_XYZW",
    "SHELF_UPRIGHT_ROTATION_AXIS",
    "SHELF_UPRIGHT_ROTATION_RAD",
]
