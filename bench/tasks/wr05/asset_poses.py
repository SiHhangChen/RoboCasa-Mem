"""Fixed shelf poses for the WR-05 ketchup and mustard bottle assets.

The real condiment bottles are already upright in their model frame, so no
rotation is applied on the shelf and the return target keeps the identity
quaternion (unlike the WR-02 sponge/soap, which stand on a long local-Y
edge).  Shelf surface Z is handled by the model bottom_offset, so the
placement offset is zero.
"""

from __future__ import annotations


SHELF_UPRIGHT_ROTATION_AXIS = "x"
SHELF_UPRIGHT_ROTATION_RAD = 0.0
SHELF_UPRIGHT_QUAT_XYZW = (0.0, 0.0, 0.0, 1.0)

# Bottles rest directly on the shelf through their model bottom_offset.
SHELF_UPRIGHT_PLACEMENT_Z_OFFSETS_M = {
    "ketchup": 0.0,  # aigen_objs/ketchup/ketchup_3
    "mustard": 0.0,  # lightwheel/mustard/Mustard003
}


__all__ = [
    "SHELF_UPRIGHT_PLACEMENT_Z_OFFSETS_M",
    "SHELF_UPRIGHT_QUAT_XYZW",
    "SHELF_UPRIGHT_ROTATION_AXIS",
    "SHELF_UPRIGHT_ROTATION_RAD",
]
