"""Fixed shelf poses for the WR-06 real ketchup bottle asset.

The real ketchup bottle is already upright in its model frame, so no rotation
is applied on the shelf and the return target keeps the identity quaternion
(unlike the WR-04 sponge, which stands on a long local-Y edge).  Shelf surface
Z is handled by the model bottom_offset, so the placement offset is zero.
"""

from __future__ import annotations


SHELF_UPRIGHT_ROTATION_AXIS = "x"
SHELF_UPRIGHT_ROTATION_RAD = 0.0
SHELF_UPRIGHT_QUAT_XYZW = (0.0, 0.0, 0.0, 1.0)

# The bottle rests directly on the shelf through its model bottom_offset.
SHELF_UPRIGHT_PLACEMENT_Z_OFFSETS_M = {
    "ketchup": 0.0,  # aigen_objs/ketchup/ketchup_3
}


__all__ = [
    "SHELF_UPRIGHT_PLACEMENT_Z_OFFSETS_M",
    "SHELF_UPRIGHT_QUAT_XYZW",
    "SHELF_UPRIGHT_ROTATION_AXIS",
    "SHELF_UPRIGHT_ROTATION_RAD",
]
