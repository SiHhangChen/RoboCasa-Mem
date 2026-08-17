"""High-visibility target object installed inside one TS02 drawer."""

from __future__ import annotations

import xml.etree.ElementTree as ET


TARGET_OBJECT_GEOM_NAME = "bench_ts02_target_red_cube"
TARGET_OBJECT_RGBA = "0.98 0.015 0.015 1"
TARGET_OBJECT_HALF_SIZE_M = 0.050
# Keep the enlarged cube resting on the same drawer-floor height (-0.062 m).
TARGET_OBJECT_DRAWER_POS = (0.0, -0.015, -0.012)


def add_target_red_cube(
    root: ET.Element,
    *,
    drawer_body_name: str,
    geom_name: str = TARGET_OBJECT_GEOM_NAME,
) -> None:
    """Attach a stable 10 cm red cube to a named moving drawer body."""

    existing = root.find(f".//geom[@name='{geom_name}']")
    if existing is not None:
        return
    drawer = root.find(f".//body[@name='{drawer_body_name}']")
    if drawer is None:
        raise RuntimeError(f"TS02 target drawer body {drawer_body_name!r} is missing")
    ET.SubElement(
        drawer,
        "geom",
        {
            "name": str(geom_name),
            "type": "box",
            "pos": " ".join(f"{value:.6f}" for value in TARGET_OBJECT_DRAWER_POS),
            "size": " ".join(f"{TARGET_OBJECT_HALF_SIZE_M:.6f}" for _ in range(3)),
            "rgba": TARGET_OBJECT_RGBA,
            # The target is a visual search cue, not a manipulation target.
            # Keeping it collision-free prevents drawer dynamics from changing.
            "contype": "0",
            "conaffinity": "0",
            "group": "1",
        },
    )


__all__ = [
    "TARGET_OBJECT_DRAWER_POS",
    "TARGET_OBJECT_GEOM_NAME",
    "TARGET_OBJECT_HALF_SIZE_M",
    "TARGET_OBJECT_RGBA",
    "add_target_red_cube",
]
