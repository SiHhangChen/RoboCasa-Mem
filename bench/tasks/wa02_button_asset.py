"""Canonical MJCF builder for the WA02 alternate-button primitive."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from collections.abc import Sequence


WA02_BUTTON_FIXED_PARTS = (
    "base",
    "raised_shoulder",
    "faceplate",
    "side_glow_front",
    "side_glow_back",
    "side_glow_right",
    "side_glow_left",
    "glow_ring_outer",
    "glow_ring_inner_cutout",
    "bezel",
)
WA02_BUTTON_MOVING_PARTS = ("plunger_stem", "plunger")
WA02_BUTTON_MAX_TRAVEL_M = 0.012


def wa02_button_geom_names(geom_name: str) -> tuple[str, ...]:
    """Return every geom installed for one canonical WA02 button."""

    name = str(geom_name)
    return tuple(f"{name}_{suffix}" for suffix in WA02_BUTTON_FIXED_PARTS) + (
        f"{name}_plunger_stem",
        name,
    )


def add_wa02_button_asset(
    parent: ET.Element,
    *,
    geom_name: str,
    pos: Sequence[float],
    color_rgba: str,
    glow_rgba: str,
    joint_name: str | None = None,
    gravity_compensation: bool = False,
) -> None:
    """Add WA02's complete decorative and physical button assembly."""

    name = str(geom_name)
    position = tuple(float(item) for item in pos)
    if len(position) != 3:
        raise ValueError("WA02 button position must contain exactly three values")
    resolved_joint_name = str(joint_name or f"{name}_press_joint")
    body = ET.SubElement(
        parent,
        "body",
        {
            "name": f"{name}_body",
            "pos": " ".join(f"{item:.6f}" for item in position),
        },
    )

    visual_only = {"contype": "0", "conaffinity": "0", "group": "1"}
    visual_parts = (
        (
            "base",
            "box",
            "0 0 0.020",
            "0.060 0.060 0.020",
            "0.10 0.11 0.12 1",
        ),
        (
            "raised_shoulder",
            "box",
            "0 0 0.043",
            "0.052 0.052 0.008",
            "0.23 0.25 0.27 1",
        ),
        (
            "faceplate",
            "box",
            "0 0 0.055",
            "0.041 0.041 0.003",
            "0.80 0.83 0.84 1",
        ),
        (
            "side_glow_front",
            "box",
            "0 0.061 0.025",
            "0.037 0.003 0.013",
            glow_rgba,
        ),
        (
            "side_glow_back",
            "box",
            "0 -0.061 0.025",
            "0.037 0.003 0.013",
            glow_rgba,
        ),
        (
            "side_glow_right",
            "box",
            "0.061 0 0.025",
            "0.003 0.037 0.013",
            glow_rgba,
        ),
        (
            "side_glow_left",
            "box",
            "-0.061 0 0.025",
            "0.003 0.037 0.013",
            glow_rgba,
        ),
        (
            "glow_ring_outer",
            "cylinder",
            "0 0 0.059",
            "0.033 0.003",
            glow_rgba,
        ),
        (
            "glow_ring_inner_cutout",
            "cylinder",
            "0 0 0.061",
            "0.028 0.003",
            "0.80 0.83 0.84 1",
        ),
        (
            "bezel",
            "cylinder",
            "0 0 0.063",
            "0.031 0.003",
            "0.80 0.83 0.84 1",
        ),
    )
    for suffix, geom_type, local_pos, size, rgba in visual_parts:
        ET.SubElement(
            body,
            "geom",
            {
                "name": f"{name}_{suffix}",
                "type": geom_type,
                "pos": local_pos,
                "size": size,
                "rgba": rgba,
                **visual_only,
            },
        )

    plunger_attributes = {
        "name": f"{name}_plunger",
        "pos": "0 0 0",
    }
    if gravity_compensation:
        plunger_attributes["gravcomp"] = "1"
    plunger = ET.SubElement(
        body,
        "body",
        plunger_attributes,
    )
    ET.SubElement(
        plunger,
        "inertial",
        {
            "pos": "0 0 0.074",
            "mass": "0.020",
            "diaginertia": "0.00002 0.00002 0.00002",
        },
    )
    ET.SubElement(
        plunger,
        "joint",
        {
            "name": resolved_joint_name,
            "type": "slide",
            "axis": "0 0 1",
            "limited": "true",
            "range": "-0.012 0",
            "springref": "0",
            "stiffness": "18",
            "damping": "2",
        },
    )
    ET.SubElement(
        plunger,
        "geom",
        {
            "name": f"{name}_plunger_stem",
            "type": "cylinder",
            "pos": "0 0 0.068",
            "size": "0.020 0.007",
            "rgba": color_rgba,
            **visual_only,
        },
    )
    ET.SubElement(
        plunger,
        "geom",
        {
            "name": name,
            "type": "cylinder",
            "pos": "0 0 0.075",
            "size": "0.029 0.008",
            "rgba": color_rgba,
            "contype": "1",
            "conaffinity": "1",
            "group": "1",
            "friction": "0.8 0.005 0.0001",
        },
    )


__all__ = [
    "WA02_BUTTON_FIXED_PARTS",
    "WA02_BUTTON_MAX_TRAVEL_M",
    "WA02_BUTTON_MOVING_PARTS",
    "add_wa02_button_asset",
    "wa02_button_geom_names",
]
