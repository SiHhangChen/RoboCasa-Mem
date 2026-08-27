"""Canonical MJCF builder for the shared TM01-style completion button.

TM01, TS02, TS01 and TS03 all install this decorative and physical button.
Per-task modules re-export this builder under their own identifiers so the
installed geometry stays identical across tasks.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from collections.abc import Sequence

BUTTON_FIXED_PARTS = (
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
BUTTON_MOVING_PARTS = ("plunger_stem", "plunger")
BUTTON_MAX_TRAVEL_M = 0.012
BUTTON_RGBA = "0.22 0.72 0.32 1"
BUTTON_GLOW_RGBA = "0.26 0.84 0.38 1"


def add_compact_temperature_button_asset(
    parent: ET.Element,
    *,
    geom_name: str,
    pos: Sequence[float],
    color_rgba: str,
    joint_name: str | None = None,
) -> None:
    """Install the compact pressable button used by the original WA-02 scene."""

    name = str(geom_name)
    position = tuple(float(item) for item in pos)
    if len(position) != 3:
        raise ValueError("compact button position must contain exactly three values")
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
    ET.SubElement(
        body,
        "geom",
        {
            "name": f"{name}_base",
            "type": "box",
            "pos": "0 0 0.020",
            "size": "0.016 0.025 0.020",
            "rgba": "0.10 0.11 0.12 1",
            **visual_only,
        },
    )
    plunger = ET.SubElement(
        body,
        "body",
        {
            "name": f"{name}_plunger",
            "pos": "0 0 0",
            # Preserve the original compact joint parameters without allowing
            # the 20 g plunger to sag through the 4 mm press threshold at rest.
            "gravcomp": "1",
        },
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
            "name": name,
            "type": "cylinder",
            "pos": "0 0 0.075",
            "size": "0.012 0.008",
            "rgba": str(color_rgba),
            "contype": "1",
            "conaffinity": "1",
            "group": "1",
            "friction": "0.8 0.005 0.0001",
        },
    )


def button_geom_names(geom_name: str) -> tuple[str, ...]:
    """Return every geom installed for one completion button."""

    name = str(geom_name)
    return tuple(f"{name}_{suffix}" for suffix in BUTTON_FIXED_PARTS) + (
        f"{name}_plunger_stem",
        name,
    )


def add_button_asset(
    parent: ET.Element,
    *,
    geom_name: str,
    pos: Sequence[float],
    color_rgba: str,
    glow_rgba: str,
    joint_name: str | None = None,
    gravity_compensation: bool = False,
) -> None:
    """Add the complete TM01-style decorative and physical button assembly."""

    name = str(geom_name)
    position = tuple(float(item) for item in pos)
    if len(position) != 3:
        raise ValueError("completion button position must contain exactly three values")
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
    "BUTTON_FIXED_PARTS",
    "BUTTON_GLOW_RGBA",
    "BUTTON_MAX_TRAVEL_M",
    "BUTTON_MOVING_PARTS",
    "BUTTON_RGBA",
    "add_compact_temperature_button_asset",
    "add_button_asset",
    "button_geom_names",
]
