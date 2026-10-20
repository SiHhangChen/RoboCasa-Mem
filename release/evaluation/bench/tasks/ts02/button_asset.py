"""TM01-style decorative and physical completion button for TS02 (shared asset)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from bench.tasks.button_asset import (
    BUTTON_FIXED_PARTS as TS02_BUTTON_FIXED_PARTS,
    BUTTON_GLOW_RGBA as TS02_BUTTON_GLOW_RGBA,
    BUTTON_MAX_TRAVEL_M as TS02_BUTTON_MAX_TRAVEL_M,
    BUTTON_MOVING_PARTS as TS02_BUTTON_MOVING_PARTS,
    BUTTON_RGBA as TS02_BUTTON_RGBA,
    add_button_asset,
    button_geom_names as ts02_button_geom_names,
)


def add_ts02_button_asset(
    parent: Any,
    *,
    geom_name: str,
    pos: Sequence[float],
    joint_name: str | None = None,
) -> None:
    """Add the complete TM01-style button while retaining TS02 identifiers."""

    add_button_asset(
        parent,
        geom_name=geom_name,
        pos=pos,
        color_rgba=TS02_BUTTON_RGBA,
        glow_rgba=TS02_BUTTON_GLOW_RGBA,
        joint_name=joint_name,
        gravity_compensation=True,
    )


__all__ = [
    "TS02_BUTTON_FIXED_PARTS",
    "TS02_BUTTON_GLOW_RGBA",
    "TS02_BUTTON_MAX_TRAVEL_M",
    "TS02_BUTTON_MOVING_PARTS",
    "TS02_BUTTON_RGBA",
    "add_ts02_button_asset",
    "ts02_button_geom_names",
]
