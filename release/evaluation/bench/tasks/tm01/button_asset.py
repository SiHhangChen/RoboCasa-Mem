"""Canonical MJCF builder for TM01 completion buttons (shared asset)."""

from __future__ import annotations

from bench.tasks.button_asset import (
    BUTTON_FIXED_PARTS as TM01_BUTTON_FIXED_PARTS,
    BUTTON_MAX_TRAVEL_M as TM01_BUTTON_MAX_TRAVEL_M,
    BUTTON_MOVING_PARTS as TM01_BUTTON_MOVING_PARTS,
    add_button_asset as add_tm01_button_asset,
    button_geom_names as tm01_button_geom_names,
)

__all__ = [
    "TM01_BUTTON_FIXED_PARTS",
    "TM01_BUTTON_MAX_TRAVEL_M",
    "TM01_BUTTON_MOVING_PARTS",
    "add_tm01_button_asset",
    "tm01_button_geom_names",
]
