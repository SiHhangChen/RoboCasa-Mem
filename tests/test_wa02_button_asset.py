from __future__ import annotations

import xml.etree.ElementTree as ET

import pytest

from bench.tasks.wa02_button_asset import (
    WA02_BUTTON_FIXED_PARTS,
    add_wa02_button_asset,
    wa02_button_geom_names,
)


def test_wa02_environment_installs_complete_legacy_button_housing() -> None:
    worldbody = ET.Element("worldbody")
    name = "bench_wa02_test_water_button"

    add_wa02_button_asset(
        worldbody,
        geom_name=name,
        pos=(1.0, 2.0, 0.93),
        color_rgba="0.24 0.54 0.90 1",
        glow_rgba="0.32 0.66 1.00 1",
        scale=0.5,
    )

    geoms = {
        geom.get("name"): geom
        for geom in worldbody.iter("geom")
        if geom.get("name") is not None
    }
    assert set(wa02_button_geom_names(name)) == set(geoms)
    assert all(f"{name}_{suffix}" in geoms for suffix in WA02_BUTTON_FIXED_PARTS)
    assert geoms[f"{name}_faceplate"].get("rgba") == "0.80 0.83 0.84 1"
    assert geoms[f"{name}_side_glow_front"].get("rgba") == "0.32 0.66 1.00 1"
    assert geoms[name].get("contype") == "1"
    assert [float(value) for value in geoms[f"{name}_base"].get("size", "").split()] == [
        0.03,
        0.03,
        0.01,
    ]
    assert [float(value) for value in geoms[name].get("size", "").split()] == [
        0.0145,
        0.004,
    ]

    joint = worldbody.find(f".//joint[@name='{name}_press_joint']")
    assert joint is not None
    assert [float(value) for value in joint.get("range", "").split()] == [-0.006, 0.0]


@pytest.mark.parametrize("scale", [0.0, -0.5, float("inf"), float("nan")])
def test_wa02_button_scale_must_be_positive_and_finite(scale: float) -> None:
    with pytest.raises(ValueError, match="scale"):
        add_wa02_button_asset(
            ET.Element("worldbody"),
            geom_name="bench_wa02_invalid_scale_button",
            pos=(0.0, 0.0, 0.93),
            color_rgba="1 0 0 1",
            glow_rgba="1 0.2 0.2 1",
            scale=scale,
        )
