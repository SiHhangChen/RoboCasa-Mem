"""Clone Sink059's real spout mesh into two fixed WA-02 sensor faucets."""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from copy import deepcopy
from typing import Any


TEMPERATURE_COLORS = {
    "hot": "0.92 0.18 0.10 1",
    "cold": "0.10 0.38 0.95 1",
}


def wa02_sensor_faucet_stream_name(temperature: str) -> str:
    return f"bench_wa02_{_temperature(temperature)}_water_stream"


def wa02_sensor_faucet_body_name(temperature: str) -> str:
    return f"bench_wa02_{_temperature(temperature)}_sensor_faucet"


def install_wa02_twin_sensor_faucets(
    worldbody: ET.Element,
    *,
    source_water_site_name: str,
    local_x_offsets: dict[str, float],
    local_yaw_degrees: dict[str, float],
    stream_half_height: float,
) -> dict[str, ET.Element]:
    """Duplicate the scene's real sink spout as fixed hot/cold faucets.

    The source body retains Sink059's visual mesh and collision geometry.  Each
    clone has its hinge removed, receives a unique name and water site, and is
    translated along the sink's local X axis.  In layout 15, negative local X
    is robot-left (+world Y), so the hot/cold ordering is explicit in config.
    """

    if set(local_x_offsets) != set(TEMPERATURE_COLORS):
        raise ValueError("WA-02 twin faucets require hot and cold local offsets")
    offsets = {key: float(value) for key, value in local_x_offsets.items()}
    if not all(math.isfinite(value) for value in offsets.values()):
        raise ValueError("WA-02 faucet offsets must be finite")
    if set(local_yaw_degrees) != set(TEMPERATURE_COLORS):
        raise ValueError("WA-02 twin faucets require hot and cold local yaw angles")
    yaws = {key: float(value) for key, value in local_yaw_degrees.items()}
    if not all(math.isfinite(value) for value in yaws.values()):
        raise ValueError("WA-02 faucet yaw angles must be finite")
    half_height = float(stream_half_height)
    if not math.isfinite(half_height) or half_height <= 0.0:
        raise ValueError("WA-02 stream half-height must be positive and finite")

    source, parent = _body_and_parent_with_direct_site(
        worldbody, str(source_water_site_name)
    )
    installed: dict[str, ET.Element] = {}
    existing_sites = {
        site.get("name") for site in worldbody.iter("site") if site.get("name")
    }
    for temperature in TEMPERATURE_COLORS:
        stream_name = wa02_sensor_faucet_stream_name(temperature)
        if stream_name in existing_sites:
            continue
        clone = deepcopy(source)
        _remove_descendants(clone, "joint")
        source_site = next(
            (
                site
                for site in clone.iter("site")
                if site.get("name") == source_water_site_name
            ),
            None,
        )
        if source_site is None:
            raise ValueError("cloned WA-02 spout lost its water site")

        prefix = wa02_sensor_faucet_body_name(temperature)
        _rename_named_elements(clone, prefix=prefix)
        clone.set("name", prefix)
        position = _vector3(clone.get("pos", "0 0 0"), "source spout body position")
        position[0] += offsets[temperature]
        clone.set("pos", _array_to_string(position))
        if clone.get("quat") is not None or clone.get("axisangle") is not None:
            raise ValueError("WA-02 source spout has an unsupported body orientation")
        clone.set("euler", f"0 0 {math.radians(yaws[temperature]):.12g}")

        source_site.set("name", stream_name)
        original_size = _float_values(source_site.get("size", "0.0058 0.2"))
        radius = original_size[0] if original_size else 0.0058
        source_site.set("size", f"{radius:.12g} {half_height:.12g}")
        source_site.set("rgba", "0.42 0.70 1.00 0")
        source_site.set("group", "1")

        # Preserve Sink059's original metal appearance. Its renderable
        # collision approximation stays hidden; compact collars below provide
        # the only hot / cold color coding.
        for geom in clone.iter("geom"):
            if geom.get("mesh") is not None:
                geom.attrib.pop("rgba", None)
            else:
                geom.set("rgba", "0 0 0 0")

        _add_temperature_markers(clone, temperature=temperature, prefix=prefix)
        parent.append(clone)
        installed[temperature] = clone
    return installed


def _add_temperature_markers(
    body: ET.Element,
    *,
    temperature: str,
    prefix: str,
) -> None:
    color = TEMPERATURE_COLORS[_temperature(temperature)]
    common: dict[str, Any] = {
        "contype": "0",
        "conaffinity": "0",
        "group": "1",
    }
    # Sink059's stem starts just above the fixture surface. This flange bridges
    # that small visual gap and makes each clone read as mounted to the sink's
    # rear rim. The clone remains a child of the original sink body as well.
    ET.SubElement(
        body,
        "geom",
        {
            **common,
            "name": f"{prefix}_sink_mount_flange",
            "type": "cylinder",
            "pos": "-0.00364665 0.19572318 0.030",
            "size": "0.031 0.006",
            "rgba": "0.34 0.36 0.38 1",
        },
    )
    # A broad red/blue collar around the rear stem remains visible from both
    # stereo cameras; a dark window at the outlet communicates touchless use.
    ET.SubElement(
        body,
        "geom",
        {
            **common,
            "name": f"{prefix}_temperature_collar",
            "type": "cylinder",
            "pos": "-0.00364665 0.19572318 0.11842547",
            "size": "0.023 0.014",
            "rgba": color,
        },
    )
    ET.SubElement(
        body,
        "geom",
        {
            **common,
            "name": f"{prefix}_sensor_window",
            "type": "sphere",
            "pos": "-0.00350536 0.01167345 0.178",
            "size": "0.010",
            "rgba": "0.01 0.015 0.02 1",
        },
    )


def _body_and_parent_with_direct_site(
    worldbody: ET.Element,
    site_name: str,
) -> tuple[ET.Element, ET.Element]:
    for parent in worldbody.iter():
        for child in list(parent):
            if child.tag != "body":
                continue
            if any(site.get("name") == site_name for site in child.findall("site")):
                return child, parent
    raise ValueError(f"WA-02 source sink spout was not found for site {site_name!r}")


def _remove_descendants(root: ET.Element, tag: str) -> None:
    for parent in root.iter():
        for child in list(parent):
            if child.tag == tag:
                parent.remove(child)


def _rename_named_elements(root: ET.Element, *, prefix: str) -> None:
    counter = 0
    for element in root.iter():
        name = element.get("name")
        if name is None:
            continue
        counter += 1
        suffix = "".join(character if character.isalnum() else "_" for character in name)
        element.set("name", f"{prefix}_{counter}_{suffix}")


def _vector3(value: str, label: str) -> list[float]:
    values = _float_values(value)
    if len(values) != 3 or not all(math.isfinite(item) for item in values):
        raise ValueError(f"WA-02 {label} must contain three finite values")
    return values


def _float_values(value: str) -> list[float]:
    return [float(item) for item in str(value).split()]


def _array_to_string(values: list[float]) -> str:
    return " ".join(f"{value:.12g}" for value in values)


def _temperature(value: str) -> str:
    temperature = str(value).strip().lower()
    if temperature not in TEMPERATURE_COLORS:
        raise ValueError(f"invalid WA-02 faucet temperature: {value!r}")
    return temperature


__all__ = [
    "install_wa02_twin_sensor_faucets",
    "wa02_sensor_faucet_body_name",
    "wa02_sensor_faucet_stream_name",
]
