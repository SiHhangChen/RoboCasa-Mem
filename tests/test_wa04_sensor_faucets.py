"""Regression coverage for WA-04's twin touchless faucets."""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from bench.execution.contracts import NavigateStage, OperateStage
from bench.tasks import TaskCatalog, resolve_builtin_task_plugin
from bench.tasks.contracts import EpisodeRequest, ExpertPlanInputs
from bench.tasks.wa04.sensor_faucet_asset import (
    install_wa04_twin_sensor_faucets,
    wa04_sensor_faucet_body_name,
    wa04_sensor_faucet_stream_name,
)


ROOT = Path(__file__).resolve().parents[1]


def test_real_sink_spout_is_cloned_into_two_fixed_sensor_faucets() -> None:
    worldbody = ET.Element("worldbody")
    sink = ET.SubElement(worldbody, "body", {"name": "sink"})
    source = ET.SubElement(sink, "body", {"name": "sink_spout"})
    ET.SubElement(source, "joint", {"name": "sink_spout_joint", "type": "hinge"})
    ET.SubElement(
        source,
        "geom",
        {"name": "sink_spout_visual", "type": "mesh", "mesh": "sink_spout_vis"},
    )
    ET.SubElement(
        source,
        "site",
        {"name": "sink_water", "size": "0.0057 0.218", "rgba": "0.5 0.75 1 0.5"},
    )

    installed = install_wa04_twin_sensor_faucets(
        worldbody,
        source_water_site_name="sink_water",
        local_x_offsets={"hot": -0.11, "cold": 0.11},
        local_yaw_degrees={"hot": 35.0, "cold": -35.0},
        stream_half_height=0.22,
    )

    assert set(installed) == {"hot", "cold"}
    assert installed["hot"].get("name") == wa04_sensor_faucet_body_name("hot")
    assert installed["hot"].get("pos") == "-0.11 0 0"
    assert installed["hot"].get("euler") == "0 0 0.610865238198"
    assert installed["cold"].get("pos") == "0.11 0 0"
    assert installed["cold"].get("euler") == "0 0 -0.610865238198"
    for temperature, body in installed.items():
        assert body.find("joint") is None
        visual = next(
            geom for geom in body.findall("geom") if geom.get("mesh") == "sink_spout_vis"
        )
        assert visual.get("rgba") is None
        stream = body.find(
            f"site[@name='{wa04_sensor_faucet_stream_name(temperature)}']"
        )
        assert stream is not None
        assert stream.get("rgba", "").endswith(" 0")
        assert body.find(
            f"geom[@name='{wa04_sensor_faucet_body_name(temperature)}_temperature_collar']"
        ) is not None
        assert body.find(
            f"geom[@name='{wa04_sensor_faucet_body_name(temperature)}_sink_mount_flange']"
        ) is not None


@pytest.mark.parametrize("temperature", ("warm", "", "left"))
def test_sensor_faucet_rejects_unknown_temperature(temperature: str) -> None:
    with pytest.raises(ValueError):
        wa04_sensor_faucet_body_name(temperature)


def test_wa04_layout_is_left_hot_right_cold() -> None:
    path = (
        ROOT
        / "bench/tasks/wa04/resources/poses/sensor_faucet_layout_layout15_style34_seed0.json"
    )
    layout = json.loads(path.read_text(encoding="utf-8"))
    hot = layout["nozzle_positions"]["hot"]
    cold = layout["nozzle_positions"]["cold"]

    # At the calibrated Base yaw, robot-left is +world Y.
    assert hot[1] > cold[1]
    assert abs(hot[0] - cold[0]) < 0.01
    assert hot[2] == pytest.approx(cold[2])
    assert layout["spout_local_x_offsets"] == {"hot": -0.15, "cold": 0.15}
    assert layout["spout_local_yaw_degrees"] == {"hot": 35.0, "cold": -35.0}


def test_wa04_expert_plan_contains_sensor_fills_and_no_button_trip() -> None:
    episode = TaskCatalog.resolve(EpisodeRequest(task_id="WA-04", seed=0))
    api = resolve_builtin_task_plugin("WA-04").load_runtime_api()
    descriptor = api.build_descriptor(episode, False)
    plan = TaskCatalog.get("WA-04").build_expert_plan(
        ExpertPlanInputs.from_descriptor(descriptor)
    )

    operations = [
        stage.operation for stage in plan.stages if isinstance(stage, OperateStage)
    ]
    navigation_targets = [
        stage.target.station_id
        for stage in plan.stages
        if isinstance(stage, NavigateStage)
    ]
    fills = [
        stage
        for stage in plan.stages
        if isinstance(stage, OperateStage)
        and stage.operation == "fill_held_under_faucet"
    ]

    assert len(fills) == 3
    assert {stage.metadata["temperature"] for stage in fills} == {"hot", "cold"}
    assert len([f for f in fills if f.metadata["temperature"] == "hot"]) == 1
    assert len([f for f in fills if f.metadata["temperature"] == "cold"]) == 2
    assert "operate_water_button" not in operations
    assert "water_button_station" not in navigation_targets
    assert descriptor.environment.env_kwargs["sensor_faucet_layout_file"].endswith(
        "sensor_faucet_layout_layout15_style34_seed0.json"
    )
