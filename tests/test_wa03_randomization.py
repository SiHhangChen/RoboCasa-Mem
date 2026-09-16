from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from bench.execution.planning.nav_algo import normalize_angle
from bench.tasks import TaskCatalog, resolve_builtin_task_plugin
from bench.tasks.contracts import EpisodeRequest
from bench.tasks.wa03.domain.contracts import COFFEE_IDS, RobotBaseMode
from bench.tasks.wa03.execution.navigation import (
    StationRandomization,
    build_station_sampler,
    resolve_station_randomization,
    sample_initial_yaw,
    sample_turnaround_yaw,
)
from bench.tasks.wa03.initialization import apply_robot_base_initial_state
from bench.tasks.wa03.materials import (
    COFFEE_MATERIAL_PROFILES,
    SUGAR_CUBE_MATERIAL_PROFILES,
    apply_coffee_material,
    sample_coffee_materials,
    sample_sugar_cube_material,
)

ROOT = Path(__file__).resolve().parents[1]
ASSETS_ROOT = ROOT / "robocasa/robocasa/models/assets"
COFFEE = "coffee_machine_station"
DINING = "dining_table_station"
FIVE_DEG = float(np.deg2rad(5.0))


def _descriptor(seed: int):
    episode = TaskCatalog.resolve(EpisodeRequest(task_id="WA-03", seed=seed))
    api = resolve_builtin_task_plugin("WA-03").load_runtime_api()
    return api.build_descriptor(episode, False)


def test_wa03_uses_two_station_disks_and_episode_seeded_yaw() -> None:
    descriptor = _descriptor(0)
    base = descriptor.config.initial_state.robot_base
    assert base.mode is RobotBaseMode.STATION_SAMPLE
    assert base.station_id == COFFEE
    stations = json.loads(descriptor.resources["station_poses"].read_text())["stations"]
    expected = {COFFEE: ([0.244, -5.274], 3.141), DINING: ([1.151, -5.259], 0.072)}
    for station_id, (center, yaw) in expected.items():
        circle = stations[station_id]["circle"]
        assert circle["center_xy"] == center
        assert circle["radius_m"] == 0.05
        assert circle["base_yaw"] == yaw
        assert stations[station_id]["samples"] == []
    path = json.loads(descriptor.resources["manual_base_path"].read_text())
    assert {int(item["index"]) for item in path["samples"]} == {0, 1}
    assert "coffee_turn" not in path["station_targets"]
    assert path["station_targets"]["coffee_C:pick"]["routes"] == {"1": [1, 0]}
    randomization = resolve_station_randomization(descriptor.config.navigation)
    assert randomization.initial_yaw_half_range_rad == pytest.approx(FIVE_DEG)
    assert randomization.turnaround_half_range_rad == pytest.approx(FIVE_DEG)


def test_wa03_station_xy_is_redrawn_each_visit_and_reproducible() -> None:
    descriptor = _descriptor(11)
    first = build_station_sampler(descriptor.navigation, seed=11)
    repeat = build_station_sampler(descriptor.navigation, seed=11)
    other = build_station_sampler(descriptor.navigation, seed=12)
    stations = json.loads(descriptor.resources["station_poses"].read_text())["stations"]
    draws: list[tuple[float, ...]] = []
    for station_id in (COFFEE, DINING, COFFEE, DINING):
        one = first.sample(station_id)
        two = repeat.sample(station_id)
        three = other.sample(station_id)
        assert one.position[:2] == two.position[:2]
        assert one.position[:2] != three.position[:2]
        center = np.asarray(stations[station_id]["circle"]["center_xy"], dtype=float)
        assert np.linalg.norm(np.asarray(one.position[:2]) - center) <= 0.05 + 1e-9
        draws.append(tuple(one.position[:2]))
    assert len(set(draws)) == 4


def test_wa03_yaw_draws_are_seeded_and_bounded() -> None:
    randomization = StationRandomization()
    first = sample_initial_yaw(3.141, seed=17, randomization=randomization)
    assert first == sample_initial_yaw(3.141, seed=17, randomization=randomization)
    assert first != sample_initial_yaw(3.141, seed=18, randomization=randomization)
    assert abs(normalize_angle(first - 3.141)) <= FIVE_DEG + 1e-9
    target, delta = sample_turnaround_yaw(
        3.141, seed=17, draw_index=0, randomization=randomization
    )
    assert abs(normalize_angle(delta - np.pi)) <= FIVE_DEG + 1e-9
    assert target == pytest.approx(normalize_angle(3.141 + delta))


def test_wa03_reset_consumes_first_station_draw(monkeypatch) -> None:
    descriptor = _descriptor(17)
    sampler = build_station_sampler(descriptor.navigation, seed=17)
    expected_xy = np.asarray(
        sampler.sample(COFFEE, purpose="initialization").position[:2]
    )
    applied: dict[str, object] = {}

    def fake_set_base_pose(env, obs, *, target_xy, target_yaw, mode):
        del env
        applied.update(
            target_xy=np.asarray(target_xy), target_yaw=target_yaw, mode=mode
        )
        return obs, {"enabled": True, "mode": mode}

    monkeypatch.setattr(
        "bench.tasks.wa03.initialization._set_base_pose", fake_set_base_pose
    )
    env = SimpleNamespace()
    apply_robot_base_initial_state(
        env,
        {"robot0_base_pos": np.asarray([0.0, 0.0, 0.0])},
        config=descriptor.config.initial_state.robot_base,
        nav_config=descriptor.navigation,
        seed=17,
        settings=descriptor.config.navigation,
    )
    np.testing.assert_allclose(applied["target_xy"], expected_xy)
    assert tuple(env._wa03_station_sampler.sample(COFFEE).position[:2]) != tuple(
        expected_xy
    )


def test_wa03_materials_are_shared_across_three_cups_and_seeded() -> None:
    descriptors = [_descriptor(seed) for seed in range(24)]
    assert (
        len(
            {
                next(iter(item.episode_plan.coffee_material_by_id.values()))
                for item in descriptors
            }
        )
        > 1
    )
    for seed, descriptor in enumerate(descriptors):
        drawn = descriptor.episode_plan.coffee_material_by_id
        assert set(drawn) == set(COFFEE_IDS)
        assert len(set(drawn.values())) == 1
        assert drawn == sample_coffee_materials(seed, tuple(COFFEE_MATERIAL_PROFILES))
        cube = descriptor.episode_plan.sugar_cube_material
        assert cube == sample_sugar_cube_material(
            seed, tuple(SUGAR_CUBE_MATERIAL_PROFILES)
        )


def test_wa03_material_rewrite_is_scoped_to_one_cup() -> None:
    root = ET.fromstring(
        """
      <mujoco><asset>
        <texture name="coffee_A_image0" type="2d" file="a.png"/>
        <texture name="coffee_B_image0" type="2d" file="b.png"/>
        <material name="coffee_A_lambert3SG" texture="coffee_A_image0"/>
        <material name="coffee_B_lambert3SG" texture="coffee_B_image0"/>
      </asset><worldbody/></mujoco>
    """
    )
    apply_coffee_material(
        root, coffee_id="coffee_A", material_id="steel", assets_root=ASSETS_ROOT
    )
    selected = root.find("./asset/material[@name='coffee_A_lambert3SG']")
    untouched = root.find("./asset/material[@name='coffee_B_lambert3SG']")
    assert (
        selected is not None
        and selected.get("texture") == "coffee_A_wa03_material_texture"
    )
    assert untouched is not None and untouched.get("texture") == "coffee_B_image0"
