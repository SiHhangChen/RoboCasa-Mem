from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from bench.execution.contracts import NavigateStage
from bench.execution.planning.nav_algo import normalize_angle
from bench.tasks import TaskCatalog, resolve_builtin_task_plugin
from bench.tasks.contracts import EpisodeRequest, ExpertPlanInputs
from bench.tasks.wa01.domain.contracts import RobotBaseMode
from bench.tasks.wa01.execution.navigation import (
    StationRandomization,
    build_station_sampler,
    resolve_station_randomization,
    sample_initial_yaw,
    sample_turnaround_yaw,
)
from bench.tasks.wa01.initialization import apply_robot_base_initial_state


COFFEE = "coffee_machine_station"
DINING = "dining_table_station"
FIVE_DEG = float(np.deg2rad(5.0))


def _descriptor(seed: int):
    episode = TaskCatalog.resolve(EpisodeRequest(task_id="WA-01", seed=seed))
    api = resolve_builtin_task_plugin("WA-01").load_runtime_api()
    return api.build_descriptor(episode, False)


def test_wa01_uses_two_xy_only_station_disks_and_a_two_point_path() -> None:
    descriptor = _descriptor(0)
    base = descriptor.config.initial_state.robot_base
    assert base.mode is RobotBaseMode.STATION_SAMPLE
    assert base.station_id == COFFEE

    stations = json.loads(descriptor.resources["station_poses"].read_text())["stations"]
    expected = {
        COFFEE: ([0.244, -5.274], 3.141),
        DINING: ([1.151, -5.259], 0.072),
    }
    assert set(stations) == set(expected)
    for station_id, (center, yaw) in expected.items():
        circle = stations[station_id]["circle"]
        assert circle["center_xy"] == center
        assert circle["radius_m"] == 0.05
        assert circle["base_yaw"] == yaw
        assert circle["yaw_half_range_rad"] == 0.0
        assert stations[station_id]["samples"] == []

    path = json.loads(descriptor.resources["manual_base_path"].read_text())
    samples = {int(item["index"]): item for item in path["samples"]}
    assert set(samples) == {0, 1}
    assert samples[0]["station_id"] == COFFEE
    assert samples[1]["station_id"] == DINING
    assert "coffee_turn" not in path["station_targets"]
    assert path["station_targets"]["dining_delivery"]["routes"] == {"0": [0, 1]}

    randomization = resolve_station_randomization(descriptor.config.navigation)
    assert randomization.initial_yaw_half_range_rad == pytest.approx(FIVE_DEG)
    assert randomization.turnaround_half_range_rad == pytest.approx(FIVE_DEG)
    assert randomization.turnaround_delta_rad == pytest.approx(np.pi)


def test_wa01_station_xy_is_redrawn_each_visit_and_owned_by_episode_seed() -> None:
    descriptor = _descriptor(11)
    first = build_station_sampler(descriptor.navigation, seed=11)
    repeat = build_station_sampler(descriptor.navigation, seed=11)
    other = build_station_sampler(descriptor.navigation, seed=12)
    stations = json.loads(descriptor.resources["station_poses"].read_text())["stations"]

    draws = []
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
    assert draws[0] != draws[2]
    assert draws[1] != draws[3]


def test_wa01_initial_yaw_is_independent_of_the_xy_disk() -> None:
    randomization = StationRandomization()
    first = sample_initial_yaw(3.141, seed=17, randomization=randomization)
    repeat = sample_initial_yaw(3.141, seed=17, randomization=randomization)
    other = sample_initial_yaw(3.141, seed=18, randomization=randomization)
    assert first == repeat
    assert first != other
    assert abs(normalize_angle(first - 3.141)) <= FIVE_DEG + 1e-9
    assert abs(normalize_angle(other - 3.141)) <= FIVE_DEG + 1e-9


def test_wa01_turnaround_is_180_plus_or_minus_five_and_seed_owned() -> None:
    randomization = StationRandomization()
    current = 3.141
    first = [
        sample_turnaround_yaw(current, seed=17, draw_index=index, randomization=randomization)
        for index in range(3)
    ]
    repeat = [
        sample_turnaround_yaw(current, seed=17, draw_index=index, randomization=randomization)
        for index in range(3)
    ]
    other = sample_turnaround_yaw(current, seed=18, draw_index=0, randomization=randomization)
    assert first == repeat
    assert {round(delta, 9) for _, delta in first} != {first[0][1]}
    assert other[1] != first[0][1]
    for target, delta in first:
        assert abs(normalize_angle(delta - np.pi)) <= FIVE_DEG + 1e-9
        assert target == pytest.approx(normalize_angle(current + delta))
    assert sample_turnaround_yaw(
        current, seed=17, draw_index=1, randomization=randomization
    ) != sample_turnaround_yaw(
        current, seed=18, draw_index=0, randomization=randomization
    )


def test_wa01_reset_consumes_the_first_coffee_draw_and_applies_initial_yaw(
    monkeypatch,
) -> None:
    descriptor = _descriptor(17)
    sampler = build_station_sampler(descriptor.navigation, seed=17)
    expected_xy = np.asarray(
        sampler.sample(COFFEE, purpose="initialization").position[:2], dtype=float
    )
    expected_yaw = sample_initial_yaw(
        3.141,
        seed=17,
        randomization=resolve_station_randomization(descriptor.config.navigation),
    )
    applied = {}

    def fake_set_base_pose(env, obs, *, target_xy, target_yaw, mode):
        del env
        applied.update(
            target_xy=np.asarray(target_xy, dtype=float),
            target_yaw=float(target_yaw),
            mode=mode,
        )
        return obs, {"enabled": True, "mode": mode}

    monkeypatch.setattr(
        "bench.tasks.wa01.initialization._set_base_pose", fake_set_base_pose
    )
    env = SimpleNamespace()
    _, summary = apply_robot_base_initial_state(
        env,
        {"robot0_base_pos": np.asarray([0.0, 0.0, 0.0])},
        config=descriptor.config.initial_state.robot_base,
        nav_config=descriptor.navigation,
        seed=17,
        settings=descriptor.config.navigation,
    )

    np.testing.assert_allclose(applied["target_xy"], expected_xy)
    assert applied["target_yaw"] == pytest.approx(expected_yaw)
    assert applied["mode"] == RobotBaseMode.STATION_SAMPLE.value
    assert summary["sampled_yaw"] == pytest.approx(expected_yaw)
    second = env._wa01_station_sampler.sample(COFFEE)
    assert tuple(second.position[:2]) != tuple(expected_xy)


def test_wa01_expert_plan_keeps_same_station_sugar_skips_and_cross_station_delivery() -> None:
    descriptor = _descriptor(17)
    plan = TaskCatalog.get("WA-01").build_expert_plan(
        ExpertPlanInputs.from_descriptor(descriptor)
    )
    navigations = [stage for stage in plan.stages if isinstance(stage, NavigateStage)]
    sugar = [stage for stage in navigations if "sugar" in stage.name]
    delivery = [stage for stage in navigations if "dining" in stage.name]
    second_pick = [stage for stage in navigations if "second_delivery" in stage.name]
    assert sugar
    assert all(stage.skip_if_reachable for stage in sugar)
    assert delivery
    assert all(not stage.skip_if_reachable for stage in delivery)
    assert second_pick
    assert all("use_sampled_terminal_pose" not in stage.metadata for stage in navigations)
