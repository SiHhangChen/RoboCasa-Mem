from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from bench.execution.navigation.station_poses import quat_xyzw_to_yaw
from bench.execution.planning.nav_algo import normalize_angle
from bench.tasks.tm01.config import load_task_config
from bench.tasks.tm01.execution.navigation import (
    MAIN_LOOP_TURN_INDICES,
    EpisodePathPoseSampler,
    SampledPathBaseDriver,
    build_navigation_config,
    sample_episode_path_poses,
)
from bench.tasks.tm01.initialization import apply_manual_path_initial_base_pose


def _path() -> Path:
    config = load_task_config()
    path = config.path(config.resources["manual_base_path"])
    assert path is not None
    return path


def _payload() -> dict:
    return json.loads(_path().read_text(encoding="utf-8"))


def test_all_tm01_path_points_are_five_centimeter_episode_circles() -> None:
    samples = _payload()["samples"]
    assert len(samples) == 21
    assert [int(item["index"]) for item in samples] == list(range(21))
    for item in samples:
        circle = item["circle"]
        assert circle["center_xy"] == pytest.approx(item["xy"])
        assert circle["radius_m"] == pytest.approx(0.05)
        assert circle["yaw_half_range_rad"] == pytest.approx(0.0)

    turns = {
        int(item["index"])
        for item in samples
        if isinstance(item.get("relative_turn"), dict)
    }
    assert turns == {*MAIN_LOOP_TURN_INDICES, 2}


def test_sampled_path_is_bounded_reproducible_and_seed_owned() -> None:
    payload = _payload()
    first, first_summaries = sample_episode_path_poses(payload, seed=11)
    repeat, repeat_summaries = sample_episode_path_poses(payload, seed=11)
    other, _ = sample_episode_path_poses(payload, seed=12)

    assert first == repeat
    assert first_summaries == repeat_summaries
    changed = []
    for item in payload["samples"]:
        index = int(item["index"])
        center = np.asarray(item["circle"]["center_xy"], dtype=float)
        xy = np.asarray(first[index].position[:2], dtype=float)
        assert np.linalg.norm(xy - center) <= 0.05 + 1.0e-12
        changed.append(first[index].position != other[index].position)
    assert all(changed)


def test_main_loop_turn_noise_is_bounded_and_closes_heading() -> None:
    payload = _payload()
    poses, summaries = sample_episode_path_poses(payload, seed=37)
    half_range = np.deg2rad(5.0)
    offsets = []
    for index in MAIN_LOOP_TURN_INDICES:
        summary = summaries[index]
        offset = float(summary["sampled_turn_offset"])
        offsets.append(offset)
        assert abs(offset) <= half_range + 1.0e-12

        source_yaw = quat_xyzw_to_yaw(poses[int(summary["source_index"])].quaternion)
        target_yaw = quat_xyzw_to_yaw(poses[index].quaternion)
        actual_turn = normalize_angle(target_yaw - source_yaw)
        expected_turn = normalize_angle(float(summary["nominal_turn_angle"]) + offset)
        assert actual_turn == pytest.approx(expected_turn)

    assert sum(offsets) == pytest.approx(0.0, abs=1.0e-12)
    assert quat_xyzw_to_yaw(poses[18].quaternion) == pytest.approx(
        quat_xyzw_to_yaw(poses[0].quaternion)
    )


def test_station_sampler_and_manual_executor_share_sampled_waypoints() -> None:
    config = load_task_config()
    path = _path()
    navigation = build_navigation_config(
        dict(config.navigation),
        manual_base_path_file=path,
    )
    sampler = EpisodePathPoseSampler.from_file(path, seed=9)
    assert sampler.sample("stove_station") == sampler.route_poses[0]
    assert sampler.sample("drawer_station") == sampler.route_poses[7]

    driver = SampledPathBaseDriver(
        SimpleNamespace(control_timestep=0.05),
        {"robot0_base_pos": np.zeros(3), "robot0_base_quat": [0.0, 0.0, 0.0, 1.0]},
        config=navigation,
        sampler=sampler,
    )
    executor = driver._manual_path_executor()
    for index, pose in sampler.route_poses.items():
        np.testing.assert_allclose(executor.samples[index].xy, pose.position[:2])
        assert executor.samples[index].yaw == pytest.approx(
            quat_xyzw_to_yaw(pose.quaternion)
        )


def test_initial_base_uses_the_same_seeded_path_pose(monkeypatch) -> None:
    path = _path()
    expected = EpisodePathPoseSampler.from_file(path, seed=23).route_poses[0]
    observation = {
        "robot0_base_pos": np.asarray([0.0, 0.0, 0.9]),
        "robot0_base_quat": np.asarray([0.0, 0.0, 0.0, 1.0]),
    }
    env = SimpleNamespace(observation=observation)

    def set_pose(env, *, target_xy, target_yaw):
        env.observation = {
            "robot0_base_pos": np.asarray([*target_xy, 0.9]),
            "robot0_base_quat": np.asarray(
                [0.0, 0.0, np.sin(target_yaw / 2.0), np.cos(target_yaw / 2.0)]
            ),
        }
        return env.observation

    monkeypatch.setattr(
        "bench.tasks.tm01.initialization.set_base_pose_kinematic",
        set_pose,
    )
    updated, summary = apply_manual_path_initial_base_pose(
        env,
        observation,
        path_file=path,
        sample_index=0,
        seed=23,
    )
    np.testing.assert_allclose(updated["robot0_base_pos"][:2], expected.position[:2])
    assert summary["target_yaw"] == pytest.approx(quat_xyzw_to_yaw(expected.quaternion))
    assert summary["episode_seed"] == 23
