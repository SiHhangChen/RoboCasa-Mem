from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from bench.execution.navigation.contracts import Pose
from bench.tasks.wr08 import initialization
from bench.tasks.wr08.config import load_task_config, resolve_robot_base_config
from bench.tasks.wr08.domain.contracts import (
    CARTON_ASSET_BY_ITEM,
    CARTON_TYPE_BY_ITEM,
    ITEM_IDS,
    MILK_CARTON_MATERIAL_PROFILES,
    MILK_CARTON_MATERIALS,
    MILK_CARTON_TYPES,
    RobotBaseMode,
)
from bench.tasks.wr08.domain.episode import resolve_episode_plan
from bench.tasks.wr08.environment import WR08MatchMilkCartonsToColoredSugarPlates
from bench.tasks.wr08.execution.navigation import SourcePlateStationSampler


def test_default_config_enables_reset_circle_and_material_pool() -> None:
    config = load_task_config()

    assert config.initial_state.robot_base.mode is RobotBaseMode.RESET_CIRCLE
    assert config.initial_state.robot_base.radius_m == pytest.approx(0.05)
    assert tuple(config.episode["carton_type_candidates"]) == MILK_CARTON_TYPES
    assert tuple(config.episode["carton_material_candidates"]) == MILK_CARTON_MATERIALS


def test_episode_samples_independent_reproducible_carton_materials() -> None:
    plans = [resolve_episode_plan(seed=seed) for seed in range(64)]

    assert all(plan == resolve_episode_plan(seed=plan.seed) for plan in plans)
    assert all(dict(plan.carton_type_by_item) == CARTON_TYPE_BY_ITEM for plan in plans)
    assert all(dict(plan.carton_asset_by_item) == CARTON_ASSET_BY_ITEM for plan in plans)
    assert {
        material
        for plan in plans
        for material in plan.carton_material_by_item.values()
    } == set(MILK_CARTON_MATERIALS)
    assert any(
        len(set(plan.carton_material_by_item.values())) == len(ITEM_IDS)
        for plan in plans
    )


@pytest.mark.parametrize(
    "settings",
    [
        {"carton_type_candidates": ["milk_0"]},
        {"carton_type_candidates": ["milk_0", "unknown"]},
        {"carton_material_candidates": []},
        {"carton_material_candidates": ["matte", "matte"]},
        {"carton_material_candidates": ["unknown"]},
    ],
)
def test_episode_rejects_invalid_carton_candidate_pools(settings) -> None:
    with pytest.raises(ValueError):
        resolve_episode_plan(seed=0, **settings)


@pytest.mark.parametrize("radius", [-0.1, float("nan"), float("inf")])
def test_reset_circle_rejects_invalid_radius(radius: float) -> None:
    with pytest.raises(ValueError):
        resolve_robot_base_config(mode="reset_circle", radius_m=radius)


def test_reset_circle_samples_once_from_reset_pose(monkeypatch) -> None:
    plan = resolve_episode_plan(seed=17)
    robot_base = resolve_robot_base_config(mode="reset_circle", radius_m=0.05)
    configured: list[object] = []
    env = SimpleNamespace(configure_episode=configured.append)
    calls: dict[str, object] = {}

    monkeypatch.setattr(initialization, "base_yaw", lambda obs: 0.75)

    def sample(center, radius, *, seed, salt):
        calls["sample"] = (np.asarray(center).copy(), radius, seed, salt)
        return np.asarray([1.02, 1.97], dtype=float)

    def move(_env, *, target_xy, target_yaw):
        calls["move"] = (np.asarray(target_xy).copy(), target_yaw)
        return {
            "robot0_base_pos": np.asarray([*target_xy, 0.7], dtype=float),
            "robot0_base_quat": np.asarray([0.0, 0.0, 0.0, 1.0]),
        }

    monkeypatch.setattr(initialization, "sample_uniform_disk", sample)
    monkeypatch.setattr(initialization, "set_base_pose_kinematic", move)

    _, facts = initialization.initialize_episode_state(
        env,
        {
            "robot0_base_pos": np.asarray([1.0, 2.0, 0.7]),
            "robot0_base_quat": np.asarray([0.0, 0.0, 0.0, 1.0]),
        },
        episode=plan,
        robot_base=robot_base,
        objects={},
    )

    assert configured == [plan]
    center, radius, seed, salt = calls["sample"]
    np.testing.assert_allclose(center, [1.0, 2.0])
    assert (radius, seed, salt) == (0.05, 17, 0x57523038)
    np.testing.assert_allclose(calls["move"][0], [1.02, 1.97])
    assert calls["move"][1] == pytest.approx(0.75)
    assert facts["initial_base_sampling"] == {
        "policy": "reset_circle_uniform_area",
        "center_xy": [1.0, 2.0],
        "radius_m": 0.05,
        "sampled_xy": [1.02, 1.97],
        "offset_xy": [0.020000000000000018, -0.030000000000000027],
        "preserved_yaw": 0.75,
    }


def test_carton_material_updates_texture_and_optical_properties() -> None:
    raw_model = SimpleNamespace(
        mat_texid=np.asarray([[0]], dtype=int),
        tex_adr=np.asarray([0], dtype=int),
        tex_width=np.asarray([2], dtype=int),
        tex_height=np.asarray([1], dtype=int),
        tex_nchannel=np.asarray([3], dtype=int),
        tex_data=np.asarray([0, 107, 14, 255, 255, 255], dtype=np.uint8),
    )
    model = SimpleNamespace(
        body_parentid=np.asarray([0, 0], dtype=int),
        geom_bodyid=np.asarray([1], dtype=int),
        geom_rgba=np.asarray([[0.5, 0.5, 0.5, 1.0]], dtype=float),
        geom_matid=np.asarray([0], dtype=int),
        mat_specular=np.asarray([0.0], dtype=float),
        mat_shininess=np.asarray([0.0], dtype=float),
        mat_reflectance=np.asarray([0.0], dtype=float),
        _model=raw_model,
    )
    env = object.__new__(WR08MatchMilkCartonsToColoredSugarPlates)
    env.sim = SimpleNamespace(model=model, _render_context_offscreen=None)
    env.obj_body_id = {"boxed_food_a": 1}
    profile = MILK_CARTON_MATERIAL_PROFILES["foil"]

    env._apply_carton_material("boxed_food_a", profile)

    np.testing.assert_allclose(model.geom_rgba[0], [1.0, 1.0, 1.0, 1.0])
    assert model.mat_specular[0] == pytest.approx(profile["specular"])
    assert model.mat_shininess[0] == pytest.approx(profile["shininess"])
    assert model.mat_reflectance[0] == pytest.approx(profile["reflectance"])
    np.testing.assert_array_equal(raw_model.tex_data[:3], profile["texture_rgb"])
    np.testing.assert_array_equal(raw_model.tex_data[3:], [255, 255, 255])


def test_source_circle_pose_is_reused_until_sampler_reset() -> None:
    first = Pose((1.0, 2.0, 0.7), (0.0, 0.0, 0.0, 1.0), 0.0)
    second = Pose((3.0, 4.0, 0.7), (0.0, 0.0, 0.0, 1.0), 0.0)

    class Delegate:
        def __init__(self) -> None:
            self.calls = 0

        def station_id(self, station_name, *, purpose=None):
            del station_name, purpose
            return "counter_pick_station"

        def sample(self, station_name, *, purpose=None):
            del station_name, purpose
            self.calls += 1
            return first if self.calls == 1 else second

        def last_summary(self, station_name):
            del station_name
            return {"sample": {"circle": {"radius_m": 0.05}}}

        def reset_choices(self):
            return None

    delegate = Delegate()
    sampler = SourcePlateStationSampler(delegate, SimpleNamespace())

    assert sampler.sample("source_left", purpose="source_observe") is first
    assert sampler.sample("source_left", purpose="source_pick") is first
    assert delegate.calls == 1
    sampler.reset_choices()
    assert sampler.sample("source_left", purpose="source_pick") is second
    assert delegate.calls == 2
