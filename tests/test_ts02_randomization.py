from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from bench.execution.contracts import NavigateStage, OperateStage
from bench.execution.navigation.station_poses import StationPoseSampler
from bench.tasks.contracts import EpisodeRequest, ExpertPlanInputs
from bench.tasks.ts02.config import PACKAGE_ROOT, load_task_config
from bench.tasks.ts02.domain.contracts import FIXTURE_NAMES
from bench.tasks.ts02.domain.episode import resolve_episode_plan
from bench.tasks.ts02.drawer_materials import DRAWER_MATERIAL_TEXTURES, apply_drawer_material
from bench.tasks.ts02.episode import resolve_episode
from bench.tasks.ts02.execution.geometry import (
    drawer_pull_target_positions,
    resolve_drawer_motion_frame,
)
from bench.tasks.ts02.execution.manipulation import DrawerPullConfig, _drawer_pull_targets
from bench.tasks.ts02.expert import operations
from bench.tasks.ts02.expert.spec import build_expert_plan
from bench.tasks.ts02.runtime import build_descriptor


TEXTURE_ROOT = PACKAGE_ROOT.parents[2] / "robocasa/robocasa/models/assets/textures"


def _frame(yaw: float = np.pi):
    return resolve_drawer_motion_frame(
        fixture_yaw=yaw,
        current_eef_rotation_world=np.eye(3),
        hook_orientation_blend=0.9,
    )


def test_episode_randomization_is_seeded_bounded_and_covers_candidates() -> None:
    config = load_task_config()
    plans = [resolve_episode_plan(seed=seed, **dict(config.episode)) for seed in range(100)]
    assert asdict(plans[2]) == asdict(resolve_episode_plan(seed=2, **dict(config.episode)))
    assert set(plans[0].drawer_material_by_fixture) == set(FIXTURE_NAMES)
    assert {
        material for plan in plans for material in plan.drawer_material_by_fixture.values()
    } == set(config.episode["drawer_material_candidates"])
    variations = [
        variation for plan in plans for variation in plan.drawer_open_variation_by_fixture.values()
    ]
    assert {v.prehook_variant for v in variations} == set(
        config.episode["prehook_rotation_candidates_deg"]
    )
    assert {v.prehook_rotation_offset_deg for v in variations} == {
        (0.0, 0.0, 0.0)
    }
    assert all(abs(v.handle_side_offset_m) <= 0.015 for v in variations)
    assert any(v.handle_side_offset_m < -0.014 for v in variations)
    assert any(v.handle_side_offset_m > 0.014 for v in variations)


def test_random_streams_do_not_change_target_order_or_other_variations() -> None:
    config = load_task_config()
    for seed in range(20):
        sampled = resolve_episode_plan(seed=seed, **dict(config.episode))
        original = resolve_episode_plan(seed=seed)
        assert sampled.target_fixture == original.target_fixture
        assert sampled.demo_fixture_order == original.demo_fixture_order
        changed = resolve_episode_plan(
            seed=seed, **{**dict(config.episode), "drawer_material_candidates": ["original"]}
        )
        assert sampled.drawer_open_variation_by_fixture == changed.drawer_open_variation_by_fixture
        changed = resolve_episode_plan(
            seed=seed,
            **{**dict(config.episode), "prehook_rotation_candidates_deg": {"baseline": [0, 0, 0]}},
        )
        assert sampled.drawer_material_by_fixture == changed.drawer_material_by_fixture
        assert [
            v.handle_side_offset_m for v in sampled.drawer_open_variation_by_fixture.values()
        ] == [v.handle_side_offset_m for v in changed.drawer_open_variation_by_fixture.values()]


@pytest.mark.parametrize(
    "values",
    [
        {"drawer_material_candidates": []},
        {"drawer_material_candidates": "flat/white"},
        {"drawer_material_candidates": ["not_a_finish"]},
        {"drawer_material_candidates": ["original", "original"]},
        {"prehook_rotation_candidates_deg": {}},
        {"prehook_rotation_candidates_deg": {"bad": [0, 1]}},
        {"prehook_rotation_candidates_deg": {"bad": [0, float("nan"), 0]}},
        {"handle_side_offset_half_range_m": -0.01},
        {"handle_side_offset_half_range_m": 0.05},
        {"handle_side_offset_half_range_m": float("inf")},
    ],
)
def test_invalid_randomization_is_rejected(values) -> None:
    with pytest.raises(ValueError):
        resolve_episode_plan(seed=0, **values)


def test_all_catalog_textures_exist() -> None:
    for path in DRAWER_MATERIAL_TEXTURES.values():
        if path is not None:
            assert (TEXTURE_ROOT / path).is_file(), path


@pytest.mark.parametrize("material_id", tuple(DRAWER_MATERIAL_TEXTURES))
def test_finish_covers_cabinet_and_drawer_body_but_not_handle(material_id: str) -> None:
    root = ET.parse(PACKAGE_ROOT / "resources/assets/single_tabletop_drawer.xml").getroot()
    fixture = "ts02_cabinet_slot_0_fixture"
    for material in root.findall("./asset/material"):
        material.set("name", f"{fixture}_{material.get('name')}")
    before_physics = ET.tostring(root.find("worldbody"))
    before_handle = ET.tostring(root.find(f"./asset/material[@name='{fixture}_handle_dark']"))
    before = ET.tostring(root)
    apply_drawer_material(
        root, fixture_name=fixture, material_id=material_id, texture_root=TEXTURE_ROOT
    )
    assert ET.tostring(root.find("worldbody")) == before_physics
    assert (
        ET.tostring(root.find(f"./asset/material[@name='{fixture}_handle_dark']")) == before_handle
    )
    if material_id == "original":
        assert ET.tostring(root) == before
    else:
        for surface in ("cabinet_body", "drawer_front", "drawer_inside"):
            material = root.find(f"./asset/material[@name='{fixture}_{surface}']")
            assert material is not None
            assert material.get("texture") == f"{fixture}_finish_texture"
        apply_drawer_material(
            root, fixture_name=fixture, material_id=material_id, texture_root=TEXTURE_ROOT
        )
        assert len(root.findall(f"./asset/texture[@name='{fixture}_finish_texture']")) == 1


def test_drawer_station_uses_fixed_point_each_time() -> None:
    config = load_task_config()
    path = config.path(config.resources["station_poses"])
    sampler = StationPoseSampler.from_file(path, seed=2)
    repeated = StationPoseSampler.from_file(path, seed=2)
    fixed_heading = StationPoseSampler.from_file(path, seed=2)
    for _ in range(100):
        fixed_heading.sample("drawer_station")
        assert fixed_heading.last_summary("drawer_station")["target_yaw"] == pytest.approx(-np.pi / 2)
    points = np.asarray([sampler.sample("drawer_station").position[:2] for _ in range(400)])
    same = np.asarray([repeated.sample("drawer_station").position[:2] for _ in range(400)])
    np.testing.assert_array_equal(points, same)
    distances_squared = np.sum((points - [4.85, -1.37]) ** 2, axis=1)
    assert np.all(distances_squared == 0.0)
    assert len(np.unique(points, axis=0)) == 1
    different = StationPoseSampler.from_file(path, seed=3).sample("drawer_station")
    assert different.position[:2] == tuple(points[0])
    summary = sampler.last_summary("drawer_station")
    assert summary["target_yaw"] == pytest.approx(-np.pi / 2)
    assert summary["sample"]["circle"]["yaw_half_range_rad"] == 0.0
    assert sampler.sample("completion_button_station") == sampler.sample(
        "completion_button_station"
    )
    sampler.reset_choices()
    assert sampler.sample("drawer_station").position[:2] == tuple(points[0])
    maximum_distance = (
        max(
            np.linalg.norm(np.asarray(xyz[:2]) - [4.85, -1.37])
            for xyz in config.scene["drawer_world_xyzs"]
        )
        + config.navigation["manual_base_path_xy_tolerance"]
    )
    assert config.operations["drawer_open"]["max_base_distance"] >= maximum_distance


def test_expert_plan_applies_each_fixture_variation_and_sampled_navigation_goal() -> None:
    episode = resolve_episode(EpisodeRequest(task_id="TS02", seed=2))
    descriptor = build_descriptor(episode, execute=False)
    plan = build_expert_plan(ExpertPlanInputs.from_descriptor(descriptor))
    assert descriptor.environment.env_kwargs["drawer_material_by_fixture"] == dict(
        episode.facts["drawer_material_by_fixture"]
    )
    for stage in plan.stages:
        if isinstance(stage, NavigateStage) and stage.target.station_id == "drawer_station":
            assert stage.metadata["use_sampled_terminal_pose"]
        if isinstance(stage, OperateStage) and stage.operation == "open_fixture":
            variation = episode.facts["drawer_open_variation_by_fixture"][stage.target]
            assert stage.metadata["prehook_variant"] == variation["prehook_variant"]
            assert (
                stage.metadata["prehook_rotation_offset_deg"]
                == variation["prehook_rotation_offset_deg"]
            )
            assert stage.metadata["handle_side_offset"] == variation["handle_side_offset_m"]
    other = resolve_episode(EpisodeRequest(task_id="TS02", seed=3))
    assert episode.public["prompt"] == other.public["prompt"]
    assert not set(episode.facts["drawer_material_by_fixture"]) & episode.public.keys()


def test_contact_pull_distance_keeps_configured_margin() -> None:
    config = load_task_config()
    assert config.operations["drawer_open"]["pull_distance"] == pytest.approx(0.18)


def test_navigation_runtime_uses_episode_seed_not_fixed_scene_seed(monkeypatch) -> None:
    seeds = []
    monkeypatch.setattr(
        operations,
        "build_navigation_assembly",
        lambda *args, **kwargs: (
            seeds.append(kwargs["seed"]) or SimpleNamespace(controller=object())
        ),
    )
    monkeypatch.setattr(
        operations,
        "build_manipulation_assembly",
        lambda *args, **kwargs: SimpleNamespace(controller=SimpleNamespace()),
    )
    descriptor = SimpleNamespace(
        navigation=object(),
        episode_plan=SimpleNamespace(seed=17),
        scene=SimpleNamespace(scene_seed=0),
        config=SimpleNamespace(manipulation={}),
        resources={"planner_script": Path("unused")},
        execute=False,
    )
    operations.build_operation_handlers(descriptor, object(), {})
    assert seeds == [17]


@pytest.mark.parametrize("calibrated", [False, True])
@pytest.mark.parametrize("rotation", [(0, 0, 0), (-5, 0, 0), (5, 0, 0), (0, -5, 0), (0, 5, 0)])
def test_handle_offset_applies_to_all_stages_and_live_handle(calibrated, rotation) -> None:
    path = PACKAGE_ROOT / "resources/poses/cabinet_slot_1_hook_pose.json" if calibrated else None
    handle = np.asarray([4.85, -2.162, 1.1043333333333332])
    frame = _frame()
    base, _ = _drawer_pull_targets(handle, frame, DrawerPullConfig(grasp_pose_file=path))
    config = DrawerPullConfig(
        grasp_pose_file=path,
        handle_side_offset=0.02,
        handle_vertical_offset=0.003,
        prehook_rotation_offset_deg=rotation,
    )
    varied, _ = _drawer_pull_targets(handle, frame, config)
    shift = frame.side_direction_world * 0.02 + [0, 0, 0.003]
    for stage in ("prehook", "hook", "pull"):
        np.testing.assert_allclose(varied[stage][:3, 3], base[stage][:3, 3] + shift)
        np.testing.assert_allclose(
            varied[stage][:3, :3].T @ varied[stage][:3, :3], np.eye(3), atol=1e-7
        )
    np.testing.assert_array_equal(varied["hook"][:3, :3], base["hook"][:3, :3])
    np.testing.assert_array_equal(varied["pull"][:3, :3], base["pull"][:3, :3])
    if any(rotation):
        assert not np.allclose(varied["prehook"][:3, :3], base["prehook"][:3, :3])
    live_handle = handle + frame.open_direction_world * 0.05
    live, _ = _drawer_pull_targets(live_handle, frame, config)
    np.testing.assert_allclose(live["pull"][:3, 3], varied["pull"][:3, 3] + live_handle - handle)


@pytest.mark.parametrize("yaw", [0.0, np.pi / 2, np.pi])
def test_procedural_offsets_follow_handle_axis_not_world_axis(yaw) -> None:
    frame = _frame(yaw)
    kwargs = dict(prehook_distance=0.1, hook_offset=0.005, pull_distance=0.18)
    base = drawer_pull_target_positions(np.zeros(3), frame, **kwargs)
    varied = drawer_pull_target_positions(np.zeros(3), frame, handle_side_offset=0.02, **kwargs)
    for stage in base:
        np.testing.assert_allclose(
            varied[stage] - base[stage], frame.side_direction_world * 0.02, atol=1e-12
        )
