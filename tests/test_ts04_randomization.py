from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from bench.execution.contracts import NavigateStage, OperateStage
from bench.execution.navigation.manual_base_path import load_manual_path_samples
from bench.execution.navigation.station_poses import StationPoseSampler
from bench.tasks.contracts import EpisodeRequest, ExpertPlanInputs
from bench.tasks.families.drawer_inspection.materials import (
    DRAWER_MATERIAL_TEXTURES,
    apply_drawer_material,
)
from bench.tasks.ts04.config import PACKAGE_ROOT, load_task_config
from bench.tasks.ts04.domain.contracts import FIXTURE_NAMES
from bench.tasks.ts04.domain.episode import resolve_episode_plan
from bench.tasks.ts04.episode import resolve_episode
from bench.tasks.ts04.execution.geometry import (
    drawer_pull_target_positions,
    resolve_drawer_motion_frame,
)
from bench.tasks.ts04.execution.manipulation import DrawerPullConfig, _drawer_pull_targets
from bench.tasks.ts04.expert import operations
from bench.tasks.ts04.expert.spec import build_expert_plan
from bench.tasks.ts04.runtime import build_descriptor


TEXTURE_ROOT = PACKAGE_ROOT.parents[2] / "robocasa/robocasa/models/assets/textures"


def _frame(yaw: float = np.pi):
    return resolve_drawer_motion_frame(
        fixture_yaw=yaw,
        current_eef_rotation_world=np.eye(3),
        hook_orientation_blend=0.9,
    )


def test_episode_randomization_is_seeded_bounded_and_covers_candidates() -> None:
    config = load_task_config()
    plans = [resolve_episode_plan(seed=seed, **dict(config.episode)) for seed in range(200)]
    assert asdict(plans[2]) == asdict(resolve_episode_plan(seed=2, **dict(config.episode)))
    assert set(plans[0].drawer_material_by_fixture) == set(FIXTURE_NAMES)
    assert {
        material for plan in plans for material in plan.drawer_material_by_fixture.values()
    } == set(config.episode["drawer_material_candidates"])
    variations = [
        variation
        for plan in plans
        for variation in plan.drawer_open_variation_by_fixture.values()
    ]
    assert {variation.prehook_variant for variation in variations} == set(
        config.episode["prehook_rotation_candidates_deg"]
    )
    assert all(abs(variation.handle_side_offset_m) <= 0.015 for variation in variations)
    assert any(variation.handle_side_offset_m < -0.014 for variation in variations)
    assert any(variation.handle_side_offset_m > 0.014 for variation in variations)


def test_new_random_streams_preserve_target_order_and_each_other() -> None:
    config = load_task_config()
    for seed in range(30):
        sampled = resolve_episode_plan(seed=seed, **dict(config.episode))
        original = resolve_episode_plan(seed=seed)
        assert sampled.target_fixture == original.target_fixture
        assert sampled.demo_fixture_order == original.demo_fixture_order
        changed_material = resolve_episode_plan(
            seed=seed,
            **{**dict(config.episode), "drawer_material_candidates": ["original"]},
        )
        assert (
            sampled.drawer_open_variation_by_fixture
            == changed_material.drawer_open_variation_by_fixture
        )
        changed_pose = resolve_episode_plan(
            seed=seed,
            **{
                **dict(config.episode),
                "prehook_rotation_candidates_deg": {"baseline": [0, 0, 0]},
            },
        )
        assert sampled.drawer_material_by_fixture == changed_pose.drawer_material_by_fixture
        assert [
            value.handle_side_offset_m
            for value in sampled.drawer_open_variation_by_fixture.values()
        ] == [
            value.handle_side_offset_m
            for value in changed_pose.drawer_open_variation_by_fixture.values()
        ]


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
    fixture = "ts04_cabinet_slot_0_fixture"
    for material in root.findall("./asset/material"):
        material.set("name", f"{fixture}_{material.get('name')}")
    before_physics = ET.tostring(root.find("worldbody"))
    before_handle = ET.tostring(root.find(f"./asset/material[@name='{fixture}_handle_dark']"))
    before = ET.tostring(root)
    apply_drawer_material(
        root, fixture_name=fixture, material_id=material_id, texture_root=TEXTURE_ROOT
    )
    assert ET.tostring(root.find("worldbody")) == before_physics
    assert ET.tostring(root.find(f"./asset/material[@name='{fixture}_handle_dark']")) == before_handle
    if material_id == "original":
        assert ET.tostring(root) == before
    else:
        for surface in ("cabinet_body", "drawer_front", "drawer_inside"):
            material = root.find(f"./asset/material[@name='{fixture}_{surface}']")
            assert material is not None
            assert material.get("texture") == f"{fixture}_finish_texture"


def test_drawer_station_samples_15cm_disk_and_yaw() -> None:
    config = load_task_config()
    path = config.path(config.resources["station_poses"])
    sampler = StationPoseSampler.from_file(path, seed=2)
    repeated = StationPoseSampler.from_file(path, seed=2)
    points = np.asarray([sampler.sample("drawer_station").position[:2] for _ in range(400)])
    same = np.asarray([repeated.sample("drawer_station").position[:2] for _ in range(400)])
    np.testing.assert_array_equal(points, same)
    distances_squared = np.sum((points - [4.85, -1.37]) ** 2, axis=1)
    assert np.all(distances_squared <= 0.075**2)
    assert np.mean(distances_squared) / 0.075**2 == pytest.approx(0.5, abs=0.08)
    assert len(np.unique(points, axis=0)) == 400
    summary = sampler.last_summary("drawer_station")
    assert abs(summary["target_yaw"] + np.pi / 2) <= np.deg2rad(5.0)
    assert summary["sample"]["circle"]["yaw_half_range_rad"] == pytest.approx(
        np.deg2rad(5.0)
    )


def test_three_manual_position_points_sample_10cm_disks() -> None:
    config = load_task_config()
    path = config.path(config.resources["manual_base_path"])
    centers = {0: (3.17, -1.166), 1: (4.332, -1.171), 3: (3.85, -1.37)}
    for seed in range(200):
        samples = {sample.index: sample for sample in load_manual_path_samples(path, seed=seed)}
        for index, center in centers.items():
            assert np.linalg.norm(samples[index].xy - center) <= 0.05 + 1e-12
    first = load_manual_path_samples(path, seed=7)
    second = load_manual_path_samples(path, seed=7)
    np.testing.assert_array_equal(
        np.asarray([sample.xy for sample in first]),
        np.asarray([sample.xy for sample in second]),
    )


def test_expert_plan_applies_variations_and_sampled_navigation_goal() -> None:
    episode = resolve_episode(EpisodeRequest(task_id="TS04", seed=2))
    assert episode.task_version == "1.1.0"
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
            assert stage.metadata["prehook_rotation_offset_deg"] == variation[
                "prehook_rotation_offset_deg"
            ]
            assert stage.metadata["handle_side_offset"] == variation["handle_side_offset_m"]
    other = resolve_episode(EpisodeRequest(task_id="TS04", seed=3))
    assert episode.public["prompt"] == other.public["prompt"]
    assert not set(episode.facts["drawer_material_by_fixture"]) & episode.public.keys()


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
@pytest.mark.parametrize("rotation", [(0, 0, 0), (0, -5, 0), (0, 3, 0)])
def test_handle_offset_applies_to_all_stages_and_live_handle(calibrated, rotation) -> None:
    path = PACKAGE_ROOT / "resources/poses/cabinet_slot_1_hook_pose.json" if calibrated else None
    handle = np.asarray([4.85, -2.162, 1.1043333333333332])
    frame = _frame()
    base, _ = _drawer_pull_targets(handle, frame, DrawerPullConfig(grasp_pose_file=path))
    config = DrawerPullConfig(
        grasp_pose_file=path,
        handle_side_offset=0.02,
        prehook_rotation_offset_deg=rotation,
    )
    varied, _ = _drawer_pull_targets(handle, frame, config)
    expected = frame.side_direction_world * 0.02
    for stage in ("prehook", "hook", "pull"):
        np.testing.assert_allclose(
            varied[stage][:3, 3] - base[stage][:3, 3], expected, atol=1e-12
        )
    np.testing.assert_allclose(varied["hook"][:3, :3], base["hook"][:3, :3])
    np.testing.assert_allclose(varied["pull"][:3, :3], base["pull"][:3, :3])
    live, _ = _drawer_pull_targets(handle + [0, 0.04, 0], frame, config)
    np.testing.assert_allclose(live["pull"][:3, 3] - varied["pull"][:3, 3], [0, 0.04, 0])


def test_procedural_target_offset_moves_entire_approach() -> None:
    frame = _frame()
    kwargs = dict(prehook_distance=0.125, hook_offset=0.005, pull_distance=0.18)
    base = drawer_pull_target_positions(np.zeros(3), frame, **kwargs)
    varied = drawer_pull_target_positions(
        np.zeros(3), frame, handle_side_offset=0.02, **kwargs
    )
    expected = frame.side_direction_world * 0.02
    for stage in ("prehook", "hook", "pull"):
        np.testing.assert_allclose(varied[stage] - base[stage], expected, atol=1e-12)


def test_contact_pull_distance_keeps_configured_margin() -> None:
    config = load_task_config()
    assert config.operations["drawer_open"]["pull_distance"] == pytest.approx(0.18)
    assert config.operations["drawer_open"]["max_base_distance"] == pytest.approx(1.15)
