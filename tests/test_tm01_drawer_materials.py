from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import asdict

import pytest

from bench.tasks.contracts import EpisodeRequest
from bench.tasks.families.drawer_inspection.materials import (
    DRAWER_MATERIAL_TEXTURES,
    apply_drawer_material,
)
from bench.tasks.tm01.config import PACKAGE_ROOT, load_task_config
from bench.tasks.tm01.domain.contracts import DRAWER_NAMES
from bench.tasks.tm01.domain.episode import resolve_episode_plan
from bench.tasks.tm01.episode import resolve_episode
from bench.tasks.tm01.runtime import build_descriptor


TEXTURE_ROOT = PACKAGE_ROOT.parents[2] / "robocasa/robocasa/models/assets/textures"


def test_tm01_drawer_materials_are_seeded_and_cover_configured_candidates() -> None:
    config = load_task_config()
    plans = [
        resolve_episode_plan(seed=seed, **dict(config.episode))
        for seed in range(200)
    ]
    assert asdict(plans[17]) == asdict(
        resolve_episode_plan(seed=17, **dict(config.episode))
    )
    assert all(set(plan.drawer_material_by_fixture) == set(DRAWER_NAMES) for plan in plans)
    assert {
        material
        for plan in plans
        for material in plan.drawer_material_by_fixture.values()
    } == set(config.episode["drawer_material_candidates"])


def test_material_stream_does_not_change_route_or_drawer_order() -> None:
    config = load_task_config()
    for seed in range(20):
        sampled = resolve_episode_plan(seed=seed, **dict(config.episode))
        original = resolve_episode_plan(
            seed=seed,
            **{**dict(config.episode), "drawer_material_candidates": ["original"]},
        )
        assert sampled.route == original.route
        assert sampled.drawer_order == original.drawer_order


@pytest.mark.parametrize(
    "candidates",
    ([], "flat/white", ["not_a_finish"], ["original", "original"]),
)
def test_invalid_tm01_drawer_material_candidates_are_rejected(candidates) -> None:
    with pytest.raises(ValueError):
        resolve_episode_plan(seed=0, drawer_material_candidates=candidates)


def test_tm01_descriptor_passes_private_material_choices_to_environment() -> None:
    episode = resolve_episode(EpisodeRequest(task_id="TM01", seed=7))
    descriptor = build_descriptor(episode, execute=False)
    assert descriptor.environment.env_kwargs["drawer_material_by_fixture"] == dict(
        episode.facts["drawer_material_by_fixture"]
    )
    assert "drawer_material_by_fixture" not in episode.public


@pytest.mark.parametrize("material_id", tuple(DRAWER_MATERIAL_TEXTURES))
def test_tm01_finish_changes_appearance_only(material_id: str) -> None:
    root = ET.parse(
        PACKAGE_ROOT / "resources/assets/single_tabletop_drawer.xml"
    ).getroot()
    fixture = "tm01_drawer_left_fixture"
    for material in root.findall("./asset/material"):
        material.set("name", f"{fixture}_{material.get('name')}")
    before_worldbody = ET.tostring(root.find("worldbody"))
    before_handle = ET.tostring(
        root.find(f"./asset/material[@name='{fixture}_handle_dark']")
    )
    before = ET.tostring(root)

    apply_drawer_material(
        root,
        fixture_name=fixture,
        material_id=material_id,
        texture_root=TEXTURE_ROOT,
    )

    assert ET.tostring(root.find("worldbody")) == before_worldbody
    assert ET.tostring(
        root.find(f"./asset/material[@name='{fixture}_handle_dark']")
    ) == before_handle
    if material_id == "original":
        assert ET.tostring(root) == before
    else:
        for surface in ("cabinet_body", "drawer_front", "drawer_inside"):
            material = root.find(
                f"./asset/material[@name='{fixture}_{surface}']"
            )
            assert material is not None
            assert material.get("texture") == f"{fixture}_finish_texture"
