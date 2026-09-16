from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

from bench.tasks import TaskCatalog, resolve_builtin_task_plugin
from bench.tasks.contracts import EpisodeRequest
from bench.tasks.wa01.materials import (
    COFFEE_MATERIAL_PROFILES,
    SUGAR_CUBE_MATERIAL_PROFILES,
    apply_coffee_material,
    sample_coffee_materials,
    sample_sugar_cube_material,
)


ROOT = Path(__file__).resolve().parents[1]
ASSETS_ROOT = ROOT / "robocasa/robocasa/models/assets"


def _descriptor(seed: int):
    episode = TaskCatalog.resolve(EpisodeRequest(task_id="WA-01", seed=seed))
    api = resolve_builtin_task_plugin("WA-01").load_runtime_api()
    return api.build_descriptor(episode, False)


def test_wa01_material_replaces_only_the_selected_cup_binding() -> None:
    root = ET.fromstring(
        """
        <mujoco>
          <asset>
            <texture name="coffee_A_image0" type="2d" file="original-a.png"/>
            <texture name="coffee_B_image0" type="2d" file="original-b.png"/>
            <material name="coffee_A_lambert3SG" texture="coffee_A_image0"/>
            <material name="coffee_B_lambert3SG" texture="coffee_B_image0"/>
          </asset>
          <worldbody/>
        </mujoco>
        """
    )
    apply_coffee_material(
        root,
        coffee_id="coffee_A",
        material_id="steel",
        assets_root=ASSETS_ROOT,
    )
    selected = root.find("./asset/material[@name='coffee_A_lambert3SG']")
    untouched = root.find("./asset/material[@name='coffee_B_lambert3SG']")
    texture = root.find("./asset/texture[@name='coffee_A_wa01_material_texture']")
    assert texture is not None
    assert selected is not None and selected.get("texture") == texture.get("name")
    assert selected.get("specular") == "0.65"
    assert texture.get("file", "").endswith("steel-brushed.png")
    assert untouched is not None and untouched.get("texture") == "coffee_B_image0"


def test_wa01_cup_materials_are_shared_within_an_episode_and_seeded() -> None:
    descriptors = [_descriptor(seed) for seed in range(24)]
    shared = {
        next(iter(descriptor.episode_plan.coffee_material_by_id.values()))
        for descriptor in descriptors
    }
    assert len(shared) > 1
    for seed, descriptor in enumerate(descriptors):
        drawn = descriptor.episode_plan.coffee_material_by_id
        assert drawn == _descriptor(seed).episode_plan.coffee_material_by_id
        assert set(drawn) == {"coffee_A", "coffee_B"}
        assert drawn["coffee_A"] == drawn["coffee_B"]
        assert set(drawn.values()) <= set(COFFEE_MATERIAL_PROFILES)
        assert drawn == sample_coffee_materials(seed, tuple(COFFEE_MATERIAL_PROFILES))


def test_wa01_sugar_cube_material_is_seeded_from_the_ts03_catalog() -> None:
    drawn = {
        seed: _descriptor(seed).episode_plan.sugar_cube_material for seed in range(24)
    }
    assert set(drawn.values()) <= set(SUGAR_CUBE_MATERIAL_PROFILES)
    assert len(set(drawn.values())) > 1
    for seed, material in drawn.items():
        assert material == sample_sugar_cube_material(
            seed, tuple(SUGAR_CUBE_MATERIAL_PROFILES)
        )
        assert material == _descriptor(seed).episode_plan.sugar_cube_material
        assert (
            _descriptor(seed).environment.env_kwargs["sugar_cube_material"] == material
        )
