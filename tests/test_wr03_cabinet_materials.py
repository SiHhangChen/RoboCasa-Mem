from pathlib import Path
import xml.etree.ElementTree as ET

import pytest

from bench.tasks.wr03.cabinet_materials import (
    CABINET_MATERIAL_TEXTURES,
    apply_cabinet_material,
    normalize_cabinet_material_candidates,
    sample_cabinet_material,
)
from bench.tasks.wr03.config import load_task_config
from bench.tasks.wr03.domain.episode import resolve_episode_plan


def test_sampling_is_deterministic_and_covers_candidates() -> None:
    candidates = tuple(CABINET_MATERIAL_TEXTURES)
    first = [sample_cabinet_material(seed, candidates) for seed in range(100)]
    second = [sample_cabinet_material(seed, candidates) for seed in range(100)]
    assert first == second
    assert set(first) == set(candidates)


def test_material_stream_does_not_change_other_episode_decisions() -> None:
    values = dict(load_task_config().episode)
    for seed in range(20):
        default = resolve_episode_plan(seed=seed, **values)
        original = resolve_episode_plan(
            seed=seed, **{**values, "cabinet_material_candidates": ["original"]}
        )
        assert default.stored_level == original.stored_level
        assert default.close_order == original.close_order


@pytest.mark.parametrize("value", [[], "original", ["original", "original"], ["bad"]])
def test_invalid_candidate_config_is_rejected(value) -> None:
    with pytest.raises(ValueError):
        normalize_cabinet_material_candidates(value)


def test_texture_files_exist() -> None:
    root = Path("robocasa/robocasa/models/assets/textures")
    for relative_path in CABINET_MATERIAL_TEXTURES.values():
        if relative_path is not None:
            assert (root / relative_path).is_file()


def test_xml_replacement_changes_only_fixture_wood_materials(tmp_path: Path) -> None:
    texture_root = tmp_path / "textures"
    (texture_root / "wood").mkdir(parents=True)
    (texture_root / "wood" / "light_wood_planks.png").touch()
    root = ET.fromstring(
        """<mujoco><asset>
        <material name="lower_wr03_wood" texture="lower_old"/>
        <material name="upper_wr03_wood" texture="upper_old"/>
        <material name="lower_wr03_metal" texture="metal"/>
        </asset><worldbody><geom name="collision" material="collision"/></worldbody></mujoco>"""
    )
    worldbody_before = ET.tostring(root.find("worldbody"))
    apply_cabinet_material(
        root,
        fixture_names=("lower", "upper"),
        material_id="light_wood",
        texture_root=texture_root,
    )
    lower = root.find("asset/material[@name='lower_wr03_wood']")
    upper = root.find("asset/material[@name='upper_wr03_wood']")
    assert lower.get("texture") == "lower_finish_texture"
    assert upper.get("texture") == "upper_finish_texture"
    assert root.find("asset/material[@name='lower_wr03_metal']").get("texture") == "metal"
    assert ET.tostring(root.find("worldbody")) == worldbody_before
