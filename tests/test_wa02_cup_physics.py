from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest


def test_wa02_uses_same_cup_asset_and_scale_as_wa05() -> None:
    pytest.importorskip("mujoco")
    from bench.tasks.wa02.domain.geometry import (
        resolve_cup_region_constraints,
        resolve_cup_sampling_regions,
    )
    from bench.tasks.wa02.environment import (
        DEFAULT_CUP_REGION_CONSTRAINTS,
        DEFAULT_CUP_SAMPLING_REGIONS,
        IDENTICAL_WATER_CUP_XML,
        WA02ServeHotColdWaterCups,
        WATER_CUP_OBJECT_SCALE,
    )
    from bench.tasks.wa05.environment import (
        IDENTICAL_WATER_CUP_XML as WA05_CUP_XML,
        WATER_CUP_OBJECT_SCALE as WA05_CUP_SCALE,
    )

    environment = WA02ServeHotColdWaterCups.__new__(WA02ServeHotColdWaterCups)
    environment._cup_counter_reset_region = {"offset": (0.0, 0.0, 0.0)}
    environment._cup_sampling_regions = resolve_cup_sampling_regions(
        DEFAULT_CUP_SAMPLING_REGIONS
    )
    environment._cup_region_constraints = resolve_cup_region_constraints(
        DEFAULT_CUP_REGION_CONSTRAINTS
    )
    environment.counter = SimpleNamespace(rot=0.0)

    configs = environment._get_obj_cfgs()

    assert IDENTICAL_WATER_CUP_XML == WA05_CUP_XML
    assert WATER_CUP_OBJECT_SCALE == WA05_CUP_SCALE
    assert all(config["obj_groups"] == WA05_CUP_XML for config in configs)
    assert all(config["object_scale"] == WA05_CUP_SCALE for config in configs)


def test_wa02_grasp_candidates_match_wa05_geometry() -> None:
    root = Path(__file__).parents[1] / "bench/tasks"
    wa02 = json.loads(
        (root / "wa02/resources/poses/water_cup_grasp_candidates.json").read_text()
    )
    wa05 = json.loads(
        (root / "wa05/resources/poses/water_cup_grasp_candidates.json").read_text()
    )

    assert wa02["pregrasp_offset_tcp"] == wa05["pregrasp_offset_tcp"]
    assert wa02["candidates"] == wa05["candidates"]


def test_wa02_runtime_passes_wa05_gripper_physics() -> None:
    from bench.tasks import TaskCatalog, resolve_builtin_task_plugin
    from bench.tasks.contracts import EpisodeRequest

    episode = TaskCatalog.resolve(EpisodeRequest(task_id="WA-02", seed=0))
    api = resolve_builtin_task_plugin("WA-02").load_runtime_api()
    descriptor = api.build_descriptor(episode, False)

    assert descriptor.environment.env_kwargs["gripper_force_limit"] == pytest.approx(20.0)
    assert descriptor.environment.env_kwargs["gripper_pad_friction"] == pytest.approx(
        (2.0, 0.05, 0.0001)
    )
    assert descriptor.config.operations["service_place"][
        "release_separation_assist_distance"
    ] == pytest.approx(0.0)


def test_wa02_manipulation_does_not_install_object_follow() -> None:
    source = (
        Path(__file__).parents[1]
        / "bench/tasks/wa02/execution/manipulation.py"
    ).read_text(encoding="utf-8")

    assert "_install_held_cup_follow_assist" not in source
    assert "env.step =" not in source
