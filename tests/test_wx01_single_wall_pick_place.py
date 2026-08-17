from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from bench.execution.contracts import OperateStage
from bench.execution.planning.model_alignment import quat_xyzw_to_mat
from bench.tasks import TaskCatalog, resolve_builtin_task_plugin
from bench.tasks.contracts import EpisodeRequest, ExpertPlanInputs
from bench.tasks.wx01.config import load_task_config


def _wx01_descriptor_and_plan():
    episode = TaskCatalog.resolve(
        EpisodeRequest(task_id="WX-01", seed=0, allow_partial_calibration=True)
    )
    api = resolve_builtin_task_plugin("WX-01").load_runtime_api()
    descriptor = api.build_descriptor(episode, False)
    plan = TaskCatalog.get("WX-01").build_expert_plan(
        ExpertPlanInputs.from_descriptor(descriptor)
    )
    return descriptor, plan


def test_wx01_uses_large_cup_5_at_scale_point_9() -> None:
    config = load_task_config()
    environment_source = (
        Path(__file__).parents[1] / "bench/tasks/wx01/environment.py"
    ).read_text(encoding="utf-8")

    assert 'objaverse/cup/cup_5/model.xml"' in environment_source
    assert config.scene["blocking_object_scale"] == 0.9


def test_wx01_owns_topdown_single_wall_grasp_candidates() -> None:
    config = load_task_config()
    source = config.path(config.resources["blocking_cup_grasp_candidates"])

    assert source is not None
    assert source.is_relative_to(config.source_path.parent)
    assert source.name == "blocking_cup_single_wall_grasp_candidates.json"
    payload = json.loads(source.read_text(encoding="utf-8"))
    assert payload["task"] == "WX-01"
    assert payload["asset"] == "robocasa cup_5"
    assert payload["grasp_strategy"] == "single-wall-topdown"
    assert payload["pregrasp_offset_tcp"] == [0.0, 0.0, -0.12]
    assert [item["wall_azimuth_deg"] for item in payload["candidates"]] == [
        0.0,
        180.0,
        95.0,
        -95.0,
    ]

    for candidate in payload["candidates"]:
        pose = candidate["eef_site_in_object"]
        position = np.asarray(pose["p"], dtype=float)
        rotation = quat_xyzw_to_mat(np.asarray(pose["q_xyzw"], dtype=float))
        radial = position[:2] / np.linalg.norm(position[:2])

        assert 0.03 < np.linalg.norm(position[:2]) < 0.04
        assert 0.0 <= position[2] <= 0.02
        # TCP Z axis points straight down (into the cup mouth).
        assert rotation[2, 2] < -0.995
        # TCP -X axis points into the cup wall (radial), within a few degrees.
        tcp_radial = -rotation[:, 0][:2] / np.linalg.norm(rotation[:, 0][:2])
        assert float(abs(np.dot(tcp_radial, radial))) > 0.99


def test_wx01_expert_uses_single_wall_pick_and_vertical_place() -> None:
    descriptor, plan = _wx01_descriptor_and_plan()
    pick = next(
        stage
        for stage in plan.stages
        if isinstance(stage, OperateStage)
        and stage.name == "pick_displaced_blocking_object"
    )
    place = next(
        stage
        for stage in plan.stages
        if isinstance(stage, OperateStage)
        and stage.name == "return_blocking_object_to_remembered_origin"
    )

    assert pick.metadata["grasp_strategy"] == "single-wall-topdown"
    assert pick.metadata["lift_height"] == 0.12
    assert "orientation_mode" not in pick.metadata
    assert "retreat_to_pregrasp_after_grasp" not in pick.metadata
    assert place.metadata["placement_object_upright"] is True
    assert place.metadata["placement_approach_angles_deg"] == [0.0]
    assert place.metadata["placement_entry_standoff"] == 0.0
    assert place.metadata["placement_exit_standoff"] == 0.0
    assert place.metadata["placement_entry_z_offset"] > 0.0
    assert place.metadata["placement_exit_z_offset"] > 0.0
    assert place.metadata["post_release_clearance_distance"] == 0.0
    assert place.metadata["require_pre_release_check"] is True
    assert place.metadata["pre_release_xy_tolerance"] < float(
        descriptor.config.semantics["origin_threshold"]
    )


def test_wx01_cleaning_tool_keeps_its_existing_topdown_profile() -> None:
    config = load_task_config()

    assert config.manipulation["pick_place"]["orientation_mode"] == "object-y-topdown"
    assert config.operations["blocking_pick"]["grasp_strategy"] == (
        "single-wall-topdown"
    )
