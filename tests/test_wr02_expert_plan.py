from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from bench.execution.contracts import OperateStage
from bench.tasks import TaskCatalog
from bench.tasks import resolve_builtin_task_plugin
from bench.tasks.contracts import EpisodeRequest, ExpertPlanInputs


def build_wr02_plan():
    episode = TaskCatalog.resolve(
        EpisodeRequest(task_id="WR-02", seed=0, allow_partial_calibration=True)
    )
    api = resolve_builtin_task_plugin("WR-02").load_runtime_api()
    descriptor = api.build_descriptor(episode, False)
    return episode, TaskCatalog.get("WR-02").build_expert_plan(
        ExpertPlanInputs.from_descriptor(descriptor)
    )
from bench.tasks.wr02.execution.handlers import _resolve_symbolic_target


def test_wr02_shelf_pick_uses_object_specific_calibration() -> None:
    episode, plan = build_wr02_plan()

    shelf_picks = {
        str(stage.target): stage
        for stage in plan.stages
        if isinstance(stage, OperateStage)
        and stage.operation == "pick"
        and stage.name in {"pick_ketchup", "pick_mustard"}
    }

    assert set(shelf_picks) == {"ketchup", "mustard"}
    for condiment, stage in shelf_picks.items():
        metadata = dict(stage.metadata)
        assert metadata["orientation_mode"] == "side-from-base"
        assert Path(metadata["manual_grasp_pose_file"]) == Path(
            episode.resources[f"{condiment}_shelf_grasp"]
        )


def test_wr02_sink_pick_reuses_reference_shelf_grasp_calibration() -> None:
    episode, plan = build_wr02_plan()

    sink_picks = [
        stage
        for stage in plan.stages
        if isinstance(stage, OperateStage)
        and stage.operation == "pick"
        and stage.name.endswith("_from_sink")
    ]

    assert len(sink_picks) == 2
    for stage in sink_picks:
        assert stage.metadata["manual_grasp_pose_file"].endswith(
            f"shelf_{stage.target}_grasp.json"
        )
        assert stage.metadata["manual_grasp_pregrasp_distance"] == 0.04


def test_wr02_plan_injects_sink_and_home_place_calibrations() -> None:
    episode, plan = build_wr02_plan()

    sink_place = next(
        stage
        for stage in plan.stages
        if isinstance(stage, OperateStage)
        and stage.name == "place_ketchup_at_sink_side"
    )
    home_place = next(
        stage
        for stage in plan.stages
        if isinstance(stage, OperateStage) and stage.name == "return_ketchup_home"
    )

    assert sink_place.metadata["target_pos"] == "ketchup:sink_left"
    assert sink_place.metadata["manual_place_pose_file"].endswith(
        "sink_left_place_tcp.json"
    )
    assert home_place.metadata["target_pos"] == "ketchup:shelf_upper"
    assert home_place.metadata["object_relative_place_grasp_pose_file"].endswith(
        "shelf_ketchup_grasp.json"
    )


def test_wr02_symbolic_place_targets_resolve_like_reference_workflow() -> None:
    targets = SimpleNamespace(
        use_pose_by_condiment={"ketchup": (3.2, -0.8, 0.68)},
        return_place_sampling={
            "ketchup": {"sampled_target_xy": (2.1, -0.31)},
        },
        original_home_pose_by_condiment={"ketchup": (2.0, -0.3, 1.48)},
    )

    assert _resolve_symbolic_target(
        object_name="ketchup", target_spec="ketchup:sink_left", targets=targets
    ) == (3.2, -0.8, 0.68)
    assert _resolve_symbolic_target(
        object_name="ketchup", target_spec="ketchup:shelf_upper", targets=targets
    ) == (2.1, -0.31, 1.48)
