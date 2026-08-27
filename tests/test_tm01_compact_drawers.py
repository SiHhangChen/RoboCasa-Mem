from __future__ import annotations

import json

import pytest

from bench.execution.contracts import NavigateStage, OperateStage
from bench.tasks import TaskCatalog, resolve_builtin_task_plugin
from bench.tasks.contracts import EpisodeRequest, ExpertPlanInputs
from bench.tasks.tm01.config import load_task_config


def test_tm01_uses_ts04_compact_drawer_layout_and_one_station() -> None:
    config = load_task_config()
    drawer_xyzs = tuple(tuple(row) for row in config.scene["drawer_world_xyzs"])
    expected_xyzs = (
        (4.60, -2.37, 1.1363333333333332),
        (4.85, -2.37, 1.1363333333333332),
    )
    assert all(
        actual == pytest.approx(expected)
        for actual, expected in zip(drawer_xyzs, expected_xyzs)
    )
    assert drawer_xyzs[1][0] - drawer_xyzs[0][0] == pytest.approx(0.25)

    route_path = config.path(config.resources["manual_base_path"])
    assert route_path is not None
    route = json.loads(route_path.read_text(encoding="utf-8"))
    assert route["station_targets"]["drawer_left"] == 7
    assert route["station_targets"]["drawer_right"] == 7
    drawer_station = next(
        sample for sample in route["samples"] if sample["name"] == "drawer_station"
    )
    assert drawer_station["xy"] == pytest.approx([4.85, -1.37])


@pytest.mark.parametrize("seed", range(4))
def test_tm01_drawer_plan_returns_arm_and_presses_once(seed: int) -> None:
    episode = TaskCatalog.resolve(EpisodeRequest(task_id="TM01", seed=seed))
    api = resolve_builtin_task_plugin("TM01").load_runtime_api()
    descriptor = api.build_descriptor(episode, False)
    plan = TaskCatalog.get("TM01").build_expert_plan(
        ExpertPlanInputs.from_descriptor(descriptor)
    )

    operations = [
        stage.operation for stage in plan.stages if isinstance(stage, OperateStage)
    ]
    assert operations.count("retreat_and_return_arm_after_drawer") == 2
    assert operations.count("press_completion_button") == 1

    navigation_targets = [
        stage.target.station_id
        for stage in plan.stages
        if isinstance(stage, NavigateStage)
    ]
    assert navigation_targets.count("drawer_station") == 2
    assert "drawer_left_station" not in navigation_targets
    assert "drawer_right_station" not in navigation_targets

    for index, operation in enumerate(operations):
        if operation == "close_drawer_physically":
            assert operations[index + 1] == "retreat_and_return_arm_after_drawer"
