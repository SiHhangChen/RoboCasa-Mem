from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from bench.execution.probe import (
    ProbeStationPose,
    build_probe_task,
    load_eef_pose_file,
    load_stage_file,
    make_move_eef_stage,
    reset_base_to_station,
    resolve_station_pose,
    select_workflow_stages,
    stage_catalog,
)
from bench.execution.contracts import ExpertPlan, NavigateStage, OperateStage
from bench.execution.manipulation.controller import ManipulationController
from bench.execution.navigation.contracts import StationTarget


def _task() -> ExpertPlan:
    return ExpertPlan(
        task_id="TEST",
        stages=(
            NavigateStage(StationTarget("station_a"), name="go_a"),
            OperateStage(operation="press", target="button", name="press_button"),
            OperateStage(operation="pick", target="cup"),
        ),
        metadata={"production": True},
    )


def test_stage_catalog_and_selection_follow_requested_order() -> None:
    task = _task()

    assert [row["name"] for row in stage_catalog(task)] == [
        "go_a",
        "press_button",
        "pick:cup",
    ]

    selected = select_workflow_stages(task, ["press_button", "0", "pick:cup"])

    assert [stage.name for stage in selected[:2]] == ["press_button", "go_a"]
    assert isinstance(selected[2], OperateStage)
    assert selected[2].target == "cup"


def test_load_custom_stage_file(tmp_path) -> None:
    path = tmp_path / "stages.yaml"
    path.write_text(
        """
stages:
  - kind: navigate
    station: station_b
    purpose: probe
  - kind: operate
    operation: move_eef
    target: pose_1
    metadata:
      frame: base
      target_pose:
        p: [0.4, 0.0, 0.6]
        q_xyzw: [0.0, 1.0, 0.0, 0.0]
""",
        encoding="utf-8",
    )

    stages = load_stage_file(path)

    assert isinstance(stages[0], NavigateStage)
    assert stages[0].target.station_id == "station_b"
    assert isinstance(stages[1], OperateStage)
    assert stages[1].operation == "move_eef"
    assert stages[1].metadata["frame"] == "base"


def test_external_station_pose_selection(tmp_path) -> None:
    path = tmp_path / "stations.json"
    path.write_text(
        json.dumps(
            {
                "stations": {
                    "door": {
                        "samples": [
                            {
                                "sample_index": 4,
                                "base_xy": [1.2, -0.7],
                                "base_yaw": 1.57,
                                "torso_height": 0.21,
                            }
                        ]
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    pose = resolve_station_pose(
        object(),
        station_id="door",
        station_file=path,
        sample_index=4,
    )

    assert pose.xy == pytest.approx((1.2, -0.7))
    assert pose.yaw == pytest.approx(1.57)
    assert pose.torso_height == pytest.approx(0.21)
    assert pose.sample["sample_index"] == 4


def test_reset_station_applies_torso_height_and_refreshes_controllers(
    monkeypatch,
) -> None:
    qpos = np.array([0.0, *np.zeros(7)], dtype=float)
    torso = SimpleNamespace(
        qpos_index=np.array([0], dtype=int),
        joint_index=np.array([0], dtype=int),
    )
    right = SimpleNamespace(qpos_index=np.arange(1, 8, dtype=int))
    composite = SimpleNamespace(
        part_controllers={"torso": torso, "right": right},
        _action_split_indexes={"base": (0, 3)},
    )
    robot = SimpleNamespace(composite_controller=composite)
    obs = {
        "robot0_base_pos": np.array([1.2, -0.7, 0.9], dtype=float),
        "robot0_base_quat": np.array([0.0, 0.0, 0.0, 1.0], dtype=float),
    }
    env = SimpleNamespace(
        robots=[robot],
        sim=SimpleNamespace(
            data=SimpleNamespace(qpos=qpos),
            model=SimpleNamespace(jnt_range=np.array([[0.0, 0.34]], dtype=float)),
        ),
        _get_observations=lambda force_update: obs,
    )
    refresh_counts = {"nav": 0, "manip": 0}
    driver = SimpleNamespace(
        refresh_obs=lambda: refresh_counts.__setitem__(
            "nav", refresh_counts["nav"] + 1
        )
    )
    manip = SimpleNamespace(
        refresh_obs=lambda: refresh_counts.__setitem__(
            "manip", refresh_counts["manip"] + 1
        )
    )
    bundle = SimpleNamespace(
        env=env,
        runner=SimpleNamespace(
            navigation_controller=SimpleNamespace(base_driver=driver),
            manipulation_controller=manip,
        ),
    )
    monkeypatch.setattr(
        "bench.execution.probe.set_base_pose_kinematic",
        lambda *args, **kwargs: obs,
    )

    def fake_step(*args, torso_action, **kwargs):
        del args, kwargs
        qpos[0] += float(torso_action) * 0.05
        return obs

    monkeypatch.setattr(
        "bench.execution.probe.step_base_and_hold_arm",
        fake_step,
    )
    station = ProbeStationPose(
        station_id="shelf_upper_station",
        xy=(1.2, -0.7),
        yaw=1.57,
        source="test",
        sample={"torso_height": 0.2},
        torso_height=0.2,
    )

    summary = reset_base_to_station(bundle, station)

    assert summary["requested_torso_height"] == pytest.approx(0.2)
    assert summary["torso"]["applied"] is True
    assert summary["torso"]["actual_height"] == pytest.approx(0.2)
    assert summary["torso"]["steps"] == 4
    assert refresh_counts == {"nav": 1, "manip": 1}


def test_live_anchor_pose_is_resolved_against_current_site(tmp_path) -> None:
    path = tmp_path / "pose.json"
    path.write_text(
        json.dumps(
            {
                "config_kind": "anchor_pose",
                "anchor": {
                    "selector": "site:handle",
                    "world": [1.0, 2.0, 3.0],
                },
                "pose": {
                    "offset_from_anchor_world": [0.1, -0.2, 0.3],
                    "world": {
                        "p": [1.1, 1.8, 3.3],
                        "q_xyzw": [0.0, 0.0, 0.0, 1.0],
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    model = SimpleNamespace(site_name2id=lambda name: 0)
    data = SimpleNamespace(site_xpos=np.asarray([[4.0, 5.0, 6.0]], dtype=float))
    env = SimpleNamespace(sim=SimpleNamespace(model=model, data=data))

    pose, frame, summary = load_eef_pose_file(path, env=env)

    assert frame == "world"
    assert pose["p"] == pytest.approx([4.1, 4.8, 6.3])
    assert summary["selector"] == "site:handle"


def test_move_eef_is_a_generic_manipulation_operation() -> None:
    controller = object.__new__(ManipulationController)
    controller.refresh_obs = lambda: {}
    captured = {}

    def move(**kwargs):
        captured.update(kwargs)
        return {"pass": True, "primitive": "move_eef"}

    controller.move_eef_to_pose = move

    result = controller.execute(
        "move_eef",
        "probe_pose",
        target_pose={
            "p": [0.4, 0.0, 0.6],
            "q_xyzw": [0.0, 1.0, 0.0, 0.0],
        },
        frame="base",
    )

    assert result.success is True
    assert captured["target_name"] == "probe_pose"
    assert captured["frame"] == "base"


def test_probe_task_drops_production_final_predicates() -> None:
    task = _task()
    move = make_move_eef_stage(
        target_pose={
            "p": [0.4, 0.0, 0.6],
            "q_xyzw": [0.0, 1.0, 0.0, 0.0],
        },
        frame="base",
    )

    probe = build_probe_task(task, [move])

    assert probe.metadata["probe"] is True
    assert probe.metadata["diagnostic_probe"] is True
    assert probe.stages == (move,)
