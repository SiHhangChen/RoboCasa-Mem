"""WR-08 applies the WR-07 pick and place strategy to either carton."""

from types import SimpleNamespace

import numpy as np
import pytest

from bench.execution.planning import PickPlaceConfig
from bench.tasks.wr08.domain.contracts import ITEM_IDS
from bench.tasks.wr08.execution import geometry
from bench.tasks.wr08.execution.handlers import build_operation_handlers
from bench.tasks.wr08.execution.pick_controller import (
    WorldAnchoredPickExecutor,
    compiled_shelf_contact_geoms,
)
from bench.execution.manipulation.planning_bridge import PlannedStageExecutor
from bench.execution.planning import matrix_from_base_obs, matrix_from_json_pose, pose_from_matrix


def _selector(monkeypatch, outcomes, item=ITEM_IDS[0]):
    generated = []
    validated = []

    def make_payload(env, obs, **kwargs):
        generated.append(kwargs)
        return {"yaw": kwargs["grasp_yaw_offset_deg"]}

    def dry_run(payload, *, object_name):
        validated.append((payload["yaw"], object_name))
        passed, lengths, failed_stage = outcomes[payload["yaw"]]
        return {
            "pass": passed,
            "stages": [
                {
                    "stage": name,
                    "pass": name != failed_stage,
                    "failure_reason": "IK failed" if name == failed_stage else None,
                    "plan": {"trajectory_len": length},
                }
                for name, length in zip(("pregrasp", "grasp", "lift"), lengths)
            ],
        }

    monkeypatch.setattr(geometry, "make_object_pick_target_poses", make_payload)
    request = {
        "env": object(),
        "obs": {},
        "object_name": item,
        "pick_place_config": PickPlaceConfig(),
        "lift_height": 0.12,
        "kwargs": {
            "grasp_yaw_offsets_deg": [0, 90, 180, 270],
            "pregrasp_height": 0.12,
            "object_pos_world": [1.0, 2.0, 3.0],
        },
        "dry_run_pick_sequence": dry_run,
    }
    return geometry.build_pick_payload_selector(), request, generated, validated


@pytest.mark.parametrize("item", ITEM_IDS)
def test_source_grasp_minimizes_longest_stage_before_total_length(monkeypatch, item):
    selector, request, generated, validated = _selector(
        monkeypatch,
        {
            0: (False, [1, 1, 1], "lift"),
            90: (True, [30, 1, 1], None),
            180: (True, [15, 15, 15], None),
            270: (True, [15, 15, 15], None),
        },
        item,
    )
    result = selector(request)
    assert result["payload"] == {"yaw": 180.0}
    assert result["selection"]["pass"]
    assert result["selection"]["selected_score"] == [15, 45, 2]
    assert validated == [(float(yaw), item) for yaw in (0, 90, 180, 270)]
    assert all(entry["orientation_mode"] == "object-y-topdown" for entry in generated)
    assert all(entry["pregrasp_height"] == 0.12 for entry in generated)
    assert all(entry["object_pos_world"] == [1.0, 2.0, 3.0] for entry in generated)


def test_source_grasp_fails_closed_when_no_full_chain_is_reachable(monkeypatch):
    selector, request, _, _ = _selector(
        monkeypatch, {yaw: (False, [1, 1, 1], "grasp") for yaw in (0, 90, 180, 270)}
    )
    result = selector(request)
    assert result["payload"] is None
    assert not result["selection"]["pass"]
    assert result["selection"]["reason"] == "no_reachable_topdown_grasp_candidate"


def test_source_grasp_rejects_incomplete_plan_marked_pass(monkeypatch):
    selector, request, _, _ = _selector(
        monkeypatch, {yaw: (True, [1, 1], None) for yaw in (0, 90, 180, 270)}
    )
    assert selector(request)["payload"] is None


@pytest.mark.parametrize("key", ["manual_grasp_pose_file", "grasp_pose_file"])
def test_source_grasp_preserves_explicit_calibration(monkeypatch, key):
    selector, request, generated, _ = _selector(monkeypatch, {})
    request["kwargs"][key] = "calibrated.json"
    assert selector(request) is None
    assert generated == []


@pytest.mark.parametrize("offsets", [[], [float("nan")], [float("inf")], [[90]]])
def test_source_grasp_rejects_invalid_offsets(monkeypatch, offsets):
    selector, request, generated, _ = _selector(monkeypatch, {})
    request["kwargs"]["grasp_yaw_offsets_deg"] = offsets
    with pytest.raises(ValueError, match="nonempty finite list"):
        selector(request)
    assert generated == []


@pytest.mark.parametrize("item", ITEM_IDS)
def test_target_place_dispatches_all_yaws_through_candidate_planner(item):
    calls = []
    targets = {ITEM_IDS[0]: [1.0, 2.0, 0.93], ITEM_IDS[1]: [2.0, 2.0, 0.93]}
    controller = SimpleNamespace(
        env=SimpleNamespace(
            target_plate_pose=lambda slot, item: np.asarray(targets[item]),
            target_plate_color_by_slot={"target_left": "red"},
        ),
        runtime_config=SimpleNamespace(execute=True),
        place_at_world_pose_candidates=lambda name, **kwargs: (
            calls.append((name, kwargs)) or {"pass": True}
        ),
    )
    result = build_operation_handlers(controller)["place_at_accessible"](
        item,
        target_resolver="target_plate_pose",
        target_slot="target_left",
        source_color="red",
        placement_yaw_offsets_deg=[0, 90, 180, 270],
        placement_reference="bbox_bottom_center",
        placement_object_upright=True,
    )
    assert result["pass"]
    assert calls[0][0] == item
    options = calls[0][1]
    candidates = options["candidate_groups"][0]["candidates"]
    assert [entry["placement_yaw_offset_deg"] for entry in candidates] == [0, 90, 180, 270]
    assert all(entry["target_pos_world"] == targets[item] for entry in candidates)
    assert callable(options["pre_release_check"])
    assert options["placement_object_upright"]
    assert options["placement_reference"] == "bbox_bottom_center"


def test_target_place_rejects_wrong_color_before_moving():
    controller = SimpleNamespace(
        env=SimpleNamespace(
            target_plate_pose=lambda slot, item: np.array([1.0, 2.0, 0.93]),
            target_plate_color_by_slot={"target_left": "blue"},
        ),
        runtime_config=SimpleNamespace(execute=True),
    )
    result = build_operation_handlers(controller)["place_at_accessible"](
        ITEM_IDS[0],
        target_resolver="target_plate_pose",
        target_slot="target_left",
        source_color="red",
        placement_yaw_offsets_deg=[0, 90, 180, 270],
    )
    assert not result["pass"]
    assert result["reason"] == "target_plate_color_mismatch"


def _base_obs(x=0.0, y=0.0, yaw=0.0):
    return {
        "robot0_base_pos": [x, y, 0.7],
        "robot0_base_quat": [0.0, 0.0, np.sin(yaw / 2), np.cos(yaw / 2)],
    }


def _pick_executor(obs=None):
    obs = _base_obs() if obs is None else obs
    return WorldAnchoredPickExecutor(
        SimpleNamespace(_get_observations=lambda **kwargs: obs), obs,
        driver=object(), planner=lambda request: {},
        runtime_config=SimpleNamespace(execute=True),
    )


@pytest.mark.parametrize("stage", ["grasp", "lift"])
def test_pick_world_target_does_not_follow_base_drift(monkeypatch, stage):
    observed = []
    monkeypatch.setattr(PlannedStageExecutor, "run", lambda self, target, **kw: (
        observed.append(target) or {"pass": True}
    ))
    current_obs = _base_obs(0.02, -0.01, 0.08)
    initial_obs = _base_obs()
    executor = _pick_executor(current_obs)
    target = np.eye(4)
    target[:3, 3] = [0.5, -0.3, 0.27]
    with executor.pick_frame(initial_obs):
        result = executor.run(pose_from_matrix(target), stage=stage, gripper_action=-1)
    actual_world = matrix_from_base_obs(current_obs) @ matrix_from_json_pose(observed[0])
    np.testing.assert_allclose(actual_world, matrix_from_base_obs(initial_obs) @ target, atol=1e-12)
    assert result["pick_target_frame"] == "world_anchored"
    assert executor._pick_base_world is None


def test_pick_reuses_only_matching_pregrasp_plan_and_clears_cache(monkeypatch):
    calls = []
    prepared = {"selected": "validated_pregrasp"}

    def run(self, target, **kwargs):
        calls.append(kwargs)
        if kwargs.get("capture_prepared_plan"):
            return {"pass": True, "prepared_plan": prepared}
        return {"pass": True}

    monkeypatch.setattr(PlannedStageExecutor, "run", run)
    executor = _pick_executor()
    target = pose_from_matrix(np.eye(4))
    other = np.eye(4)
    other[0, 3] = 0.1
    with executor.pick_frame(_base_obs()):
        executor.run(target, stage="pregrasp", gripper_action=-1, execute=False)
        executor.run(pose_from_matrix(other), stage="pregrasp", gripper_action=-1)
        executor.run(target, stage="pregrasp", gripper_action=-1)
    assert "prepared_plan" not in calls[1]
    assert calls[2]["prepared_plan"] is prepared
    assert executor._pregrasp_plans == {}
    executor.run(target, stage="pregrasp", gripper_action=-1)
    assert "prepared_plan" not in calls[-1]


def test_dry_pick_keeps_virtual_base_targets_and_restores_context_on_error(monkeypatch):
    observed = []
    monkeypatch.setattr(PlannedStageExecutor, "run", lambda self, target, **kw: (
        observed.append(target) or {"pass": True}
    ))
    executor = _pick_executor(_base_obs(0.5, 0.2, 1.0))
    target = pose_from_matrix(np.eye(4))
    with pytest.raises(RuntimeError, match="probe"):
        with executor.pick_frame(_base_obs()):
            executor.run(target, stage="grasp", gripper_action=-1, execute=False)
            raise RuntimeError("probe")
    assert observed[0] is target
    assert executor._pick_base_world is None
    assert executor._pregrasp_plans == {}


def _shelf_env():
    names = ["shelf", "shelf_visual", "nested_shelf", "other_fixture", "one_way_shelf"]
    model = SimpleNamespace(
        ngeom=len(names), body_name2id=lambda name: 1,
        body_parentid=[0, 0, 1, 0], geom_bodyid=[1, 1, 2, 3, 2],
        geom_contype=[1, 0, 1, 1, 0], geom_conaffinity=[1, 0, 1, 1, 1],
        geom_id2name=lambda index: names[index],
    )
    return SimpleNamespace(
        sim=SimpleNamespace(model=model),
        fixtures={"opencabinet": SimpleNamespace(root_body="cabinet_root", contact_geoms=[])},
    )


def test_pick_collision_geoms_include_procedural_shelves_but_not_visuals():
    assert compiled_shelf_contact_geoms(_shelf_env()) == {"shelf", "nested_shelf", "one_way_shelf"}


def test_pick_collision_monitor_includes_compiled_shelves_for_arm_and_carton(monkeypatch):
    options = []

    def monitor(self, holding_object, *, collision_options):
        options.append(collision_options)
        return {key: ["existing"] for key in ("obstacle_geoms", "fixture_obstacle_geoms", "held_obstacle_geoms")}

    monkeypatch.setattr(PlannedStageExecutor, "_collision_monitor", monitor)
    executor = _pick_executor()
    executor.env = _shelf_env()
    with executor.pick_frame(_base_obs()):
        result = executor._collision_monitor(ITEM_IDS[0])
    assert options[0]["validate_collisions"] is True
    for key in ("obstacle_geoms", "fixture_obstacle_geoms", "held_obstacle_geoms"):
        assert set(result[key]) == {"existing", "shelf", "nested_shelf", "one_way_shelf"}
