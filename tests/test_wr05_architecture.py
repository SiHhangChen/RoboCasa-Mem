"""Focused WR-05 canonical-architecture tests (config/episode/expert/resources)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from bench.execution.contracts import NavigateStage, OperateStage
from bench.tasks.wr05.config import (
    DEFAULT_TASK_CONFIG_FILE,
    load_task_config,
    resolve_action_config,
)
from bench.tasks.wr05.domain.episode import EPISODE_CASES, resolve_episode_plan
from bench.tasks.wr05.domain.semantics import SemanticState, evaluate_success, record_plate_stage
from bench.tasks.wr05.episode import resolve_episode
from bench.tasks.wr05.expert.spec import build_expert_plan
from bench.tasks.wr05.runtime import _ADAPTER
from bench.tasks.contracts import EpisodeRequest, ExpertPlanInputs

ROOT = Path(__file__).resolve().parents[1]


def test_config_loads_with_canonical_identity() -> None:
    config = load_task_config()
    assert config.task["id"] == "WR-05"
    assert config.scene["env_name"] == "WR05ReturnCondimentPair"
    assert config.episode["max_horizon"] == 5000
    assert set(config.resources) == {
        "ketchup_shelf_grasp",
        "mustard_shelf_grasp",
        "plate_ketchup_return_grasp",
        "plate_mustard_return_grasp",
        "planner_script",
        "stations",
    }


def test_episode_cases_are_balanced_and_deterministic() -> None:
    assert len(EPISODE_CASES) == 4
    assert {item for case in EPISODE_CASES for item in case[0].values()} == {
        "shelf_lower",
        "shelf_upper",
    }
    assert {item for case in EPISODE_CASES for item in case[1].values()} == {
        "plate_left",
        "plate_right",
    }
    seen = {resolve_episode_plan(seed=seed, max_horizon=5000).case_id for seed in range(8)}
    assert seen == {0, 1, 2, 3}


def test_wr05_success_semantics_require_both_staged_and_returned() -> None:
    state = SemanticState()
    assert record_plate_stage(state, "ketchup")
    assert not record_plate_stage(state, "ketchup")  # idempotent
    report = evaluate_success(
        state,
        type("P", (), {
            "condiments_in_return_shelf": True,
            "condiment_shelf_level_relation_correct": True,
            "condiments_released": True,
        })(),
    )
    assert report.success is False  # mustard not staged
    assert record_plate_stage(state, "mustard")
    report = evaluate_success(
        state,
        type("P", (), {
            "condiments_in_return_shelf": True,
            "condiment_shelf_level_relation_correct": True,
            "condiments_released": True,
        })(),
    )
    assert report.success


def test_guarded_place_config_is_enabled() -> None:
    config = load_task_config()
    assert config.manipulation["planner"]["nearest_ik_samples"] == 64
    actions = resolve_action_config(config=load_task_config())
    plate_place = actions.plate_place_by_region["plate_left"]
    assert plate_place["require_pre_release_support"] is True
    assert plate_place["allow_place_trajectory_early_stop"] is True
    assert plate_place["place_descent_stop_bottom_offset"] == 0.005
    assert plate_place["dry_run_only"] is False
    assert plate_place["release_separation_assist_distance"] == 0.0
    assert plate_place["release_anchor_align_support"] is False
    home_place = actions.home_place_by_condiment["mustard"]
    assert actions.plate_pick_by_condiment["mustard"][
        "grasp_near_miss_max_position_error"
    ] == 0.11
    assert actions.plate_pick_by_condiment["ketchup"]["grasp_offset_z"] == -0.055
    assert actions.plate_pick_by_condiment["mustard"]["grasp_offset_z"] == -0.055
    assert actions.plate_pick_by_condiment["mustard"]["grasp_settle_steps"] == 60
    assert home_place["placement_reference"] == "object_origin"
    assert home_place["require_pre_release_support"] is True
    assert home_place["placement_entry_standoff"] == 0.08
    assert home_place["placement_exit_standoff"] == 0.08
    assert home_place["settle_condiment_upright"] is True
    assert home_place["position_retry_count"] == 1
    assert home_place["position_retry_max_error"] == 0.05
    assert home_place["position_retry_settle_steps"] == 60
    assert home_place["allow_place_near_miss"] is True
    assert home_place["place_near_miss_max_position_error"] == 0.05
    assert home_place["allow_place_trajectory_early_stop"] is True
    assert home_place["place_descent_stop_bottom_offset"] == 0.005
    assert home_place["allow_shelf_region_release"] is False
    assert home_place["semantic_place_if_unreachable"] is False
    assert home_place["object_relative_place_grasp_pose_file"] is None
    ketchup_home = actions.home_place_by_condiment["ketchup"]
    assert ketchup_home["placement_entry_standoff"] == 0.08
    assert ketchup_home["placement_exit_standoff"] == 0.08
    assert ketchup_home["object_relative_place_grasp_pose_file"] is None
    assert actions.return_navigation["tolerance"] == 0.02
    assert actions.return_navigation["yaw_tolerance"] == 0.03
    ketchup_return = actions.plate_pick_by_condiment["ketchup"]
    assert ketchup_return["plate_side_grasp_yaw_offsets_deg"] == [
        0.0,
        -45.0,
        45.0,
        90.0,
    ]


def test_expert_plan_is_dual_object_round_trip() -> None:
    episode = resolve_episode(EpisodeRequest(task_id="WR-05", seed=3, variant="memory"))
    descriptor = _ADAPTER.build_descriptor(episode, execute=False)
    plan = build_expert_plan(ExpertPlanInputs.from_descriptor(descriptor))

    assert len(plan.stages) == 26
    navigations = [
        stage
        for stage in plan.stages
        if isinstance(stage, NavigateStage)
    ]
    operations = [
        stage
        for stage in plan.stages
        if isinstance(stage, OperateStage)
    ]
    assert sum(stage.target.holding is not None for stage in navigations) == 4
    # The right slot uses the translated left-slot geometry for both bottles.
    ketchup_plate_nav = next(
        stage
        for stage in navigations
        if stage.name == "navigate_ketchup_plate_place"
    )
    assert ketchup_plate_nav.target.station_id == "ketchup_plate_right_station"
    # The plate places resolve to this episode's use assignment and the shelf
    # returns to its home assignment (one token per condiment each).
    home = dict(episode.facts["home_assignment"])
    use = dict(episode.facts["use_assignment"])
    tokens = [
        dict(stage.metadata)["target_pos"]
        for stage in operations
        if stage.operation == "place_at_accessible"
    ]
    assert set(tokens) == {
        "ketchup:" + use["ketchup"],
        "mustard:" + use["mustard"],
        "ketchup:" + home["ketchup"],
        "mustard:" + home["mustard"],
    }


def test_plate_left_station_is_shared_by_ketchup_and_mustard() -> None:
    station_by_seed_and_stage: dict[tuple[int, str], str] = {}
    for seed, stage_name in (
        (0, "navigate_ketchup_plate_place"),
        (1, "navigate_mustard_plate_place"),
    ):
        episode = resolve_episode(
            EpisodeRequest(task_id="WR-05", seed=seed, variant="memory")
        )
        descriptor = _ADAPTER.build_descriptor(episode, execute=False)
        plan = build_expert_plan(ExpertPlanInputs.from_descriptor(descriptor))
        navigation = next(
            stage
            for stage in plan.stages
            if isinstance(stage, NavigateStage) and stage.name == stage_name
        )
        station_by_seed_and_stage[(seed, stage_name)] = navigation.target.station_id

    assert set(station_by_seed_and_stage.values()) == {"plate_left_station"}


def test_plate_right_station_is_shared_by_ketchup_and_mustard() -> None:
    station_by_seed_and_stage: dict[tuple[int, str], str] = {}
    for seed, stage_name in (
        (0, "navigate_mustard_plate_place"),
        (1, "navigate_ketchup_plate_place"),
    ):
        episode = resolve_episode(
            EpisodeRequest(task_id="WR-05", seed=seed, variant="memory")
        )
        descriptor = _ADAPTER.build_descriptor(episode, execute=False)
        plan = build_expert_plan(ExpertPlanInputs.from_descriptor(descriptor))
        navigation = next(
            stage
            for stage in plan.stages
            if isinstance(stage, NavigateStage) and stage.name == stage_name
        )
        station_by_seed_and_stage[(seed, stage_name)] = navigation.target.station_id

    assert set(station_by_seed_and_stage.values()) == {
        "ketchup_plate_right_station"
    }


def test_public_prompt_does_not_leak_private_binding() -> None:
    from bench.tasks.wr05.prompts import resolve_prompt_spec

    for seed in range(8):
        prompt = resolve_prompt_spec(variant="memory")
        text = prompt.initial_prompt + " " + prompt.policy_prompt
        assert "shelf_upper" not in text
        assert "shelf_lower" not in text
        assert "plate_left" not in text
        assert "plate_right" not in text
        del seed


def test_all_resources_resolve_within_the_task_package() -> None:
    config = load_task_config()
    task_root = (ROOT / "bench" / "tasks" / "wr05").resolve()
    for name in sorted(config.resources):
        path = config.path(config.resources[name])
        assert path is not None and path.is_file(), name
        if name == "planner_script":
            # The shared mplib planner is a bench-owned resource, not task-local.
            assert path.resolve().name == "mplib_planner.py"
        else:
            assert path.resolve().is_relative_to(task_root), name


def test_ketchup_right_station_preserves_left_slot_base_geometry() -> None:
    import json

    stations_file = load_task_config().path(load_task_config().resources["stations"])
    assert stations_file is not None
    stations = json.loads(stations_file.read_text(encoding="utf-8"))["stations"]
    left = stations["plate_left_station"]["samples"][0]
    ketchup_right = stations["ketchup_plate_right_station"]["samples"][0]

    np.testing.assert_allclose(
        np.asarray(ketchup_right["base_xy"], dtype=float)
        - np.asarray(left["base_xy"], dtype=float),
        [0.5, 0.0],
    )
    assert ketchup_right["base_yaw"] == left["base_yaw"]
    assert ketchup_right["torso_height"] == left["torso_height"]


def test_catalog_and_plugin_resolve_wr05() -> None:
    from bench.tasks import TaskCatalog, resolve_builtin_task_plugin

    assert "WR-05" in TaskCatalog.ids()
    plugin = resolve_builtin_task_plugin("WR-05")
    assert plugin.env_name == "WR05ReturnCondimentPair"
    assert plugin.default_task_config.resolve() == DEFAULT_TASK_CONFIG_FILE.resolve()


def test_symbolic_target_resolution_matches_episode_targets() -> None:
    from bench.tasks.wr05.execution.handlers import _resolve_symbolic_target

    targets = type(
        "Targets",
        (),
        {
            "use_pose_by_condiment": {"ketchup": (1.0, 2.0, 0.9), "mustard": (1.5, 2.0, 0.9)},
            "return_place_sampling": {
                "ketchup": {"sampled_target_xy": [0.1, -1.0]},
                "mustard": {"sampled_target_xy": [0.2, -1.0]},
            },
            "original_home_pose_by_condiment": {
                "ketchup": (0.0, 0.0, 1.5),
                "mustard": (0.0, 0.0, 1.8),
            },
        },
    )()

    assert _resolve_symbolic_target(
        object_name="ketchup", target_spec="ketchup:plate_left", targets=targets
    ) == (1.0, 2.0, 0.9)
    assert _resolve_symbolic_target(
        object_name="mustard", target_spec="mustard:shelf_upper", targets=targets
    ) == (0.2, -1.0, 1.8)
    # A non-token target passes through unchanged.
    assert _resolve_symbolic_target(
        object_name="ketchup", target_spec=(1.0, 1.0, 1.0), targets=targets
    ) == (1.0, 1.0, 1.0)


def test_pick_payload_uses_mujoco_wxyz_object_quaternion() -> None:
    from bench.execution.planning import matrix_from_pos_quat_xyzw
    from bench.tasks.wr05.execution.manipulation import (
        _eef_in_object_for_pick_payload,
    )

    data = type(
        "Data",
        (),
        {
            "body_xpos": np.asarray([[0.0, 0.0, 0.0]], dtype=float),
            # MuJoCo body_xquat is WXYZ; this is the identity rotation.
            "body_xquat": np.asarray([[1.0, 0.0, 0.0, 0.0]], dtype=float),
        },
    )()
    env = type(
        "Env",
        (),
        {"obj_body_id": {"ketchup": 0}, "sim": type("Sim", (), {"data": data})()},
    )()
    payload = {
        "poses": {
            "grasp": {
                "world": {"p": [0.0, 0.0, 0.0], "q_xyzw": [0.0, 0.0, 0.0, 1.0]}
            }
        }
    }

    relative = _eef_in_object_for_pick_payload(
        env, object_name="ketchup", payload=payload
    )
    relative_matrix = matrix_from_pos_quat_xyzw(
        np.asarray(relative["p"], dtype=float),
        np.asarray(relative["q_xyzw"], dtype=float),
    )
    np.testing.assert_allclose(relative_matrix, np.eye(4), atol=1e-8)


def test_equal_length_plate_pick_uses_object_calibrated_side_grasp() -> None:
    from bench.tasks.wr05.execution.manipulation import (
        _pick_plan_score,
        _preferred_plate_side_grasp_yaw,
    )

    validation = {
        "pass": True,
        "stages": [
            {"plan": {"trajectory_len": 32}},
            {"plan": {"trajectory_len": 23}},
        ],
    }
    assert _pick_plan_score(
        validation, index=3, yaw_offset_deg=90.0
    ) < _pick_plan_score(
        validation, index=1, yaw_offset_deg=-45.0
    )
    assert _pick_plan_score(
        validation, index=2, yaw_offset_deg=45.0, preferred_yaw_deg=45.0
    ) < _pick_plan_score(
        validation, index=3, yaw_offset_deg=90.0, preferred_yaw_deg=45.0
    )
    left_env = type(
        "Env", (), {"use_assignment": {"mustard": "plate_left"}}
    )()
    right_env = type(
        "Env", (), {"use_assignment": {"mustard": "plate_right"}}
    )()
    assert _preferred_plate_side_grasp_yaw(
        left_env, object_name="mustard"
    ) == 45.0
    assert _preferred_plate_side_grasp_yaw(
        right_env, object_name="mustard"
    ) == 90.0
    assert _preferred_plate_side_grasp_yaw(
        left_env, object_name="ketchup"
    ) == 90.0


def test_successful_place_consumes_legacy_options_without_pose_correction() -> None:
    from bench.tasks.wr05.execution.handlers import place_at_accessible

    calls: dict[str, object] = {}

    class FakeController:
        obs: dict[str, object] = {}
        runtime_config = type("Runtime", (), {"execute": True})()

        def place_at_world_pose(self, object_name: str, **kwargs: object) -> dict[str, object]:
            calls["place_kwargs"] = kwargs
            return {"pass": True}

        def refresh_obs(self) -> None:
            pass

    result = place_at_accessible(
        FakeController(),
        object_name="ketchup",
        target_pos_world=[10.0, 20.0, 1.6],
        settle_condiment_upright=True,
        upright_quat_xyzw=[0.0, 0.0, 0.0, 1.0],
        position_retry_count=1,
        position_retry_max_error=0.05,
        position_retry_settle_steps=60,
        allow_shelf_region_release=False,
        semantic_place_if_unreachable=False,
        object_relative_place_grasp_pose_file=None,
    )

    assert result["pass"] is True
    forwarded = calls["place_kwargs"]
    for legacy_key in (
        "position_retry_count",
        "position_retry_max_error",
        "position_retry_settle_steps",
        "allow_shelf_region_release",
        "semantic_place_if_unreachable",
        "object_relative_place_grasp_pose_file",
        "settle_condiment_upright",
    ):
        assert legacy_key not in forwarded
    assert forwarded["allow_place_entry_near_miss"] is True
    assert forwarded["place_entry_near_miss_max_position_error"] == 0.05
