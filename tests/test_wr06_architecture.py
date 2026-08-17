"""Focused WR-06 canonical-architecture tests (config/episode/expert/resources)."""

from __future__ import annotations

from pathlib import Path

from bench.execution.contracts import NavigateStage, OperateStage
from bench.tasks.wr06.config import (
    DEFAULT_TASK_CONFIG_FILE,
    load_task_config,
    resolve_action_config,
)
from bench.tasks.wr06.domain.episode import (
    demo_original_level_for_seed,
    plate_stage_region_for_seed,
    resolve_episode_plan,
)
from bench.tasks.wr06.episode import resolve_episode
from bench.tasks.wr06.expert.spec import build_expert_plan
from bench.tasks.wr06.runtime import _ADAPTER
from bench.tasks.contracts import EpisodeRequest, ExpertPlanInputs

ROOT = Path(__file__).resolve().parents[1]


def test_config_loads_with_canonical_identity() -> None:
    config = load_task_config()
    assert config.task["id"] == "WR-06"
    assert config.scene["env_name"] == "WR06ReturnSingleBottleToOriginalLevel"
    assert config.episode["max_horizon"] == 3200
    assert config.semantics["terminal_verification_steps"] == 100
    assert set(config.resources) == {
        "item_shelf_grasp",
        "plate_return_grasp",
        "planner_script",
        "stations",
    }


def test_home_regions_keep_bottle_away_from_layout_wall() -> None:
    config = load_task_config()
    for level in ("shelf_lower", "shelf_upper"):
        region = config.initial_state.home_regions[level]
        assert region["size"] == (0.58, 0.13)
        assert region["pos"] == (0.25, -1.0)


def test_original_level_follows_seed_parity() -> None:
    assert demo_original_level_for_seed(0) == "shelf_lower"
    assert demo_original_level_for_seed(1) == "shelf_upper"
    assert demo_original_level_for_seed(2) == "shelf_lower"
    assert demo_original_level_for_seed(3) == "shelf_upper"
    plan = resolve_episode_plan(seed=7, max_horizon=3200)
    assert plan.demo_original_level == "shelf_upper"
    assert plan.placement_pair_seed == 3


def test_plate_stage_region_alternates_with_pair_seed() -> None:
    # The four (level, side) cases are balanced and cover every WR-05 pairing.
    cases = [
        (demo_original_level_for_seed(s), plate_stage_region_for_seed(s))
        for s in range(4)
    ]
    assert cases == [
        ("shelf_lower", "plate_left"),
        ("shelf_upper", "plate_left"),
        ("shelf_lower", "plate_right"),
        ("shelf_upper", "plate_right"),
    ]
    for s in range(8):
        plan = resolve_episode_plan(seed=s, max_horizon=3200)
        assert plan.plate_stage_region == plate_stage_region_for_seed(s)
        assert plan.plate_stage_region in {"plate_left", "plate_right"}


def test_action_config_wires_plate_pick_and_guarded_place() -> None:
    actions = resolve_action_config(config=load_task_config())
    assert actions.plate_pick_metadata["manual_grasp_pose_file"].endswith(
        "plate_ketchup_return_grasp.json"
    )
    assert actions.shelf_pick_metadata["manual_grasp_pose_file"].endswith(
        "shelf_ketchup_grasp.json"
    )
    assert actions.plate_place_metadata["require_pre_release_support"] is True
    assert actions.plate_place_metadata["allow_place_trajectory_early_stop"] is True
    assert actions.plate_place_metadata["placement_entry_standoff"] == 0.0
    assert actions.plate_place_metadata["pre_release_xy_tolerance"] == 0.06
    assert actions.plate_place_metadata["post_release_xy_tolerance"] == 0.08
    assert actions.home_place_metadata["placement_reference"] == "object_origin"


def test_expert_plan_is_single_object_round_trip() -> None:
    expected_plate_stage_station = {
        "plate_left": "plate_left_station",
        "plate_right": "ketchup_plate_right_station",
    }
    expected_plate_pick_station = {
        "plate_left": "plate_left_pick_station",
        "plate_right": "plate_right_pick_station",
    }
    for seed in (0, 1, 2, 3):
        episode = resolve_episode(EpisodeRequest(task_id="WR-06", seed=seed, variant="memory"))
        descriptor = _ADAPTER.build_descriptor(episode, execute=False)
        plan = build_expert_plan(ExpertPlanInputs.from_descriptor(descriptor))

        assert len(plan.stages) == 15
        assert plan.metadata["demo_original_level"] == episode.facts["demo_original_level"]
        assert plan.metadata["plate_stage_region"] == episode.facts["plate_stage_region"]
        navigations = [s for s in plan.stages if isinstance(s, NavigateStage)]
        operations = [s for s in plan.stages if isinstance(s, OperateStage)]
        # 3 plate/level navigate purposes are exercised for a single bottle
        purposes = {s.target.purpose for s in navigations}
        assert purposes == {"shelf_observe", "shelf_pick", "plate_stage", "plate_pick", "home_return"}
        op_names = [s.operation for s in operations]
        assert op_names.count("pick") == 2
        assert op_names.count("place_at_accessible") == 1
        assert op_names.count("return_item_to_level") == 1
        assert op_names.count("latch_original_level_hidden") == 1
        assert op_names.count("set_torso_height") == 1
        # The plate place resolves to the episode-decided side and the home
        # return to the remembered level.
        tokens = [
            dict(s.metadata)["target_resolver"]
            for s in operations
            if s.operation == "place_at_accessible"
        ]
        assert tokens == ["plate_stage_pose"]
        return_tokens = [
            dict(s.metadata)["target_resolver"]
            for s in operations
            if s.operation == "return_item_to_level"
        ]
        assert return_tokens == ["home_return_pose"]
        side = episode.facts["plate_stage_region"]
        stage_targets = [
            s.target.station_id
            for s in navigations
            if s.target.purpose == "plate_stage"
        ]
        assert stage_targets == [expected_plate_stage_station[side]]
        pick_targets = [
            s.target.station_id
            for s in navigations
            if s.target.purpose == "plate_pick"
        ]
        assert pick_targets == [expected_plate_pick_station[side]]
        place_names = [
            dict(s.metadata)["target_name"]
            for s in operations
            if s.operation == "place_at_accessible"
        ]
        assert place_names == [f"ketchup_{side}"]


def test_public_prompt_does_not_reveal_original_level() -> None:
    from bench.tasks.wr06.prompts import resolve_prompt_spec

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
    task_root = (ROOT / "bench" / "tasks" / "wr06").resolve()
    for name in sorted(config.resources):
        path = config.path(config.resources[name])
        assert path is not None and path.is_file(), name
        if name == "planner_script":
            # The shared mplib planner is a bench-owned resource, not task-local.
            assert path.resolve().name == "mplib_planner.py"
        else:
            assert path.resolve().is_relative_to(task_root), name


def test_catalog_and_plugin_resolve_wr06() -> None:
    from bench.tasks import TaskCatalog, resolve_builtin_task_plugin

    assert "WR-06" in TaskCatalog.ids()
    plugin = resolve_builtin_task_plugin("WR-06")
    assert plugin.env_name == "WR06ReturnSingleBottleToOriginalLevel"
    assert plugin.default_task_config.resolve() == DEFAULT_TASK_CONFIG_FILE.resolve()
