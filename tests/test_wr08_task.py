"""WR-08 v3 distinct-carton color-matching and physical event tests."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from bench.execution.contracts import NavigateStage, OperateStage
from bench.tasks import TaskCatalog, resolve_builtin_task_plugin
from bench.tasks.contracts import EpisodeRequest, ExpertPlanInputs
from bench.tasks.wr08.config import load_task_config, resolve_action_config
from bench.tasks.wr08.domain.contracts import (
    CARTON_ASSET_BY_ITEM,
    COLOR_NAMES,
    ENV_NAME,
    ITEM_IDS,
    SOURCE_SLOTS,
    TARGET_SLOTS,
    TASK_REVISION,
)
from bench.tasks.wr08.domain.episode import resolve_episode_plan
from bench.tasks.wr08.domain.semantics import (
    PhysicalState,
    SemanticState,
    evaluate_success,
    latch_source_color_memory,
    record_initial_sources,
    record_item_picked,
    record_target_attempt,
    record_target_placement,
)
from bench.tasks.wr08.physical_events import PhysicalSample, WR08PhysicalObserver


def _state_and_targets():
    plan = resolve_episode_plan(seed=0)
    state = SemanticState()
    assert record_initial_sources(
        state, plan.source_slot_by_item, plan.source_color_by_item
    )
    targets = {
        item: plan.target_slot_by_color[plan.source_color_by_item[item]]
        for item in ITEM_IDS
    }
    return state, targets


def _sample(step, targets_by_item, *, departed=(), held=(), placed=None):
    placed = dict(placed or {})
    return PhysicalSample(
        step_id=step,
        item_in_source={item: item not in departed for item in ITEM_IDS},
        item_in_target={item: item in placed for item in ITEM_IDS},
        released={item: item not in held for item in ITEM_IDS},
        item_grasped={item: item in held for item in ITEM_IDS},
        observed_target_slot_by_item=placed,
        target_slot_by_item=targets_by_item,
    )


def _observer():
    state, targets = _state_and_targets()
    observer = WR08PhysicalObserver(state, target_slot_by_item=targets, dwell_steps=3)
    observer.reset(_sample(0, targets))
    return state, targets, observer


def _dwell(observer, targets, start, **kwargs):
    for step in range(start, start + 3):
        update = observer.observe(_sample(step, targets, **kwargs))
    return update


def test_wr08_config_declares_color_matching_and_physical_execution() -> None:
    config = load_task_config()
    assert tuple(config.initial_state.source_regions) == SOURCE_SLOTS
    assert config.scene["env_name"] == ENV_NAME
    assert config.recording["record_depth"] is False
    assert config.navigation["kinematic_base_navigation"] is False
    assert config.manipulation["runtime"]["max_joint_step"] == 0.025
    assert config.manipulation["runtime"]["position_threshold"] == 0.02
    assert config.manipulation["runtime"]["qpos_threshold"] == 0.05

    actions = resolve_action_config(config=config)
    assert set(actions.source_pick_metadata) == set(ITEM_IDS)
    assert set(actions.target_place_metadata) == set(TARGET_SLOTS)
    assert actions.return_arm_holding["gripper_action"] == 1.0
    assert actions.return_arm_empty["gripper_action"] == -1.0
    assert all(
        "manual_grasp_pose_file" not in metadata
        for metadata in actions.source_pick_metadata.values()
    )
    assert all(
        metadata["grasp_yaw_offsets_deg"] == [0, 90, 180, 270]
        for metadata in actions.source_pick_metadata.values()
    )
    assert all(
        metadata["placement_yaw_offsets_deg"] == [0, 90, 180, 270]
        for metadata in actions.target_place_metadata.values()
    )


def test_wr08_uses_two_distinct_existing_carton_assets() -> None:
    assert set(CARTON_ASSET_BY_ITEM) == set(ITEM_IDS)
    assert len(set(CARTON_ASSET_BY_ITEM.values())) == len(ITEM_IDS)
    asset_root = (
        Path(__file__).resolve().parents[1]
        / "robocasa/robocasa/models/assets/objects"
    )
    assert all(
        (asset_root / relative_path).is_file()
        for relative_path in CARTON_ASSET_BY_ITEM.values()
    )


def test_wr08_uses_wr07_motion_and_station_calibration() -> None:
    from bench.tasks.wr07.config import load_task_config as load_wr07
    from bench.tasks.wr08.execution.stations import station_aliases

    reference = load_wr07()
    config = load_task_config()
    assert config.manipulation == reference.manipulation
    assert config.operations == {
        **reference.operations,
        "source_pick": {**reference.operations["source_pick"], "pregrasp_height": 0.06},
    }
    assert config.navigation == reference.navigation
    stations = json.loads(config.path(config.resources["stations"]).read_text())
    reference_stations = json.loads(
        reference.path(reference.resources["stations"]).read_text()
    )
    assert stations["stations"] == reference_stations["stations"]
    aliases = station_aliases()
    assert {aliases[slot] for slot in SOURCE_SLOTS} == {"counter_pick_station"}
    assert aliases["target_plates"] == "plate_station"


def test_wr08_horizon_covers_recorded_rollouts() -> None:
    # The two pre-validation recordings took 2065 and 2063 action steps.
    horizon = int(load_task_config().episode["max_horizon"])
    assert horizon == 5200
    assert horizon > 2065
    assert resolve_episode_plan(seed=0).max_horizon == horizon
    assert TaskCatalog.get("WR-08").default_horizon == horizon
    assert (
        TaskCatalog.resolve(EpisodeRequest(task_id="WR-08", seed=0)).max_horizon
        == horizon
    )


def test_wr08_sampling_is_reproducible_and_preserves_color_correspondence() -> None:
    plans = [resolve_episode_plan(seed=seed) for seed in range(50)]
    assert len(COLOR_NAMES) == len(set(COLOR_NAMES)) == 20
    assert {plan.transport_order for plan in plans} == {ITEM_IDS, ITEM_IDS[::-1]}
    for seed, plan in enumerate(plans):
        assert plan == resolve_episode_plan(seed=seed)
        source = plan.source_plate_color_by_slot
        target = plan.target_plate_color_by_slot
        assert set(source) == set(SOURCE_SLOTS)
        assert set(target) == set(TARGET_SLOTS)
        assert len(set(source.values())) == 3
        assert set(source.values()) == set(target.values()) <= set(COLOR_NAMES)
        assert all(source[a] != target[b] for a, b in zip(SOURCE_SLOTS, TARGET_SLOTS))
        assert len(set(plan.source_slot_by_item.values())) == 2
        assert len(set(plan.source_color_by_item.values())) == 2
        for item in ITEM_IDS:
            color = source[plan.source_slot_by_item[item]]
            assert plan.source_color_by_item[item] == color
            if item == "boxed_food_a" and plan.source_slot_by_item[item] == "source_right":
                assert plan.source_item_offset_xy_by_item[item][1] <= -0.01
            assert target[plan.target_slot_by_color[color]] == color
            for offsets in (
                plan.source_item_offset_xy_by_item,
                plan.target_item_offset_xy_by_item,
            ):
                assert np.linalg.norm(offsets[item]) <= plan.item_sample_radius_m
            for yaws in (
                plan.source_item_yaw_by_item,
                plan.target_item_yaw_by_item,
            ):
                assert -np.pi <= yaws[item] <= np.pi
    for item in ITEM_IDS:
        assert len({plan.source_item_offset_xy_by_item[item] for plan in plans}) == 50
        assert len({plan.target_item_offset_xy_by_item[item] for plan in plans}) == 50

    zero_radius = resolve_episode_plan(seed=3, item_sample_radius_m=0.0)
    assert all(
        offset == (0.0, 0.0)
        for offset in zero_radius.source_item_offset_xy_by_item.values()
    )


@pytest.mark.parametrize(
    "settings",
    [
        {"seed": -1},
        {"seed": 0, "color_count": 2},
        {"seed": 0, "color_count": 4},
        {"seed": 0, "item_sample_radius_m": -0.1},
        {"seed": 0, "item_sample_radius_m": float("nan")},
        {"seed": 0, "item_sample_radius_m": float("inf")},
    ],
)
def test_wr08_rejects_invalid_episode_sampling(settings) -> None:
    with pytest.raises(ValueError):
        resolve_episode_plan(**settings)


@pytest.mark.parametrize("seed", [0, 1, 17])
def test_wr08_expert_plan_transports_each_item_to_its_source_color(seed) -> None:
    episode = TaskCatalog.resolve(EpisodeRequest(task_id="WR-08", seed=seed))
    api = resolve_builtin_task_plugin("WR-08").load_runtime_api()
    descriptor = api.build_descriptor(episode, execute=False)
    plan = TaskCatalog.get("WR-08").build_expert_plan(
        ExpertPlanInputs.from_descriptor(descriptor)
    )
    assert len(plan.stages) == 17
    assert plan.metadata["task_revision"] == TASK_REVISION
    assert plan.metadata["carton_asset_by_item"] == dict(CARTON_ASSET_BY_ITEM)
    assert descriptor.release_facts["carton_asset_by_item"] == dict(CARTON_ASSET_BY_ITEM)
    assert plan.stages[0].target.purpose == "source_observe"

    for index, item in enumerate(episode.facts["transport_order"]):
        stages = plan.stages[1 + 8 * index : 9 + 8 * index]
        source_slot = episode.facts["source_slot_by_item"][item]
        color = episode.facts["source_color_by_item"][item]
        target_slot = episode.facts["target_slot_by_color"][color]
        source_nav, pick, stow, latch, target_nav, place, verify, retreat = stages
        assert isinstance(source_nav, NavigateStage)
        assert source_nav.target.station_id == source_slot
        assert source_nav.target.purpose == "source_pick"
        assert source_nav.metadata["xy_tolerance"] == 0.08
        assert isinstance(target_nav, NavigateStage)
        assert target_nav.target.purpose == "target_place"
        assert target_nav.target.holding == item
        assert target_nav.metadata["target_slot"] == target_slot
        assert target_nav.metadata["holding_navigation_mode"] == "rotate_then_translate"
        operations = [pick, stow, latch, place, verify, retreat]
        assert all(isinstance(stage, OperateStage) for stage in operations)
        assert [stage.operation for stage in operations] == [
            "pick",
            "return_arm",
            "latch_source_color_memory",
            "place_at_accessible",
            "verify_target_plate",
            "return_arm",
        ]
        assert all(stage.target == item for stage in operations)
        assert latch.metadata["source_color"] == color
        assert (
            place.metadata["target_slot"]
            == verify.metadata["target_slot"]
            == target_slot
        )
        assert (
            place.metadata["source_color"] == verify.metadata["source_color"] == color
        )
        assert stow.metadata["gripper_action"] == 1.0
        assert retreat.metadata["gripper_action"] == -1.0


def test_wr08_color_memory_requires_a_verified_source_and_physical_pick() -> None:
    item = ITEM_IDS[0]
    empty = SemanticState()
    assert not record_item_picked(empty, item, physically_picked=True)
    assert not latch_source_color_memory(empty, item)
    state, _ = _state_and_targets()
    assert not latch_source_color_memory(state, item)
    assert not record_item_picked(state, item, physically_picked=False)
    assert record_item_picked(state, item, physically_picked=True)
    assert latch_source_color_memory(state, item)
    assert not state.source_color_memory_latched_by_item[ITEM_IDS[1]]


def test_wr08_wrong_color_is_a_permanent_protocol_failure() -> None:
    state, targets = _state_and_targets()
    item = ITEM_IDS[0]
    assert record_item_picked(state, item, physically_picked=True)
    assert latch_source_color_memory(state, item)
    wrong = next(slot for slot in TARGET_SLOTS if slot != targets[item])
    assert not record_target_attempt(state, item, wrong, matches=False)
    assert state.protocol_violation
    assert not record_target_placement(state, item, matches=True)
    assert not evaluate_success(
        state,
        PhysicalState(item_in_target=dict.fromkeys(ITEM_IDS, True)),
        target_slot_by_item=targets,
    ).success


def test_wr08_success_requires_both_correctly_placed_and_released() -> None:
    state, targets = _state_and_targets()
    for item in ITEM_IDS:
        assert record_item_picked(state, item, physically_picked=True)
        assert latch_source_color_memory(state, item)
        assert record_target_attempt(state, item, targets[item], matches=True)
        assert record_target_placement(state, item, matches=True)
    physical = dict.fromkeys(ITEM_IDS, True)
    assert evaluate_success(
        state,
        PhysicalState(item_in_target=physical, released=physical),
        target_slot_by_item=targets,
    ).success
    for item in ITEM_IDS:
        report = evaluate_success(
            state,
            PhysicalState(item_in_target=physical, released={**physical, item: False}),
            target_slot_by_item=targets,
        )
        assert not report.success
        assert not report.all_items_released
        report = evaluate_success(
            state,
            PhysicalState(item_in_target={**physical, item: False}, released=physical),
            target_slot_by_item=targets,
        )
        assert not report.success


def test_wr08_physical_observer_reconstructs_transport_without_expert_callbacks() -> (
    None
):
    state, targets, observer = _observer()
    first, second = ITEM_IDS
    _dwell(observer, targets, 1, departed=(first,), held=(first,))
    assert state.item_picked_by_item[first]
    assert state.source_color_memory_latched_by_item[first]

    # Passing over a plate while holding the carton is not a placement.
    _dwell(
        observer,
        targets,
        4,
        departed=(first,),
        held=(first,),
        placed={first: targets[first]},
    )
    assert not state.item_placed_by_item[first]
    _dwell(observer, targets, 7, departed=(first,), placed={first: targets[first]})
    assert state.item_placed_by_item[first]
    _dwell(
        observer,
        targets,
        10,
        departed=ITEM_IDS,
        held=(second,),
        placed={first: targets[first]},
    )
    assert state.item_picked_by_item[second]
    update = _dwell(observer, targets, 13, departed=ITEM_IDS, placed=targets)
    assert update.diagnostics["success"]
    assert observer.phase == "complete"
    assert not state.protocol_violation


def test_wr08_ungrasped_departure_does_not_count_as_picking() -> None:
    state, targets, observer = _observer()
    _dwell(observer, targets, 1, departed=(ITEM_IDS[0],))
    assert not any(state.item_picked_by_item.values())
    assert not any(state.source_color_memory_latched_by_item.values())


def test_wr08_stable_grasp_counts_before_source_radius_departure() -> None:
    state, targets, observer = _observer()
    _dwell(observer, targets, 1, held=(ITEM_IDS[0],))
    assert state.item_picked_by_item[ITEM_IDS[0]]
    assert state.source_color_memory_latched_by_item[ITEM_IDS[0]]


def test_wr08_wrong_plate_crossing_only_fails_after_release_and_dwell() -> None:
    state, targets, observer = _observer()
    item = ITEM_IDS[0]
    wrong = next(slot for slot in TARGET_SLOTS if slot != targets[item])
    _dwell(observer, targets, 1, departed=(item,), held=(item,))
    _dwell(observer, targets, 4, departed=(item,), held=(item,), placed={item: wrong})
    assert not state.protocol_violation
    _dwell(observer, targets, 7, departed=(item,), placed={item: wrong})
    assert state.protocol_violation
    assert not state.item_placed_by_item[item]


def test_wr08_duplicate_observations_do_not_advance_physical_dwell() -> None:
    state, targets, observer = _observer()
    item = ITEM_IDS[0]
    sample = _sample(1, targets, departed=(item,), held=(item,))
    observer.observe(sample)
    for _ in range(5):
        assert observer.observe(sample).duplicate_step
    assert not state.item_picked_by_item[item]
    observer.observe(_sample(2, targets, departed=(item,), held=(item,)))
    assert not state.item_picked_by_item[item]
    observer.observe(_sample(3, targets, departed=(item,), held=(item,)))
    assert state.item_picked_by_item[item]
