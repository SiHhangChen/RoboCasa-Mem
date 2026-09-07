from __future__ import annotations

import numpy as np

from bench.tasks import TaskCatalog
from bench.tasks.contracts import EpisodeRequest, ExpertPlanInputs
from bench.tasks.wx02.config import load_task_config, resolve_action_config
from bench.tasks.wx02.domain.contracts import OBJECT_NAMES, PRIMARY_CUP, SECONDARY_CUP
from bench.tasks.wx02.domain.episode import (
    resolve_episode_plan,
    resolve_scene_config,
    resolve_task_definition,
)
from bench.tasks.wx02.domain.semantics import (
    PhysicalState,
    SemanticState,
    evaluate_success,
    record_object_moved,
    record_object_returned,
    record_stain_cleaned,
    record_tool_returned,
)
from bench.tasks.wx02.evaluation import WX02EvaluationSnapshot, WX02Evaluator
from bench.tasks.wx02.expert.spec import build_expert_plan


EXPECTED_STAGE_NAMES = (
    "stand_in_front_of_two_cups",
    "navigate_to_and_pick_cleaning_tool",
    "return_to_cups_and_stage_cleaning_tool",
    "move_primary_cup_to_reveal_stain",
    "pick_tool_and_wipe_primary_stain",
    "place_cleaning_tool_after_primary_wipe",
    "pick_displaced_primary_cup",
    "return_primary_cup_to_remembered_origin",
    "move_secondary_cup_to_reveal_stain",
    "pick_tool_and_wipe_secondary_stain",
    "place_cleaning_tool_after_secondary_wipe",
    "pick_displaced_secondary_cup",
    "return_secondary_cup_to_remembered_origin",
    "return_cleaning_tool_to_yellow_region",
)


def _episode_plan(seed: int = 7):
    config = load_task_config()
    return resolve_episode_plan(
        seed=seed,
        episode=config.episode,
        object_settings={
            name: config.initial_state.objects[name] for name in OBJECT_NAMES
        },
    )


def _expert_plan(seed: int = 7):
    config = load_task_config()
    episode = TaskCatalog.resolve(EpisodeRequest(task_id="WX-02", seed=seed))
    episode_plan = _episode_plan(seed)
    actions = resolve_action_config(
        episode=episode_plan,
        operations=config.operations,
    )
    scene = resolve_scene_config(seed=seed, env_split="pretrain", **config.scene)
    return build_expert_plan(
        ExpertPlanInputs(episode, scene, episode_plan, actions, False)
    )


def _successful_physical_state(**overrides: bool) -> PhysicalState:
    values = {
        "primary_cup_at_origin": True,
        "primary_cup_on_dining_table": True,
        "secondary_cup_at_origin": True,
        "secondary_cup_on_dining_table": True,
        "cleaning_tool_in_return_region": True,
        "cleaning_tool_on_dining_table": True,
        "cleaning_tool_released": True,
        "primary_cup_released": True,
        "secondary_cup_released": True,
    }
    values.update(overrides)
    return PhysicalState(**values)


def test_wx02_owns_the_requested_fourteen_stage_plan() -> None:
    plan = _expert_plan()

    assert tuple(stage.name for stage in plan.stages) == EXPECTED_STAGE_NAMES
    assert tuple(stage.operation for stage in plan.stages) == (
        "return_arm",
        "fetch_cleaning_tool",
        "return_and_stage_cleaning_tool",
        "move_object",
        "pick_tool_and_wipe",
        "place_at_accessible",
        "pick",
        "place_at_accessible",
        "move_object",
        "pick_tool_and_wipe",
        "place_at_accessible",
        "pick",
        "place_at_accessible",
        "return_cleaning_tool_to_region",
    )


def test_wx02_starts_at_cup_front_with_no_setup_pick() -> None:
    config = load_task_config()

    assert config.initial_state.robot_base.sample_index == 1
    assert config.initial_state.short_start.mode == "cup_front_unheld"
    assert config.initial_state.short_start.setup_operation == "none"


def test_wx02_samples_two_origins_and_independent_world_temporary_targets() -> None:
    config = load_task_config()
    first = _episode_plan(0)
    second = _episode_plan(1)
    temporary = config.episode["temporary_targets"]
    center = np.asarray(temporary["object_region_center_xy"], dtype=float)
    half = 0.5 * np.asarray(temporary["object_region_size_xy"], dtype=float)

    assert set(first.initial_object_poses) == set(OBJECT_NAMES)
    assert set(first.temporary_object_xy) == {PRIMARY_CUP, SECONDARY_CUP}
    for plan in (first, second):
        for cup in (PRIMARY_CUP, SECONDARY_CUP):
            target = np.asarray(plan.temporary_object_xy[cup], dtype=float)
            assert np.all(np.abs(target - center) <= half + 1e-12)

    first_offset = np.asarray(first.temporary_object_xy[PRIMARY_CUP]) - np.asarray(
        first.initial_object_poses[PRIMARY_CUP].xy
    )
    second_offset = np.asarray(second.temporary_object_xy[PRIMARY_CUP]) - np.asarray(
        second.initial_object_poses[PRIMARY_CUP].xy
    )
    assert not np.allclose(first_offset, second_offset)
    assert first.wipe_metadata_by_object[PRIMARY_CUP]["seed"] == 2000
    assert first.wipe_metadata_by_object[SECONDARY_CUP]["seed"] == 3000


def test_wx02_yellow_region_is_the_sponge_origin_and_final_region() -> None:
    config = load_task_config()
    episode = _episode_plan(0)
    center = np.asarray(
        config.scene["cleaning_tool_return_region_center_xy"], dtype=float
    )
    size = np.asarray(
        config.scene["cleaning_tool_return_region_size_xy"], dtype=float
    )
    tool_xy = np.asarray(episode.initial_object_poses["cleaning_tool"].xy)
    staged_xy = np.asarray(episode.staged_tool_xy)
    tolerance = float(config.semantics["tool_return_region_tolerance"])

    assert np.allclose(size, [0.22, 0.20])
    assert np.all(np.abs(tool_xy - center) <= size * 0.5)
    assert not np.all(np.abs(staged_xy - center) <= size * 0.5 + tolerance)


def test_wx02_has_two_irregular_collision_free_fading_stains() -> None:
    config = load_task_config()
    spots = config.semantics["stain_spots"]
    source = (config.source_path.parents[1] / "environment.py").read_text(
        encoding="utf-8"
    )

    assert len(spots) >= 5
    assert all(len(spot) == 5 for spot in spots)
    assert "wx02_primary_cup_stain_spot_" in source
    assert "wx02_secondary_cup_stain_spot_" in source
    assert '"contype": "0"' in source
    assert '"conaffinity": "0"' in source
    assert "model.geom_rgba[geom_id][3]" in source


def test_wx02_evaluator_qualifies_the_two_ordered_memory_loops() -> None:
    evaluator = WX02Evaluator()
    evaluator.reset(WX02EvaluationSnapshot(step_id=0))
    snapshots = [
        WX02EvaluationSnapshot(step_id=1, cleaning_tool_acquired=True),
        WX02EvaluationSnapshot(
            step_id=2, cleaning_tool_acquired=True, primary_cup_moved=True
        ),
        WX02EvaluationSnapshot(
            step_id=3,
            cleaning_tool_acquired=True,
            primary_cup_moved=True,
            primary_stain_cleaned=True,
        ),
        WX02EvaluationSnapshot(
            step_id=4,
            cleaning_tool_acquired=True,
            primary_cup_moved=True,
            primary_stain_cleaned=True,
            primary_cup_returned=True,
        ),
        WX02EvaluationSnapshot(
            step_id=5,
            cleaning_tool_acquired=True,
            primary_cup_moved=True,
            primary_stain_cleaned=True,
            primary_cup_returned=True,
            secondary_cup_moved=True,
        ),
        WX02EvaluationSnapshot(
            step_id=6,
            cleaning_tool_acquired=True,
            primary_cup_moved=True,
            primary_stain_cleaned=True,
            primary_cup_returned=True,
            secondary_cup_moved=True,
            secondary_stain_cleaned=True,
        ),
        WX02EvaluationSnapshot(
            step_id=7,
            cleaning_tool_acquired=True,
            primary_cup_moved=True,
            primary_stain_cleaned=True,
            primary_cup_returned=True,
            secondary_cup_moved=True,
            secondary_stain_cleaned=True,
            secondary_cup_returned=True,
        ),
    ]
    for snapshot in snapshots:
        result = evaluator.observe(snapshot)
        assert not result.success

    result = evaluator.observe(
        WX02EvaluationSnapshot(
            step_id=8,
            cleaning_tool_acquired=True,
            primary_cup_moved=True,
            primary_stain_cleaned=True,
            primary_cup_returned=True,
            secondary_cup_moved=True,
            secondary_stain_cleaned=True,
            secondary_cup_returned=True,
            cleaning_tool_returned=True,
            cleaning_tool_released=True,
            primary_cup_at_origin=True,
            primary_cup_on_dining_table=True,
            secondary_cup_at_origin=True,
            secondary_cup_on_dining_table=True,
            cleaning_tool_in_return_region=True,
            cleaning_tool_on_dining_table=True,
            primary_cup_released=True,
            secondary_cup_released=True,
            domain_success=True,
        )
    )

    assert result.success
    assert {
        name: result.stages[name].qualified_step for name in result.stages
    } == {
        "tool_acquired": 1,
        "primary_cup_moved": 2,
        "primary_stain_cleaned": 3,
        "primary_cup_returned": 4,
        "secondary_cup_moved": 5,
        "secondary_stain_cleaned": 6,
        "secondary_cup_returned": 7,
        "tool_returned": 8,
        "success": 8,
    }


def test_wx02_domain_rejects_out_of_order_second_loop() -> None:
    state = SemanticState(cleaning_tool_acquired=True)

    assert not record_object_moved(state, SECONDARY_CUP)
    assert record_object_moved(state, PRIMARY_CUP)
    assert not record_stain_cleaned(state, SECONDARY_CUP)
    assert record_stain_cleaned(state, PRIMARY_CUP)
    assert record_object_returned(state, PRIMARY_CUP)
    assert record_object_moved(state, SECONDARY_CUP)


def test_wx02_tool_return_requires_a_post_cleanup_return_edge() -> None:
    state = SemanticState(
        cleaning_tool_acquired=True,
        cleaning_tool_left_return_region=True,
        primary_cup_moved=True,
        primary_stain_cleaned=True,
        primary_cup_returned=True,
        secondary_cup_moved=True,
        secondary_stain_cleaned=True,
    )

    assert not record_tool_returned(state)
    assert record_object_returned(state, SECONDARY_CUP)
    assert record_tool_returned(state)
    source = (load_task_config().source_path.parents[1] / "environment.py").read_text(
        encoding="utf-8"
    )
    assert "previous is False and tool_stable" in source


def test_wx02_success_requires_both_cups_and_the_final_tool_return() -> None:
    state = SemanticState(
        cleaning_tool_acquired=True,
        primary_cup_moved=True,
        primary_stain_cleaned=True,
        primary_cup_returned=True,
        secondary_cup_moved=True,
        secondary_stain_cleaned=True,
        secondary_cup_returned=True,
        cleaning_tool_left_return_region=True,
        cleaning_tool_returned=False,
    )

    assert not evaluate_success(state, _successful_physical_state()).success
    state.cleaning_tool_returned = True
    assert evaluate_success(state, _successful_physical_state()).success


def test_wx02_declared_success_conditions_match_runtime_checks() -> None:
    config = load_task_config()
    definition = resolve_task_definition(**config.task)
    state = SemanticState(
        cleaning_tool_acquired=True,
        primary_cup_moved=True,
        primary_stain_cleaned=True,
        primary_cup_returned=True,
        secondary_cup_moved=True,
        secondary_stain_cleaned=True,
        secondary_cup_returned=True,
        cleaning_tool_left_return_region=True,
        cleaning_tool_returned=True,
    )
    report = evaluate_success(state, _successful_physical_state())

    assert report.success
    assert definition.success_conditions == tuple(report.checks())
