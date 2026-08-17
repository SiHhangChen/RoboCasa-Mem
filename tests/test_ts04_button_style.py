from __future__ import annotations

import xml.etree.ElementTree as ET

import pytest

from bench.execution.contracts import OperateStage
from bench.tasks import TaskCatalog, resolve_builtin_task_plugin
from bench.tasks.contracts import EpisodeRequest, ExpertPlanInputs
from bench.tasks.ts04.button_asset import (
    TS04_BUTTON_FIXED_PARTS,
    TS04_BUTTON_GLOW_RGBA,
    TS04_BUTTON_MAX_TRAVEL_M,
    TS04_BUTTON_RGBA,
    add_ts04_button_asset,
    ts04_button_geom_names,
)
from bench.tasks.ts04.domain.contracts import FIXTURE_NAMES
from bench.tasks.ts04.domain.episode import sample_target_fixture
from bench.tasks.ts04.domain.semantics import (
    PhysicalState,
    SemanticState,
    evaluate_success,
    record_close_attempt,
    record_completion_button_attempt,
    record_inspection_attempt,
    record_open_attempt,
)
from bench.tasks.ts04.target_object import (
    TARGET_OBJECT_DRAWER_POS,
    TARGET_OBJECT_GEOM_NAME,
    TARGET_OBJECT_HALF_SIZE_M,
    TARGET_OBJECT_RGBA,
    add_target_red_cube,
)


def test_ts04_completion_button_uses_tm01_decorative_style() -> None:
    root = ET.fromstring("<mujoco><worldbody/></mujoco>")
    worldbody = root.find("worldbody")
    assert worldbody is not None

    geom_name = "bench_ts04_completion_button"
    joint_name = f"{geom_name}_press_joint"
    add_ts04_button_asset(
        worldbody,
        geom_name=geom_name,
        pos=(1.0, 2.0, 3.0),
        joint_name=joint_name,
    )

    installed = {
        geom.get("name") for geom in root.iter("geom") if geom.get("name")
    }
    assert installed == set(ts04_button_geom_names(geom_name))
    assert len(TS04_BUTTON_FIXED_PARTS) == 10
    assert root.find(f".//geom[@name='{geom_name}_raised_shoulder']") is not None
    assert root.find(f".//geom[@name='{geom_name}_faceplate']") is not None
    assert root.find(f".//geom[@name='{geom_name}_glow_ring_outer']").get(
        "rgba"
    ) == TS04_BUTTON_GLOW_RGBA
    assert root.find(f".//geom[@name='{geom_name}_plunger_stem']").get(
        "rgba"
    ) == TS04_BUTTON_RGBA
    assert root.find(f".//joint[@name='{joint_name}']").get("range") == (
        f"-{TS04_BUTTON_MAX_TRAVEL_M:.3f} 0"
    )


def test_ts04_red_cube_is_ten_centimetres_and_attached_to_one_drawer() -> None:
    root = ET.fromstring(
        "<mujoco><worldbody>"
        "<body name='drawer_a'/><body name='drawer_b'/>"
        "</worldbody></mujoco>"
    )
    add_target_red_cube(root, drawer_body_name="drawer_b")

    cube = root.find(f".//geom[@name='{TARGET_OBJECT_GEOM_NAME}']")
    assert cube is not None
    assert cube.get("rgba") == TARGET_OBJECT_RGBA
    assert cube.get("size") == " ".join(
        f"{TARGET_OBJECT_HALF_SIZE_M:.6f}" for _ in range(3)
    )
    assert cube.get("pos") == " ".join(
        f"{value:.6f}" for value in TARGET_OBJECT_DRAWER_POS
    )
    assert TARGET_OBJECT_HALF_SIZE_M * 2 == pytest.approx(0.10)
    assert TARGET_OBJECT_DRAWER_POS[2] - TARGET_OBJECT_HALF_SIZE_M == pytest.approx(
        -0.062
    )
    assert root.find(".//body[@name='drawer_a']/geom") is None
    assert (
        root.find(
            f".//body[@name='drawer_b']/geom[@name='{TARGET_OBJECT_GEOM_NAME}']"
        )
        is cube
    )


def test_ts04_target_drawer_is_seeded_and_passed_to_the_environment() -> None:
    sampled = [sample_target_fixture(seed) for seed in range(20)]
    assert sampled == [sample_target_fixture(seed) for seed in range(20)]
    assert set(sampled) == set(FIXTURE_NAMES)

    episode = TaskCatalog.resolve(EpisodeRequest(task_id="TS04", seed=0))
    api = resolve_builtin_task_plugin("TS04").load_runtime_api()
    descriptor = api.build_descriptor(episode, False)

    assert episode.facts["target_fixture"] in FIXTURE_NAMES
    assert descriptor.environment.env_kwargs["target_fixture"] == (
        episode.facts["target_fixture"]
    )


def _complete_drawer_cycle(state: SemanticState, fixture: str) -> None:
    assert record_open_attempt(state, fixture, drawer_open=True).passed
    assert record_inspection_attempt(state, fixture, drawer_open=True).passed
    assert record_close_attempt(state, fixture, drawer_closed=True).passed


def test_ts04_stops_searching_after_target_and_requires_one_button_press() -> None:
    target, remaining = FIXTURE_NAMES
    state = SemanticState(FIXTURE_NAMES, target)
    closed = PhysicalState({name: 0.0 for name in FIXTURE_NAMES})

    _complete_drawer_cycle(state, target)
    assert state.target_found
    assert state.completed_fixture_history == [target]
    assert record_completion_button_attempt(
        state,
        all_drawers_closed=True,
        button_physically_pressed=True,
    ).passed
    assert evaluate_success(state, closed).success

    report = record_open_attempt(state, remaining, drawer_open=True)
    assert not report.passed
    assert report.reason == "search_continued_after_target_found"
    assert state.search_after_target_detected
    assert not evaluate_success(state, closed).success


@pytest.mark.parametrize("seed", (0, 2))
def test_ts04_expert_plan_ends_at_target(seed: int) -> None:
    episode = TaskCatalog.resolve(EpisodeRequest(task_id="TS04", seed=seed))
    api = resolve_builtin_task_plugin("TS04").load_runtime_api()
    descriptor = api.build_descriptor(episode, False)
    plan = TaskCatalog.get("TS04").build_expert_plan(
        ExpertPlanInputs.from_descriptor(descriptor)
    )
    full_order = tuple(episode.facts["demo_fixture_order"])
    target = str(episode.facts["target_fixture"])
    search_count = full_order.index(target) + 1
    operations = [
        stage.operation for stage in plan.stages if isinstance(stage, OperateStage)
    ]

    assert operations.count("open_fixture") == search_count
    assert operations.count("inspect_fixture") == search_count
    assert operations.count("close_fixture") == search_count
    assert operations.count("press_completion_button") == 1
    assert plan.metadata["fixture_order"] == list(full_order[:search_count])
    assert plan.metadata["target_fixture"] == target
    assert plan.metadata["search_count"] == search_count
