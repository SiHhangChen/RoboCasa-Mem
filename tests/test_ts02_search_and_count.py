from __future__ import annotations

import xml.etree.ElementTree as ET

import pytest

from bench.execution.contracts import OperateStage
from bench.tasks import TaskCatalog
from bench.tasks.contracts import EpisodeRequest, ExpertPlanInputs
from bench.tasks.ts02.button_asset import (
    TS02_BUTTON_FIXED_PARTS,
    TS02_BUTTON_GLOW_RGBA,
    TS02_BUTTON_MAX_TRAVEL_M,
    TS02_BUTTON_RGBA,
    add_ts02_button_asset,
    ts02_button_geom_names,
)
from bench.tasks.ts02.config import load_task_config
from bench.tasks.ts02.domain.contracts import FIXTURE_NAMES
from bench.tasks.ts02.domain.episode import (
    resolve_episode_plan,
    sample_demo_fixture_order,
    sample_target_fixture,
)
from bench.tasks.ts02.domain.semantics import (
    PhysicalState,
    SemanticState,
    evaluate_success,
    record_close_attempt,
    record_completion_button_attempt,
    record_inspection_attempt,
    record_open_attempt,
)
from bench.tasks.ts02.physical_events import PhysicalSample, TS02PhysicalObserver
from bench.tasks.ts02.target_object import (
    TARGET_OBJECT_DRAWER_POS,
    TARGET_OBJECT_GEOM_NAME,
    TARGET_OBJECT_HALF_SIZE_M,
    TARGET_OBJECT_RGBA,
    add_target_red_cube,
)


def _closed_physical_state() -> PhysicalState:
    return PhysicalState({name: 0.0 for name in FIXTURE_NAMES})


def _complete_drawer_cycle(state: SemanticState, fixture: str) -> None:
    assert record_open_attempt(state, fixture, drawer_open=True).passed
    assert record_inspection_attempt(state, fixture, drawer_open=True).passed
    assert record_close_attempt(state, fixture, drawer_closed=True).passed


@pytest.mark.parametrize("search_count", (1, 2, 3))
def test_success_requires_one_button_press_per_inspected_drawer(
    search_count: int,
) -> None:
    order = FIXTURE_NAMES
    target = order[search_count - 1]
    state = SemanticState(order, target)
    physical = _closed_physical_state()

    for fixture in order[:search_count]:
        _complete_drawer_cycle(state, fixture)

    assert state.target_found
    assert state.required_button_presses == search_count
    for press_index in range(search_count):
        assert record_completion_button_attempt(
            state,
            all_drawers_closed=True,
            button_physically_pressed=True,
        ).passed
        assert evaluate_success(state, physical).success is (
            press_index + 1 == search_count
        )


def test_search_must_stop_after_target_drawer() -> None:
    state = SemanticState(FIXTURE_NAMES, FIXTURE_NAMES[0])
    _complete_drawer_cycle(state, FIXTURE_NAMES[0])

    report = record_open_attempt(state, FIXTURE_NAMES[1], drawer_open=True)

    assert not report.passed
    assert report.reason == "search_continued_after_target_found"
    assert state.search_after_target_detected
    assert not evaluate_success(state, _closed_physical_state()).success


def test_extra_button_press_invalidates_exact_count() -> None:
    state = SemanticState(FIXTURE_NAMES, FIXTURE_NAMES[0])
    _complete_drawer_cycle(state, FIXTURE_NAMES[0])
    assert record_completion_button_attempt(
        state,
        all_drawers_closed=True,
        button_physically_pressed=True,
    ).passed

    report = record_completion_button_attempt(
        state,
        all_drawers_closed=True,
        button_physically_pressed=True,
    )

    assert not report.passed
    assert report.reason == "completion_button_pressed_too_many_times"
    assert state.completion_button_press_count == 2
    assert not evaluate_success(state, _closed_physical_state()).success


def test_target_sampling_is_deterministic_and_separate_from_search_order() -> None:
    sampled = [sample_target_fixture(seed) for seed in range(60)]
    assert sampled == [sample_target_fixture(seed) for seed in range(60)]
    assert set(sampled) == set(FIXTURE_NAMES)
    assert any(
        sample_demo_fixture_order(seed)[0] != sample_target_fixture(seed)
        for seed in range(60)
    )

    configured = resolve_episode_plan(
        seed=9,
        demo_fixture_order=("left", "center", "right"),
        target_fixture="center",
        max_horizon=2200,
    )
    assert configured.demo_fixture_order == FIXTURE_NAMES
    assert configured.target_fixture == FIXTURE_NAMES[1]


def test_expert_plan_stops_at_target_and_presses_k_times() -> None:
    episode = TaskCatalog.resolve(
        EpisodeRequest(task_id="TS02", seed=2, variant="memory")
    )
    from bench.tasks.ts02.runtime import build_descriptor

    descriptor = build_descriptor(episode, execute=False)
    plan = TaskCatalog.get("TS02").build_expert_plan(
        ExpertPlanInputs.from_descriptor(descriptor)
    )
    target = str(episode.facts["target_fixture"])
    full_order = tuple(str(item) for item in episode.facts["demo_fixture_order"])
    search_count = full_order.index(target) + 1
    operations = [
        stage.operation for stage in plan.stages if isinstance(stage, OperateStage)
    ]

    assert operations.count("open_fixture") == search_count
    assert operations.count("inspect_fixture") == search_count
    assert operations.count("close_fixture") == search_count
    assert operations.count("press_completion_button") == search_count
    assert plan.metadata["search_count"] == search_count


def test_physical_observer_debounces_a_long_button_hold() -> None:
    observer = TS02PhysicalObserver(
        fixture_names=FIXTURE_NAMES,
        target_fixture=FIXTURE_NAMES[0],
        drawer_open_threshold=0.70,
        drawer_closed_threshold=0.08,
        button_pressed_threshold_m=0.004,
        button_released_threshold_m=0.001,
        button_release_dwell_steps=2,
    )

    def sample(step: int, drawer: float, button: float) -> PhysicalSample:
        fractions = {name: 0.0 for name in FIXTURE_NAMES}
        fractions[FIXTURE_NAMES[0]] = drawer
        return PhysicalSample(step, fractions, button)

    observer.reset(sample(0, 0.0, 0.0))
    observer.observe(sample(1, 0.75, 0.0))
    observer.observe(sample(2, 0.0, 0.0))
    assert observer.state.target_found

    observer.observe(sample(3, 0.0, 0.006))
    observer.observe(sample(4, 0.0, 0.006))
    observer.observe(sample(5, 0.0, 0.006))

    assert observer.state.completion_button_press_count == 1
    assert observer.state.completion_button_pressed


def test_physical_observer_requires_release_before_the_next_press() -> None:
    observer = TS02PhysicalObserver(
        fixture_names=FIXTURE_NAMES,
        target_fixture=FIXTURE_NAMES[1],
        drawer_open_threshold=0.70,
        drawer_closed_threshold=0.08,
        button_pressed_threshold_m=0.004,
        button_released_threshold_m=0.001,
        button_release_dwell_steps=2,
    )

    def sample(
        step: int,
        first_drawer: float,
        second_drawer: float,
        button: float,
    ) -> PhysicalSample:
        return PhysicalSample(
            step,
            {
                FIXTURE_NAMES[0]: first_drawer,
                FIXTURE_NAMES[1]: second_drawer,
                FIXTURE_NAMES[2]: 0.0,
            },
            button,
        )

    observer.reset(sample(0, 0.0, 0.0, 0.0))
    observer.observe(sample(1, 0.75, 0.0, 0.0))
    observer.observe(sample(2, 0.0, 0.0, 0.0))
    observer.observe(sample(3, 0.0, 0.75, 0.0))
    observer.observe(sample(4, 0.0, 0.0, 0.0))
    assert observer.state.required_button_presses == 2

    observer.observe(sample(5, 0.0, 0.0, 0.006))
    observer.observe(sample(6, 0.0, 0.0, 0.0))
    observer.observe(sample(7, 0.0, 0.0, 0.006))
    assert observer.state.completion_button_press_count == 1

    observer.observe(sample(8, 0.0, 0.0, 0.0))
    observer.observe(sample(9, 0.0, 0.0, 0.0))
    observer.observe(sample(10, 0.0, 0.0, 0.006))
    assert observer.state.completion_button_press_count == 2
    assert observer.state.completion_button_pressed


def test_red_cube_is_attached_only_to_the_target_drawer_body() -> None:
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
    assert TARGET_OBJECT_DRAWER_POS[2] - TARGET_OBJECT_HALF_SIZE_M == pytest.approx(
        -0.062
    )
    assert root.find(".//body[@name='drawer_a']/geom") is None
    assert (
        root.find(f".//body[@name='drawer_b']/geom[@name='{TARGET_OBJECT_GEOM_NAME}']")
        is cube
    )


def test_completion_button_uses_tm01_decorative_style() -> None:
    root = ET.fromstring("<mujoco><worldbody/></mujoco>")
    worldbody = root.find("worldbody")
    assert worldbody is not None

    geom_name = "bench_ts02_completion_button"
    joint_name = f"{geom_name}_press_joint"
    add_ts02_button_asset(
        worldbody,
        geom_name=geom_name,
        pos=(1.0, 2.0, 3.0),
        joint_name=joint_name,
    )

    installed = {
        geom.get("name") for geom in root.iter("geom") if geom.get("name")
    }
    assert installed == set(ts02_button_geom_names(geom_name))
    assert len(TS02_BUTTON_FIXED_PARTS) == 10
    assert root.find(f".//geom[@name='{geom_name}_raised_shoulder']") is not None
    assert root.find(f".//geom[@name='{geom_name}_faceplate']") is not None
    assert root.find(f".//geom[@name='{geom_name}_glow_ring_outer']").get(
        "rgba"
    ) == TS02_BUTTON_GLOW_RGBA
    assert root.find(f".//geom[@name='{geom_name}_plunger_stem']").get(
        "rgba"
    ) == TS02_BUTTON_RGBA
    assert root.find(f".//joint[@name='{joint_name}']").get("range") == (
        f"-{TS02_BUTTON_MAX_TRAVEL_M:.3f} 0"
    )


def test_task_config_exposes_button_release_debounce() -> None:
    config = load_task_config()
    button = config.operations["completion_button"]
    assert button["release_depth_threshold"] < button["press_depth_threshold"]
    assert button["release_dwell_steps"] >= 1
