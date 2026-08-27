from __future__ import annotations

import json
import xml.etree.ElementTree as ET

import pytest

from bench.execution.contracts import OperateStage
from bench.execution.contracts import NavigateStage
from bench.tasks import TaskCatalog, resolve_builtin_task_plugin
from bench.tasks.contracts import EpisodeRequest, ExpertPlanInputs
from bench.tasks.ts02.config import load_task_config as load_ts02_task_config
from bench.tasks.ts04.button_asset import (
    TS04_BUTTON_FIXED_PARTS,
    TS04_BUTTON_GLOW_RGBA,
    TS04_BUTTON_MAX_TRAVEL_M,
    TS04_BUTTON_RGBA,
    add_ts04_button_asset,
    ts04_button_geom_names,
)
from bench.tasks.ts04.config import load_task_config
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
from bench.tasks.ts04.physical_events import PhysicalSample, TS04PhysicalObserver
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


def test_ts04_completion_button_is_right_of_all_drawers_and_navigation_matches() -> None:
    config = load_task_config()
    button_xyz = tuple(config.scene["completion_button_world_xyz"])
    drawer_xyzs = tuple(tuple(row) for row in config.scene["drawer_world_xyzs"])
    assert button_xyz == pytest.approx((3.85, -2.06, 0.92))
    assert button_xyz[0] < min(row[0] for row in drawer_xyzs)

    station_path = config.path(config.resources["station_poses"])
    route_path = config.path(config.resources["manual_base_path"])
    assert station_path is not None
    assert route_path is not None
    stations = json.loads(station_path.read_text(encoding="utf-8"))
    route = json.loads(route_path.read_text(encoding="utf-8"))
    completion_station = stations["stations"]["completion_button_station"]["samples"][0]
    completion_route = next(
        sample
        for sample in route["samples"]
        if sample["name"] == "completion_button_station"
    )
    assert tuple(stations["completion_button_world_xyz"]) == button_xyz
    assert tuple(completion_station["button_world_xyz"]) == button_xyz
    assert completion_station["base_xy"][0] == pytest.approx(button_xyz[0])
    assert completion_route["xy"][0] == pytest.approx(button_xyz[0])


def test_ts04_drawers_match_ts02_compact_spacing_and_shared_station() -> None:
    ts04 = load_task_config()
    ts02 = load_ts02_task_config()
    drawer_xyzs = tuple(tuple(row) for row in ts04.scene["drawer_world_xyzs"])
    ts02_drawer_xyzs = tuple(tuple(row) for row in ts02.scene["drawer_world_xyzs"])
    expected = ((4.60, -2.37, 1.1363333333333332), (4.85, -2.37, 1.1363333333333332))
    assert all(
        actual == pytest.approx(want)
        for actual, want in zip(drawer_xyzs, expected)
    )
    assert all(
        actual == pytest.approx(want)
        for actual, want in zip(drawer_xyzs, ts02_drawer_xyzs[:2])
    )
    assert drawer_xyzs[1][0] - drawer_xyzs[0][0] == pytest.approx(0.25)
    station_path = ts04.path(ts04.resources["station_poses"])
    assert station_path is not None
    stations = json.loads(station_path.read_text(encoding="utf-8"))
    drawer_station = stations["stations"]["drawer_station"]["samples"][0]
    assert drawer_station["base_xy"] == pytest.approx([4.85, -1.37])


def test_ts02_and_ts04_share_accelerated_safe_arm_profile() -> None:
    ts02 = load_ts02_task_config()
    ts04 = load_task_config()
    for config in (ts02, ts04):
        runtime = config.manipulation["runtime"]
        assert runtime["max_joint_step"] == pytest.approx(0.025)
        assert runtime["settle_steps"] == 10


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


@pytest.mark.parametrize("search_count", (1, 2))
def test_ts04_stops_searching_after_target_and_requires_search_length_presses(
    search_count: int,
) -> None:
    target = FIXTURE_NAMES[search_count - 1]
    remaining = (
        FIXTURE_NAMES[search_count] if search_count < len(FIXTURE_NAMES) else None
    )
    state = SemanticState(FIXTURE_NAMES, target)
    closed = PhysicalState({name: 0.0 for name in FIXTURE_NAMES})

    for fixture in FIXTURE_NAMES[:search_count]:
        _complete_drawer_cycle(state, fixture)
    assert state.target_found
    assert state.required_button_presses == search_count
    for press_index in range(search_count):
        assert record_completion_button_attempt(
            state,
            all_drawers_closed=True,
            button_physically_pressed=True,
        ).passed
        assert evaluate_success(state, closed).success is (
            press_index + 1 == search_count
        )
    assert evaluate_success(state, closed).success

    if remaining is not None:
        report = record_open_attempt(state, remaining, drawer_open=True)
        assert not report.passed
        assert report.reason == "search_continued_after_target_found"
        assert state.search_after_target_detected
        assert not evaluate_success(state, closed).success


def test_ts04_extra_button_press_invalidates_exact_search_count() -> None:
    state = SemanticState(FIXTURE_NAMES, FIXTURE_NAMES[0])
    closed = PhysicalState({name: 0.0 for name in FIXTURE_NAMES})
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
    assert not evaluate_success(state, closed).success


def test_ts04_physical_observer_requires_release_between_search_count_presses() -> None:
    observer = TS04PhysicalObserver(
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


def test_ts04_config_exposes_button_release_debounce() -> None:
    button = load_task_config().operations["completion_button"]
    assert button["release_depth_threshold"] < button["press_depth_threshold"]
    assert button["release_dwell_steps"] >= 1


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
    assert operations.count("retreat_and_return_arm_after_drawer") == search_count
    assert operations.count("press_completion_button") == search_count
    navigation_targets = [
        stage.target.station_id
        for stage in plan.stages
        if isinstance(stage, NavigateStage)
    ]
    assert navigation_targets[:search_count] == ["drawer_station"] * search_count
    assert navigation_targets[-1] == "completion_button_station"
    assert plan.metadata["fixture_order"] == list(full_order[:search_count])
    assert plan.metadata["target_fixture"] == target
    assert plan.metadata["search_count"] == search_count
    assert plan.metadata["required_button_presses"] == search_count
