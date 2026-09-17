from types import SimpleNamespace

import numpy as np

from bench.execution.contracts import NavigateStage, OperateStage
from bench.tasks import TaskCatalog, resolve_builtin_task_plugin
from bench.tasks.contracts import EpisodeRequest, ExpertPlanInputs
from bench.tasks.wa05.execution.water_operations import place_cup_on_service_slot


def test_expert_plan_uses_one_fill_one_slot_and_matching_answer() -> None:
    for seed, temperature in ((0, "hot"), (1, "cold")):
        episode = TaskCatalog.resolve(EpisodeRequest(task_id="WA-05", seed=seed))
        api = resolve_builtin_task_plugin("WA-05").load_runtime_api()
        descriptor = api.build_descriptor(episode, False)
        plan = TaskCatalog.get("WA-05").build_expert_plan(
            ExpertPlanInputs.from_descriptor(descriptor)
        )

        operations = [
            stage for stage in plan.stages if isinstance(stage, OperateStage)
        ]
        fills = [
            stage for stage in operations
            if stage.operation == "fill_held_under_faucet"
        ]
        answers = [
            stage for stage in operations
            if stage.operation == "press_temperature_answer_button"
        ]
        placements = [
            stage for stage in operations
            if stage.operation == "place_on_service_slot"
        ]
        service_navigation = [
            stage for stage in plan.stages
            if isinstance(stage, NavigateStage)
            and stage.target.station_id == "service_area"
        ]
        button_navigation = [
            stage for stage in plan.stages
            if isinstance(stage, NavigateStage)
            and stage.target.station_id == "water_button_station"
        ]

        assert len(fills) == 1
        assert fills[0].target == "cup_A"
        assert fills[0].metadata["temperature"] == temperature
        assert len(service_navigation) == 1
        assert service_navigation[0].target.purpose == "slot_1"
        assert button_navigation == []
        assert len(placements) == 1
        assert placements[0].metadata["placement_approach_angles_deg"] == [
            180.0,
            165.0,
            150.0,
            135.0,
            124.0,
        ]
        assert descriptor.config.initial_state.objects["service_slots"]["regions"][
            "slot_1"
        ]["region_center_xy"] == [-1.095, -0.5]
        assert len(answers) == 1
        assert answers[0].metadata["temperature"] == temperature
        assert all(stage.operation != "record_final_take" for stage in operations)


def test_expert_recording_restores_region_gate_for_evaluation_episode(monkeypatch) -> None:
    from bench.tasks.wa05.expert import operations

    env = SimpleNamespace(require_service_region_for_staging=False)
    descriptor = SimpleNamespace(
        episode=SimpleNamespace(split="evaluation"),
        navigation=object(),
        episode_plan=SimpleNamespace(seed=0),
        config=SimpleNamespace(manipulation={}),
        resources={"planner_script": "planner.py", "cup_grasp_candidates": "grasps.json"},
        execute=True,
        actions=SimpleNamespace(handler_settings={}),
    )
    monkeypatch.setattr(
        operations, "build_navigation_assembly",
        lambda *args, **kwargs: SimpleNamespace(
            controller=object(),
            station_sampler=SimpleNamespace(route_poses={}),
        ),
    )
    monkeypatch.setattr(
        operations, "build_manipulation_assembly",
        lambda *args, **kwargs: SimpleNamespace(controller=SimpleNamespace()),
    )
    monkeypatch.setattr(operations, "StageRunner", lambda **kwargs: SimpleNamespace(**kwargs))

    parts = operations.build_operation_handlers(descriptor, env, {})

    assert env.require_service_region_for_staging is True
    assert parts["runner"] is not None


def test_service_place_releases_before_the_independent_physical_check() -> None:
    class FakeEnv:
        def service_slot_position(self, slot: str) -> np.ndarray:
            assert slot == "slot_1"
            return np.asarray([2.0, -2.0, 0.9], dtype=float)

    class FakeController:
        def __init__(self) -> None:
            self.runtime_config = SimpleNamespace(execute=True)
            self.env = FakeEnv()
            self.call: dict[str, object] | None = None

        def place_at_world_pose_candidates(
            self,
            object_name: str,
            **kwargs: object,
        ) -> dict[str, object]:
            self.call = {"object_name": object_name, **kwargs}
            return {"pass": True}

    controller = FakeController()
    result = place_cup_on_service_slot(
        controller,
        cup_id="cup_A",
        slot="slot_1",
        require_pre_release_check=True,
        pre_release_min_height_from_slot=0.0,
        pre_release_max_height_from_slot=0.1,
        placement_approach_angles_deg=[180.0, 165.0, 150.0, 135.0, 124.0],
        placement_reference="bbox_bottom_center",
        placement_object_upright=True,
    )

    assert result["pass"] is True
    assert result["semantic_event_deferred"] == "record_verified_staging"
    assert controller.call is not None
    assert "pre_release_check" not in controller.call
    assert "placement_approach_angles_deg" not in controller.call
    candidate_groups = controller.call["candidate_groups"]
    assert isinstance(candidate_groups, list)
    assert len(candidate_groups) == 1
    candidates = candidate_groups[0]["candidates"]
    assert [candidate["placement_approach_angle_deg"] for candidate in candidates] == [
        180.0,
        165.0,
        150.0,
        135.0,
        124.0,
    ]
    assert [candidate["metadata"]["fixed_candidate_order"] for candidate in candidates] == [
        0,
        1,
        2,
        3,
        4,
    ]
    assert all(
        candidate["target_pos_world"] == [2.0, -2.0, 0.9]
        and candidate["placement_reference"] == "bbox_bottom_center"
        and candidate["placement_object_upright"] is True
        for candidate in candidates
    )
