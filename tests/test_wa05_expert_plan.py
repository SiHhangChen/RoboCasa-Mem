from types import SimpleNamespace

from bench.execution.contracts import NavigateStage, OperateStage
from bench.tasks import TaskCatalog, resolve_builtin_task_plugin
from bench.tasks.contracts import EpisodeRequest, ExpertPlanInputs


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
        assert len(answers) == 1
        assert answers[0].metadata["temperature"] == temperature
        assert all(stage.operation != "record_final_take" for stage in operations)


def test_expert_recording_restores_region_gate_for_evaluation_episode(monkeypatch) -> None:
    from bench.tasks.wa05.expert import operations

    env = SimpleNamespace(require_service_region_for_staging=False)
    descriptor = SimpleNamespace(
        episode=SimpleNamespace(split="evaluation"),
        navigation=object(),
        scene=SimpleNamespace(scene_seed=0),
        config=SimpleNamespace(manipulation={}),
        resources={"planner_script": "planner.py", "cup_grasp_candidates": "grasps.json"},
        execute=True,
        actions=SimpleNamespace(handler_settings={}),
    )
    monkeypatch.setattr(
        operations, "build_navigation_assembly",
        lambda *args, **kwargs: SimpleNamespace(controller=object()),
    )
    monkeypatch.setattr(
        operations, "build_manipulation_assembly",
        lambda *args, **kwargs: SimpleNamespace(controller=SimpleNamespace()),
    )
    monkeypatch.setattr(operations, "StageRunner", lambda **kwargs: SimpleNamespace(**kwargs))

    parts = operations.build_operation_handlers(descriptor, env, {})

    assert env.require_service_region_for_staging is True
    assert parts["runner"] is not None
