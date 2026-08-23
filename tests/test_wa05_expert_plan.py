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
