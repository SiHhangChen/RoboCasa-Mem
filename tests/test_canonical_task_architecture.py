from __future__ import annotations

import importlib
from pathlib import Path

import pytest

from bench.execution.contracts import ExpertPlan, NavigateStage, OperateStage
from bench.tasks import TaskCatalog, resolve_builtin_task_plugin
from bench.tasks.contracts import EpisodeRequest, ExpertPlanInputs, ResolvedEpisode


TASK_IDS = TaskCatalog.ids()


@pytest.mark.parametrize("task_id", TASK_IDS)
def test_catalog_resolves_one_immutable_episode_and_expert_plan(task_id: str) -> None:
    episode = TaskCatalog.resolve(
        EpisodeRequest(task_id=task_id, seed=17, allow_partial_calibration=True)
    )
    api = resolve_builtin_task_plugin(task_id).load_runtime_api()
    descriptor = api.build_descriptor(episode, False)
    inputs = ExpertPlanInputs.from_descriptor(descriptor)
    plan = TaskCatalog.get(task_id).build_expert_plan(inputs)

    assert isinstance(episode, ResolvedEpisode)
    assert episode.task_id == task_id
    assert episode.identity.startswith("sha256:")
    assert descriptor.actions is not None
    assert inputs.action_config is descriptor.actions
    assert isinstance(plan, ExpertPlan)
    assert plan.task_id == task_id
    assert plan.metadata["episode_identity"] == episode.identity
    assert plan.stages
    assert all(isinstance(stage, (NavigateStage, OperateStage)) for stage in plan.stages)


@pytest.mark.parametrize("task_id", TASK_IDS)
def test_expert_plan_cannot_be_built_from_unresolved_episode(task_id: str) -> None:
    episode = TaskCatalog.resolve(
        EpisodeRequest(task_id=task_id, seed=17, allow_partial_calibration=True)
    )

    with pytest.raises(TypeError, match="ExpertPlanInputs"):
        TaskCatalog.get(task_id).build_expert_plan(episode)  # type: ignore[arg-type]


def test_action_config_specific_metadata_survives_plan_assembly() -> None:
    from bench.execution.contracts import OperateStage

    def build(task_id: str):
        episode = TaskCatalog.resolve(
            EpisodeRequest(task_id=task_id, seed=17, allow_partial_calibration=True)
        )
        descriptor = resolve_builtin_task_plugin(task_id).load_runtime_api().build_descriptor(
            episode, False
        )
        plan = TaskCatalog.get(task_id).build_expert_plan(
            ExpertPlanInputs.from_descriptor(descriptor)
        )
        return descriptor, plan

    wa01_descriptor, wa01_plan = build("WA-01")
    dining_places = [
        stage
        for stage in wa01_plan.stages
        if isinstance(stage, OperateStage) and stage.operation == "place_dynamic"
    ]
    assert dining_places
    assert all(stage.metadata["target_kind"] == "dining" for stage in dining_places)
    assert all(
        stage.metadata["target_kind"]
        == wa01_descriptor.actions.dining_place_metadata_by_mug[stage.target]["target_kind"]
        for stage in dining_places
    )

    wa02_descriptor, wa02_plan = build("WA-02")
    slot_places = [
        stage
        for stage in wa02_plan.stages
        if isinstance(stage, OperateStage) and stage.operation == "place_on_service_slot"
    ]
    assert slot_places
    for stage in slot_places:
        slot = str(stage.metadata["service_slot"])
        assert dict(stage.metadata) == dict(
            wa02_descriptor.actions.service_place_metadata_by_slot[slot]
        )

    for task_id in ("TS01", "TS03"):
        descriptor, plan = build(task_id)
        delivery_stages = [
            stage
            for stage in plan.stages
            if isinstance(stage, OperateStage)
            and stage.operation == "place_sugar_in_live_saucepan"
        ]
        assert delivery_stages
        for index, stage in enumerate(delivery_stages, start=1):
            assert stage.metadata["release_region_order"] == list(
                descriptor.actions.sugar_delivery_metadata_by_index[index][
                    "release_region_order"
                ]
            )


@pytest.mark.parametrize("task_id", TASK_IDS)
def test_recording_and_evaluation_load_the_same_canonical_task_runtime(task_id: str) -> None:
    plugin = resolve_builtin_task_plugin(task_id)
    api = plugin.load_runtime_api()
    episode = TaskCatalog.resolve(
        EpisodeRequest(task_id=task_id, seed=3, allow_partial_calibration=True)
    )
    descriptor = api.build_descriptor(episode, False)

    assert descriptor.episode.identity == episode.identity
    assert descriptor.environment.env_name == plugin.env_name
    assert descriptor.control_frequency_hz == 20
    assert descriptor.sensors.profile_id == "rgbd_v3"
    assert descriptor.sensors.image_size == (224, 224)
    assert callable(plugin.load_expert_plan_factory())


def test_canonical_task_packages_do_not_import_legacy_execution_models() -> None:
    legacy_prefixes = (
        "bench.task_context",
        "bench.model",
        "bench.stages",
        "bench.navigation",
        "bench.manipulation",
        "bench.planning",
        "bench.sampling",
    )
    for task_id in TASK_IDS:
        package = TaskCatalog.get(task_id).package
        module = importlib.import_module(f"{package}.expert.spec")
        source = module.__file__
        assert source is not None
        text = Path(source).read_text(encoding="utf-8")
        assert not any(prefix in text for prefix in legacy_prefixes), source
