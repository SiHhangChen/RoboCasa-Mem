from __future__ import annotations

from collections import Counter

from bench.execution.contracts import OperateStage
from bench.tasks import TaskCatalog, resolve_builtin_task_plugin
from bench.tasks.contracts import EpisodeRequest, ExpertPlanInputs
from bench.tasks.wr03.domain.contracts import TASK_REVISION
from bench.tasks.wr03.domain.episode import resolve_episode_plan


def _resolve(seed: int):
    episode = TaskCatalog.resolve(EpisodeRequest(task_id="WR-03", seed=seed))
    descriptor = resolve_builtin_task_plugin("WR-03").load_runtime_api().build_descriptor(
        episode, False
    )
    plan = TaskCatalog.get("WR-03").build_expert_plan(
        ExpertPlanInputs.from_descriptor(descriptor)
    )
    return episode, descriptor, plan


def test_wr03_storage_level_and_close_order_are_independently_balanced() -> None:
    episodes = [resolve_episode_plan(seed=seed) for seed in range(200)]
    combinations = Counter(
        (episode.stored_level, episode.close_order) for episode in episodes
    )

    assert combinations == {
        ("lower", ("lower", "upper")): 50,
        ("lower", ("upper", "lower")): 50,
        ("upper", ("lower", "upper")): 50,
        ("upper", ("upper", "lower")): 50,
    }


def test_wr03_expert_uses_reset_storage_and_closes_in_resolved_order() -> None:
    episode, descriptor, plan = _resolve(17)
    names = [stage.name for stage in plan.stages]
    operations = [stage for stage in plan.stages if isinstance(stage, OperateStage)]
    close_stages = [stage for stage in operations if stage.operation == "close_storage_door"]

    assert descriptor.episode_plan.stored_level == episode.facts["stored_level"]
    assert descriptor.config.recording["record_depth"] is False
    assert [stage.target for stage in close_stages] == list(episode.facts["close_order"])
    assert "pick_can_for_storage" not in names
    assert "place_can_in_lower_cabinet" not in names
    assert "place_can_in_upper_cabinet" not in names
    assert names.index("canonical_memory_reset") < names.index("open_remembered_cabinet")
    assert names.index("open_remembered_cabinet") < names.index("retrieve_can")
    assert names.index("retrieve_can") < names.index("place_can_on_table")
    assert plan.metadata["stored_level"] == episode.facts["stored_level"]
    assert plan.metadata["close_order"] == list(episode.facts["close_order"])
    assert plan.metadata["task_revision"] == TASK_REVISION == "WR03-v4"
