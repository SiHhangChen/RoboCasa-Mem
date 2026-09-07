from __future__ import annotations

import pytest

from bench.execution.contracts import NavigateStage, OperateStage
from bench.tasks import TaskCatalog, resolve_builtin_task_plugin
from bench.tasks.contracts import EpisodeRequest, ExpertPlanInputs
from bench.tasks.wa02.config import resolve_task_options
from bench.tasks.wa02.domain.episode import resolve_episode_plan


def _resolve(seed: int):
    episode = TaskCatalog.resolve(EpisodeRequest(task_id="WA-02", seed=seed))
    descriptor = resolve_builtin_task_plugin("WA-02").load_runtime_api().build_descriptor(
        episode, False
    )
    return episode, descriptor


def test_wa02_default_execution_sequence_is_balanced_independently_from_prompt() -> None:
    facts = [dict(_resolve(seed)[0].facts) for seed in range(16)]

    combinations = {
        (
            item["hot_cup_id"],
            item["requested_temperature"],
            tuple(sorted(dict(item["service_slot_by_cup"]).items())),
            tuple(item["fill_temperature_order"]),
        )
        for item in facts
    }
    assert len(combinations) == 16

    for requested_temperature in ("hot", "cold"):
        requested_ordinals = [
            tuple(item["fill_temperature_order"]).index(requested_temperature)
            for item in facts
            if item["requested_temperature"] == requested_temperature
        ]
        assert requested_ordinals.count(0) == 4
        assert requested_ordinals.count(1) == 4


@pytest.mark.parametrize("value", ("hot-first", "cold-first", "balanced"))
def test_wa02_rejects_fill_order_overrides(value: str) -> None:
    with pytest.raises(ValueError, match="unknown"):
        resolve_task_options({"fill_order": value})
    with pytest.raises(ValueError, match="no longer supports"):
        resolve_episode_plan(seed=0, fill_order=value)


def test_wa02_same_prompt_can_target_first_or_second_staged_cup() -> None:
    target_was_staged_last = set()

    for seed in (0, 8):
        episode, descriptor = _resolve(seed)
        plan = TaskCatalog.get("WA-02").build_expert_plan(
            ExpertPlanInputs.from_descriptor(descriptor)
        )
        placements = [
            stage.target
            for stage in plan.stages
            if isinstance(stage, OperateStage)
            and stage.operation == "place_on_service_slot"
        ]
        final_navigation = next(
            stage
            for stage in plan.stages
            if isinstance(stage, NavigateStage)
            and stage.name.startswith("navigate_final_pick_")
        )

        assert episode.facts["requested_temperature"] == "hot"
        assert final_navigation.name.endswith(str(episode.facts["requested_cup_id"]))
        target_was_staged_last.add(
            placements[-1] == str(episode.facts["requested_cup_id"])
        )

    assert target_was_staged_last == {False, True}
