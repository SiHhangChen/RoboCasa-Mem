from __future__ import annotations

import pytest

from bench.tasks import TaskCatalog
from bench.tasks.contracts import EpisodeRequest
from bench.tasks.wa02.config import resolve_task_options
from bench.tasks.wa02.domain.episode import resolve_episode_plan


def _facts(seed: int) -> dict[str, object]:
    episode = TaskCatalog.resolve(EpisodeRequest(task_id="WA-02", seed=seed))
    return dict(episode.facts)


def test_wa02_default_execution_order_is_balanced_over_sixteen_seeds() -> None:
    facts = [_facts(seed) for seed in range(16)]

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


@pytest.mark.parametrize("value", ("hot-first", "cold-first", "balanced"))
def test_wa02_rejects_fill_order_overrides(value: str) -> None:
    with pytest.raises(ValueError, match="unknown"):
        resolve_task_options({"fill_order": value})
    with pytest.raises(ValueError, match="no longer supports"):
        resolve_episode_plan(seed=0, fill_order=value)
