from __future__ import annotations

from bench.tasks import TaskCatalog
from bench.tasks.contracts import EpisodeRequest
from bench.tasks.tm02.config import load_task_config
from bench.tasks.tm02.domain.episode import resolve_episode_plan, sample_episode_plan


VERIFIED_ACTION_REPLAY_SUCCESS_STEPS = (3290, 3206)


def test_tm02_horizon_covers_verified_action_replays() -> None:
    config = load_task_config()
    horizon = int(config.episode["max_horizon"])

    assert horizon == 3600
    assert horizon > max(VERIFIED_ACTION_REPLAY_SUCCESS_STEPS)


def test_tm02_horizon_is_consistent_across_resolution_boundaries() -> None:
    expected = 3600

    assert sample_episode_plan(0).max_horizon == expected
    assert resolve_episode_plan(seed=1).max_horizon == expected
    assert TaskCatalog.get("TM02").default_horizon == expected
    assert TaskCatalog.resolve(
        EpisodeRequest(task_id="TM02", seed=0)
    ).max_horizon == expected
