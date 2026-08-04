from __future__ import annotations

import pytest

from bench.tasks import TaskCatalog
from bench.tasks import resolve_builtin_task_plugin
from bench.tasks.contracts import EpisodeRequest


@pytest.mark.contract
@pytest.mark.parametrize("task_id", TaskCatalog.ids())
def test_recording_and_evaluation_share_resolved_episode_identity(task_id: str) -> None:
    seed = 42
    request = EpisodeRequest(
        task_id=task_id,
        seed=seed,
        variant="memory",
        allow_partial_calibration=True,
    )
    recording_episode = TaskCatalog.resolve(request)
    evaluation_episode = TaskCatalog.resolve(request)
    api = resolve_builtin_task_plugin(task_id).load_runtime_api()
    recording_descriptor = api.build_descriptor(recording_episode, False)
    evaluation_descriptor = api.build_descriptor(evaluation_episode, False)

    assert recording_episode.identity == evaluation_episode.identity
    assert recording_descriptor.episode.identity == evaluation_descriptor.episode.identity
    assert recording_descriptor.environment == evaluation_descriptor.environment
    assert recording_descriptor.prompt == evaluation_descriptor.prompt
