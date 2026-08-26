from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from bench.evaluation.replay import (
    ReplayPolicyClient,
    discover_recorded_episodes,
    parse_seed_spec,
    replay_policy_contract,
)


def _episode(root: Path, branch: str, seed: int, attempt: int = 0) -> Path:
    path = (
        root
        / branch
        / "WR-06"
        / f"seed_{seed:06d}_attempt_{attempt:02d}"
    )
    path.mkdir(parents=True)
    (path / "trajectory.h5").touch()
    return path


def test_parse_seed_spec_accepts_ranges_and_descending_ranges() -> None:
    assert parse_seed_spec("0-2,5;7-6") == {0, 1, 2, 5, 6, 7}


def test_recording_discovery_includes_failed_tree_with_partial_manifest(
    tmp_path: Path,
) -> None:
    accepted = _episode(tmp_path, "train", seed=0)
    failed = _episode(tmp_path, "debug_failed", seed=1)
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "items": [
                    {
                        "task_id": "WR-06",
                        "seed": 0,
                        "attempt": 0,
                        "success": True,
                        "episode_identity": "recorded-zero",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    episodes = discover_recorded_episodes(
        tmp_path,
        "WR-06",
        set(),
        source_status="all",
    )

    assert [(item["seed"], item["source_status"]) for item in episodes] == [
        (0, "success"),
        (1, "failed"),
    ]
    assert episodes[0]["episode_dir"] == accepted
    assert episodes[0]["episode_identity"] == "recorded-zero"
    assert episodes[1]["episode_dir"] == failed


def test_recording_discovery_status_filter_selects_failed_copy(
    tmp_path: Path,
) -> None:
    _episode(tmp_path, "train", seed=3)
    failed = _episode(tmp_path, "debug_failed", seed=3)

    episodes = discover_recorded_episodes(
        tmp_path,
        "WR-06",
        {3},
        source_status="failed",
    )

    assert len(episodes) == 1
    assert episodes[0]["episode_dir"] == failed
    assert episodes[0]["source_success"] is False


def test_recording_discovery_uses_manifest_episode_directory(
    tmp_path: Path,
) -> None:
    episode = tmp_path / "custom" / "recording"
    episode.mkdir(parents=True)
    (episode / "trajectory.h5").touch()
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "items": [
                    {
                        "task_id": "WR-06",
                        "seed": 8,
                        "attempt": 2,
                        "success": True,
                        "episode_dir": str(episode),
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    episodes = discover_recorded_episodes(
        tmp_path,
        "WR-06",
        set(),
        source_status="all",
    )

    assert len(episodes) == 1
    assert episodes[0]["seed"] == 8
    assert episodes[0]["episode_dir"] == episode


def test_replay_policy_returns_recorded_actions_without_skips() -> None:
    actions = np.arange(65, dtype=np.float32).reshape(5, 13)
    client = ReplayPolicyClient(
        actions,
        contract=replay_policy_contract(horizon=2),
        chunk_size=2,
    )
    client.reset("episode")

    chunks = [
        client.infer(None, episode_id="episode", step=0).actions,
        client.infer(None, episode_id="episode", step=2).actions,
        client.infer(None, episode_id="episode", step=4).actions,
    ]

    np.testing.assert_array_equal(np.concatenate(chunks), actions)
