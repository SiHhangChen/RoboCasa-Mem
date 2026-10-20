from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from bench.runtime.setup_replay import apply_transcript, config_identity


class Environment:
    def __init__(self):
        self.sim = SimpleNamespace(
            data=SimpleNamespace(qpos=np.zeros(2), qvel=np.zeros(2), time=0.0)
        )
        self.observations = []
        self.steps = []

    def step(self, action):
        self.steps.append(action.copy())
        self.sim.data.qpos[0] += action[0]
        self.sim.data.time += 0.05
        return {"state": self.sim.data.qpos.copy()}, 0.0, False, {}

    def _get_observations(self, *, force_update):
        self.observations.append(force_update)
        return {"state": self.sim.data.qpos.copy()}


def transcript():
    action = np.zeros(13)
    action[0] = 0.1
    return {
        "events": [
            {"kind": "observe", "force_update": True},
            {"kind": "step", "action": action.tolist()},
            {"kind": "observe", "force_update": False},
        ],
        "expected": {"qpos": [0.1, 0.0], "qvel": [0.0, 0.0], "time": 0.05},
    }


def test_replay_preserves_step_and_observation_order():
    env = Environment()
    observation = apply_transcript(env, transcript())
    np.testing.assert_array_equal(observation["state"], [0.1, 0.0])
    assert env.observations == [True, False]
    assert len(env.steps) == 1


@pytest.mark.parametrize("field,value", [("qpos", [0.2, 0.0]), ("qvel", [0.1, 0.0]), ("time", 0.1)])
def test_changed_initial_physics_fails_explicitly(field, value):
    payload = transcript()
    payload["expected"][field] = value
    with pytest.raises(RuntimeError, match="differs"):
        apply_transcript(Environment(), payload)


@pytest.mark.parametrize("action", [[0.0] * 12, [float("nan")] * 13])
def test_invalid_setup_action_is_rejected(action):
    payload = transcript()
    payload["events"][1]["action"] = action
    with pytest.raises(ValueError, match="invalid"):
        apply_transcript(Environment(), payload)


def test_unknown_and_empty_transcripts_are_rejected():
    with pytest.raises(ValueError, match="unknown"):
        apply_transcript(Environment(), {"events": [{"kind": "hidden_mutation"}]})
    with pytest.raises(ValueError, match="empty"):
        apply_transcript(Environment(), {"events": []})


def test_configuration_identity_excludes_only_removed_planner(tmp_path):
    a = tmp_path / "a.yaml"
    b = tmp_path / "b.yaml"
    a.write_text(
        "resources:\n  planner_script: original.py\n  stations: station.json\nepisode:\n  seed: 1050\n"
    )
    b.write_text("resources:\n  stations: station.json\nepisode:\n  seed: 1050\n")
    assert config_identity(a) == config_identity(b)
    b.write_text(b.read_text().replace("1050", "1051"))
    assert config_identity(a) != config_identity(b)


def test_all_shipped_seed_transcripts_are_verified():
    root = Path(__file__).resolve().parents[2] / "configs/benchmark/setup_replays/wx01"
    artifacts = list(root.glob("[0-9]*.json"))
    # Development runs may generate these incrementally. The exported
    # candidate test requires the full formal seed range.
    if (root.parents[3] / "release_candidate.json").is_file():
        assert {int(p.stem) for p in artifacts} == set(range(1050, 1100))
    for path in artifacts:
        payload = json.loads(path.read_text())
        assert payload["verified"] is True
        assert payload["seed"] == int(path.stem)
        assert payload["task_version"] == "1.0.0"
        assert payload["resource_sha256"]
        assert payload["validation"]["replayed_model_actions"] == 20
        assert payload["validation"]["trajectory_observations_equal"]


def test_result_metadata_identifies_transcript_and_rejects_changed_resources(tmp_path, monkeypatch):
    import hashlib

    import bench.runtime.setup_replay as reader
    from bench.tasks.wx01.execution import geometry

    config = tmp_path / "task.yaml"
    config.write_text("resources:\n  stations: station.json\n")
    station = tmp_path / "station.json"
    station.write_text("{}")
    replay = tmp_path / "configs/benchmark/setup_replays/wx01/1050.json"
    replay.parent.mkdir(parents=True)
    payload = {
        "schema_version": reader.SCHEMA,
        "seed": 1050,
        "verified": True,
        "task_version": "1.0.0",
        "config_identity": reader.config_identity(config),
        "resource_sha256": {"stations": hashlib.sha256(station.read_bytes()).hexdigest()},
        "initialization": {"short_start": {"mode": "pregrasped"}},
    }
    replay.write_text(json.dumps(payload))
    monkeypatch.setattr(reader, "__file__", str(tmp_path / "bench/runtime/setup_replay.py"))
    monkeypatch.setattr(reader, "prepare", lambda *args: None)
    monkeypatch.setattr(reader, "apply_transcript", lambda *args: {"ready": True})
    monkeypatch.setattr(geometry, "resolve_episode_targets", lambda *args: None)
    descriptor = SimpleNamespace(
        episode=SimpleNamespace(seed=1050, task_version="1.0.0", task_config_file=config),
        resources={"stations": station},
        episode_plan=None,
    )
    observation, summary = reader.replay_setup(None, descriptor)
    assert observation == {"ready": True}
    assert summary["setup_replay"]["file_sha256"] == hashlib.sha256(replay.read_bytes()).hexdigest()
    station.write_text('{"modified":true}')
    with pytest.raises(ValueError, match="resource"):
        reader.replay_setup(None, descriptor)
