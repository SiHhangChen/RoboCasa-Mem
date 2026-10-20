"""Replay a verified WX-01 reset transcript without loading a planner.

Setup actions run before policy control and are excluded from suite budgets.
Missing artifacts or mismatched input/state identities fail explicitly.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import yaml

SCHEMA = "membench.wx01_setup_replay/v1"


def config_identity(config_path):
    payload = yaml.safe_load(Path(config_path).read_text())
    payload["resources"].pop("planner_script", None)
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def observation_digests(observation):
    return {
        name: hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()
        for name, value in observation.items()
        if isinstance(value, np.ndarray)
    }


def prepare(env, targets):
    from bench.tasks.wx01.initialization import (
        object_joint_quaternion_wxyz,
        set_object_free_joint_pose,
    )

    quaternion = object_joint_quaternion_wxyz(env, "blocking_object")
    if quaternion is None or not set_object_free_joint_pose(
        env,
        "blocking_object",
        position=np.asarray(targets.temporary_blocking_pose),
        quaternion_wxyz=quaternion,
    ):
        raise RuntimeError("WX-01 setup replay could not preplace blocking object")
    env.set_original_object_poses(
        cleaning_tool=targets.original_tool_pose, blocking_object=targets.original_blocking_pose
    )
    env.sim.forward()


def apply_transcript(env, payload):
    observation = None
    for event in payload["events"]:
        if event["kind"] == "step":
            action = np.asarray(event["action"], dtype=float)
            if action.shape != (13,) or not np.all(np.isfinite(action)):
                raise ValueError("invalid WX-01 setup action")
            observation, _, _, _ = env.step(action)
        elif event["kind"] == "observe":
            observation = env._get_observations(force_update=event["force_update"])
        else:
            raise ValueError(f"unknown setup event {event['kind']!r}")
    if observation is None:
        raise ValueError("empty WX-01 setup transcript")
    expected = payload["expected"]
    for field in ("qpos", "qvel"):
        actual = np.asarray(getattr(env.sim.data, field))
        reference = np.asarray(expected[field])
        if actual.shape != reference.shape or not np.allclose(actual, reference, rtol=0, atol=1e-9):
            raise RuntimeError(f"WX-01 setup replay {field} differs from verified source")
    if abs(float(env.sim.data.time) - expected["time"]) > 1e-9:
        raise RuntimeError("WX-01 setup replay simulation time differs from source")
    # Pixel equality is verified when freezing. Driver-dependent rendering is
    # not used as a run-time rejection criterion; simulator state is checked.
    return observation


def replay_setup(env, descriptor):
    seed = descriptor.episode.seed
    root = Path(__file__).resolve().parents[2]
    replay_path = root / "configs/benchmark/setup_replays/wx01" / f"{seed}.json"
    if not replay_path.is_file():
        raise FileNotFoundError(
            f"WX-01 requires a verified setup replay for seed {seed}: {replay_path}"
        )
    payload = json.loads(replay_path.read_text())
    if payload.get("schema_version") != SCHEMA or payload.get("seed") != seed:
        raise ValueError("WX-01 setup replay schema/seed mismatch")
    if not payload.get("verified"):
        raise ValueError("WX-01 setup replay has not passed equivalence validation")
    if payload["config_identity"] != config_identity(descriptor.episode.task_config_file):
        raise ValueError("WX-01 setup replay task configuration differs from source")
    if payload.get("task_version") != descriptor.episode.task_version:
        raise ValueError("WX-01 setup replay task version differs from source")
    for name, expected in payload.get("resource_sha256", {}).items():
        resource_path = descriptor.resources.get(name)
        if (
            resource_path is None
            or hashlib.sha256(Path(resource_path).read_bytes()).hexdigest() != expected
        ):
            raise ValueError(f"WX-01 setup replay resource {name!r} differs from source")
    from bench.tasks.wx01.execution.geometry import resolve_episode_targets

    targets = resolve_episode_targets(env, descriptor.episode_plan)
    prepare(env, targets)
    observation = apply_transcript(env, payload)
    return observation, {
        **payload["initialization"]["short_start"],
        "setup_replay": {
            "schema_version": SCHEMA,
            "seed": seed,
            "file_sha256": hashlib.sha256(replay_path.read_bytes()).hexdigest(),
        },
    }
