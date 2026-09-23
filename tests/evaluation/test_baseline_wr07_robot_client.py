from __future__ import annotations

import argparse
from dataclasses import replace

import numpy as np
import pytest

from bench.policy.backends.fake import FakePolicyBackend
from bench.policy.protocol import InferenceResponse
from bench.real_robot.contracts import RUNTIME_PROFILE
from bench.real_robot.wr07_prompts import SUBGOAL_PROMPTS
from scripts.eval.real.run_baseline_wr07_client import run


class _Robot:
    def __init__(self) -> None:
        self.chunks = []
        self.holds = 0
        self.verified = 0

    def observe(self):
        images = {
            alias: np.zeros((480, 640, 3), dtype=np.uint8) for alias in ("front", "left", "right")
        }
        return np.zeros(17, dtype=np.float32), images

    def verify_execution_contract(self, contract):
        self.verified += 1
        return True

    def execute_chunk(self, actions, *, frequency_hz):
        self.chunks.append((actions.copy(), frequency_hz))

    def hold(self):
        self.holds += 1


class _Client:
    def __init__(self) -> None:
        self.contract = replace(
            FakePolicyBackend(runtime_profile=RUNTIME_PROFILE, horizon=50).contract,
            policy={
                "family": "baseline_wr07_omega",
                "config": "pi05_real_robot_wr07_v3",
                "prompt_choices": SUBGOAL_PROMPTS,
                "decision_stride_frames": 10,
            },
        )
        self.resets = []
        self.steps = []

    def reset(self, episode_id):
        self.resets.append(episode_id)

    def infer(self, observation, *, episode_id, step):
        assert observation.extra_modalities["raw_images"]["front"].shape == (480, 640, 3)
        self.steps.append(step)
        return InferenceResponse("test", episode_id, np.zeros((50, 17), dtype=np.float32))


def _args(*, execute=False):
    return argparse.Namespace(
        color="pink", episode_id="wr07-test", max_decisions=2, replan_steps=10, execute=execute
    )


def test_robot_client_shadow_and_verified_execution() -> None:
    client, robot = _Client(), _Robot()
    journal = []
    shadow = run(client=client, robot=robot, args=_args(), on_record=journal.append)
    assert shadow["mode"] == "shadow"
    assert client.steps == [0, 10]
    assert len(journal) == 2
    assert not robot.chunks and not robot.verified

    client, robot = _Client(), _Robot()
    execution = run(client=client, robot=robot, args=_args(execute=True))
    assert execution["mode"] == "execute"
    assert robot.verified == 1
    assert len(robot.chunks) == 2
    assert all(chunk.shape == (10, 17) and hz == 30.0 for chunk, hz in robot.chunks)
    assert robot.holds == 1


def test_robot_client_holds_on_inference_error() -> None:
    client, robot = _Client(), _Robot()

    def fail(*args, **kwargs):
        raise TimeoutError("policy unavailable")

    client.infer = fail
    with pytest.raises(TimeoutError):
        run(client=client, robot=robot, args=_args())
    assert robot.holds == 1


def test_robot_client_requires_explicit_execution_verification() -> None:
    client, robot = _Client(), _Robot()
    robot.verify_execution_contract = lambda contract: False
    with pytest.raises(ValueError, match="did not verify"):
        run(client=client, robot=robot, args=_args(execute=True))
    assert not robot.chunks
    assert robot.holds == 1


def test_robot_client_requires_training_decision_stride() -> None:
    client, robot = _Client(), _Robot()
    args = _args()
    args.replan_steps = 5
    with pytest.raises(ValueError, match="training decision stride"):
        run(client=client, robot=robot, args=args)
