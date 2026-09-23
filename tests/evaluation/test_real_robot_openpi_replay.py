from __future__ import annotations

import argparse
from dataclasses import replace

import numpy as np
import pytest

from bench.policy.backends.fake import FakePolicyBackend
from bench.policy.protocol import InferenceResponse
from bench.real_robot.contracts import RUNTIME_PROFILE
from bench.real_robot.wr07_prompts import FULL_CONFIG, SUBGOAL_CONFIG, SUBGOAL_PROMPTS, prompt_policy
from scripts.eval.real import replay_remote_dataset as replay


class _Dataset:
    def __init__(self) -> None:
        self.samples = [self._sample(0, 0), self._sample(0, 1), self._sample(1, 0)]

    def _sample(self, episode: int, frame: int) -> dict:
        return {
            "episode_index": episode,
            "frame_index": frame,
            "observation.state": np.arange(17, dtype=np.float64),
            "action": np.zeros(17, dtype=np.float32),
            **{f"observation.images.{camera}": np.zeros((3, 224, 224), dtype=np.uint8)
               for camera in ("front", "left", "right")},
        }

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict:
        return self.samples[index]


class _Client:
    def __init__(self, policy: dict) -> None:
        config = FULL_CONFIG if policy["prompt_mode"] == "fixed" else SUBGOAL_CONFIG
        policy = {"family": "openpi", "config": config, **policy}
        self.contract = replace(FakePolicyBackend(runtime_profile=RUNTIME_PROFILE).contract, policy=policy)
        self.resets: list[str] = []
        self.prompts: list[str] = []

    def reset(self, episode_id: str) -> None:
        self.resets.append(episode_id)

    def infer(self, observation, *, episode_id: str, step: int) -> InferenceResponse:
        self.prompts.append(observation.prompt)
        return InferenceResponse("test", episode_id, np.zeros((4, 17), dtype=np.float32))


def _args(**kwargs) -> argparse.Namespace:
    options = dict(
        stride=1, max_frames=3, prompt=None, prompt_source="auto", annotation_root=None,
        expected_config=None,
    )
    options.update(kwargs)
    return argparse.Namespace(**options)


def test_full_model_replay_uses_training_prompt_for_every_episode(monkeypatch) -> None:
    prompt = "Remember the color of the container holding the movable item."
    client = _Client(prompt_policy(FULL_CONFIG, default_prompt=prompt))
    monkeypatch.setattr("builtins.input", lambda _: pytest.fail("fixed prompt must not ask for a color"))

    result = replay.evaluate_dataset(client=client, dataset=_Dataset(), args=_args())

    assert client.prompts == [prompt, prompt, prompt]
    assert len(client.resets) == 2
    assert result["episode_prompts"] == {"0": prompt, "1": prompt}
    assert result["action_mae"] == 0


def test_subgoal_replay_asks_once_per_episode_and_keeps_prompt(monkeypatch) -> None:
    class _TTY:
        def isatty(self) -> bool:
            return True

    answers = iter(("1", "pink"))
    monkeypatch.setattr(replay.sys, "stdin", _TTY())
    monkeypatch.setattr("builtins.input", lambda _: next(answers))
    client = _Client(prompt_policy(SUBGOAL_CONFIG))

    result = replay.evaluate_dataset(client=client, dataset=_Dataset(), args=_args())

    assert client.prompts == [SUBGOAL_PROMPTS["brown"], SUBGOAL_PROMPTS["brown"], SUBGOAL_PROMPTS["pink"]]
    assert result["episode_prompts"] == {"0": SUBGOAL_PROMPTS["brown"], "1": SUBGOAL_PROMPTS["pink"]}


def test_subgoal_replay_requires_selection_without_terminal(monkeypatch) -> None:
    class _NotTTY:
        def isatty(self) -> bool:
            return False

    monkeypatch.setattr(replay.sys, "stdin", _NotTTY())
    with pytest.raises(ValueError, match="needs a terminal"):
        replay.evaluate_dataset(client=_Client(prompt_policy(SUBGOAL_CONFIG)), dataset=_Dataset(), args=_args())


def test_openpi_replay_keeps_camera_aspect_ratio() -> None:
    image = np.full((3, 480, 640), 128, dtype=np.uint8)
    result = replay._image_hwc_uint8(image, target_size=(224, 224), preserve_aspect=True)
    assert result.shape == (224, 224, 3)
    assert np.all(result[0] == 0)
    assert np.all(result[27] == 0)
    assert np.all(result[28:196] == 128)
    assert np.all(result[196] == 0)


def test_replay_rejects_wrong_remote_config() -> None:
    client = _Client(prompt_policy(FULL_CONFIG, default_prompt="full task"))
    with pytest.raises(ValueError, match="remote config must be"):
        replay.evaluate_dataset(
            client=client, dataset=_Dataset(), args=_args(expected_config=SUBGOAL_CONFIG)
        )


def test_baseline_replay_sends_raw_images_and_subgoal_prompts(tmp_path) -> None:
    client = _Client(prompt_policy(SUBGOAL_CONFIG))
    client.contract = replace(
        client.contract,
        policy={**client.contract.policy, "family": "baseline_wr07_omega", "config": FULL_CONFIG},
    )
    annotation = tmp_path / "teacher_raw"
    for episode, prompt in ((0, SUBGOAL_PROMPTS["brown"]), (1, SUBGOAL_PROMPTS["pink"])):
        import json

        path = annotation / f"episode_{episode:06d}" / "annotation.json"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps({"stages": [{"active_subgoal": prompt}]}))
    observations = []
    original_infer = client.infer

    def capture(observation, *, episode_id, step):
        observations.append(observation)
        return original_infer(observation, episode_id=episode_id, step=step)

    client.infer = capture
    replay.evaluate_dataset(
        client=client, dataset=_Dataset(),
        args=_args(prompt_source="annotation", annotation_root=tmp_path),
    )
    assert client.prompts == [SUBGOAL_PROMPTS["brown"], SUBGOAL_PROMPTS["brown"], SUBGOAL_PROMPTS["pink"]]
    assert len(client.resets) == 2
    assert observations[0].extra_modalities["raw_images"]["front"].shape == (224, 224, 3)
