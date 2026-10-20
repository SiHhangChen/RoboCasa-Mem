from __future__ import annotations

import json
from pathlib import Path

import pytest

from bench.policy.backends.lerobot import inspect_lerobot_checkpoint
from bench.real_robot.contracts import ACTION_SCHEMA, RUNTIME_PROFILE, STATE_SCHEMA


def _checkpoint(
    path: Path,
    *,
    state_dim: int,
    image_size: int = 224,
    metadata: bool = True,
) -> Path:
    path.mkdir()
    (path / "model.safetensors").write_bytes(b"test")
    (path / "config.json").write_text(
        json.dumps(
            {
                "type": "act",
                "input_features": {
                    "observation.images.agentview_left": {
                        "type": "VISUAL",
                        "shape": [3, image_size, image_size],
                    },
                    "observation.images.agentview_right": {
                        "type": "VISUAL",
                        "shape": [3, image_size, image_size],
                    },
                    "observation.images.eye_in_hand": {
                        "type": "VISUAL",
                        "shape": [3, image_size, image_size],
                    },
                    "observation.state": {"type": "STATE", "shape": [state_dim]},
                },
                "output_features": {
                    "action": {"type": "ACTION", "shape": [13]}
                },
                "chunk_size": 50,
            }
        ),
        encoding="utf-8",
    )
    if metadata:
        (path / "membench_policy.json").write_text(
            json.dumps(
                {"state_schema": "membench.robot_state37.v2"}
            ),
            encoding="utf-8",
        )
    return path


def test_wrong_state_dimension_checkpoint_is_rejected(tmp_path: Path):
    checkpoint = _checkpoint(tmp_path / "wrong_state", state_dim=23)
    with pytest.raises(ValueError, match="state_dim=23"):
        inspect_lerobot_checkpoint(checkpoint)


def test_state37_checkpoint_contract_is_extracted(tmp_path: Path):
    checkpoint = _checkpoint(tmp_path / "current", state_dim=37)

    inspected = inspect_lerobot_checkpoint(checkpoint)

    assert inspected["image_size"] == (224, 224)
    assert inspected["camera_aliases"] == ["agentview_left", "agentview_right", "eye_in_hand"]
    assert inspected["camera_features"] == {
        "agentview_left": "observation.images.agentview_left",
        "agentview_right": "observation.images.agentview_right",
        "eye_in_hand": "observation.images.eye_in_hand",
    }
    assert inspected["horizon"] == 50


def test_state37_checkpoint_requires_schema_metadata(tmp_path: Path):
    checkpoint = _checkpoint(tmp_path / "no_metadata", state_dim=37, metadata=False)
    with pytest.raises(ValueError, match="membench_policy.json"):
        inspect_lerobot_checkpoint(checkpoint)


def _real_checkpoint(path: Path, *, metadata: bool) -> Path:
    path.mkdir()
    (path / "model.safetensors").write_bytes(b"test")
    (path / "config.json").write_text(
        json.dumps(
            {
                "type": "act",
                "input_features": {
                    **{
                        f"observation.images.{camera}": {
                            "type": "VISUAL",
                            "shape": [3, 224, 224],
                        }
                        for camera in ("front", "left", "right")
                    },
                    "observation.state": {"type": "STATE", "shape": [17]},
                },
                "output_features": {"action": {"type": "ACTION", "shape": [17]}},
                "chunk_size": 100,
            }
        ),
        encoding="utf-8",
    )
    if metadata:
        (path / "membench_policy.json").write_text(
            json.dumps(
                {
                    "runtime_profile": RUNTIME_PROFILE,
                    "state_schema": STATE_SCHEMA,
                    "state_dim": 17,
                    "action_schema": ACTION_SCHEMA,
                    "action_dim": 17,
                    "camera_aliases": ["front", "left", "right"],
                    "camera_mapping": {
                        camera: f"observation.images.{camera}"
                        for camera in ("front", "left", "right")
                    },
                    "image_size": [224, 224],
                    "control_frequency_hz": 30.0,
                    "hardware_execution_semantics": "unverified",
                }
            ),
            encoding="utf-8",
        )
    return path


def test_real_robot_checkpoint_contract_is_extracted(tmp_path: Path):
    inspected = inspect_lerobot_checkpoint(_real_checkpoint(tmp_path / "real", metadata=True))
    assert inspected["runtime_profile"] == RUNTIME_PROFILE
    assert inspected["state_dim"] == 17
    assert inspected["action_dim"] == 17
    assert inspected["camera_aliases"] == ["front", "left", "right"]
    assert inspected["camera_features"] == {
        "front": "observation.images.front",
        "left": "observation.images.left",
        "right": "observation.images.right",
    }
    assert inspected["control_frequency_hz"] == 30.0
    assert inspected["hardware_execution_semantics"] == "unverified"


def test_real_robot_checkpoint_requires_profile_sidecar(tmp_path: Path):
    checkpoint = _real_checkpoint(tmp_path / "real", metadata=False)
    with pytest.raises(ValueError, match="runtime_profile"):
        inspect_lerobot_checkpoint(checkpoint)
