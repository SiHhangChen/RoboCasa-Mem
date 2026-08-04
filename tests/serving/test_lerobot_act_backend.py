from __future__ import annotations

import json
from pathlib import Path

import pytest

from bench.policy.backends.lerobot import inspect_lerobot_checkpoint


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
