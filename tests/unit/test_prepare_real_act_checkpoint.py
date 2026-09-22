from __future__ import annotations

import json
from pathlib import Path

import pytest

from bench.real_robot.contracts import (
    ACTION_FIELDS,
    CAMERA_ALIASES,
    RUNTIME_PROFILE,
    STATE_FIELDS,
)
from scripts.eval.real.prepare_act_checkpoint import build_metadata, write_metadata


def _dataset(path: Path) -> Path:
    meta = path / "meta"
    meta.mkdir(parents=True)
    features = {
        "observation.state": {
            "dtype": "float64",
            "shape": [17],
            "names": [list(STATE_FIELDS)],
        },
        "action": {
            "dtype": "float64",
            "shape": [17],
            "names": [list(ACTION_FIELDS)],
        },
        **{
            f"observation.images.{camera}": {
                "dtype": "video",
                "shape": [3, 480, 640],
            }
            for camera in CAMERA_ALIASES
        },
    }
    (meta / "info.json").write_text(
        json.dumps({"fps": 30, "features": features}), encoding="utf-8"
    )
    return path


def _checkpoint(path: Path) -> Path:
    path.mkdir()
    (path / "config.json").write_text(
        json.dumps(
            {
                "type": "act",
                "input_features": {
                    "observation.state": {"shape": [17]},
                    **{
                        f"observation.images.{camera}": {"shape": [3, 224, 224]}
                        for camera in CAMERA_ALIASES
                    },
                },
                "output_features": {"action": {"shape": [17]}},
                "chunk_size": 100,
            }
        ),
        encoding="utf-8",
    )
    return path


def test_build_and_atomically_write_real_act_metadata(tmp_path: Path) -> None:
    payload = build_metadata(_checkpoint(tmp_path / "checkpoint"), _dataset(tmp_path / "data"))
    assert payload["runtime_profile"] == RUNTIME_PROFILE
    assert payload["hardware_execution_semantics"] == "unverified"
    destination = tmp_path / "checkpoint" / "membench_policy.json"
    write_metadata(destination, payload)
    assert json.loads(destination.read_text(encoding="utf-8")) == payload


def test_dataset_field_order_mismatch_is_rejected(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path / "data")
    info_path = dataset / "meta" / "info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    info["features"]["action"]["names"][0].reverse()
    info_path.write_text(json.dumps(info), encoding="utf-8")
    with pytest.raises(ValueError, match="names"):
        build_metadata(_checkpoint(tmp_path / "checkpoint"), dataset)
