"""TM02 reset-time mobile-base initialization."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from bench.execution.navigation.base_motion import base_yaw, set_base_pose_kinematic
from bench.runtime.contracts import InitializationReport


def apply_manual_path_initial_base_pose(
    env: Any,
    obs: dict[str, Any],
    *,
    path_file: str | Path,
    sample_index: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Place the robot at one user-measured TM02 path sample after reset."""

    path = Path(path_file)
    payload = json.loads(path.read_text(encoding="utf-8"))
    samples = {
        int(item.get("index", fallback)): dict(item)
        for fallback, item in enumerate(payload.get("samples", ()))
    }
    index = int(sample_index)
    if index not in samples:
        raise KeyError(f"TM02 initial manual-path sample {index} is unavailable")
    sample = samples[index]
    xy = np.asarray(sample.get("xy", ()), dtype=float).reshape(-1)
    yaw = float(sample.get("yaw"))
    if xy.size != 2 or not np.all(np.isfinite(xy)) or not np.isfinite(yaw):
        raise ValueError(f"TM02 initial manual-path sample {index} is invalid")

    before_xy = np.asarray(obs["robot0_base_pos"], dtype=float).reshape(-1)[:2]
    before_yaw = float(base_yaw(obs))
    updated = set_base_pose_kinematic(env, target_xy=xy, target_yaw=yaw)
    after_xy = np.asarray(updated["robot0_base_pos"], dtype=float).reshape(-1)[:2]
    after_yaw = float(base_yaw(updated))
    return updated, {
        "mode": "manual_path_sample",
        "recorded_in_hdf5_actions": False,
        "manual_base_path_file": str(path),
        "sample_index": index,
        "sample_name": sample.get("name"),
        "before_xy": before_xy.astype(float).tolist(),
        "before_yaw": before_yaw,
        "target_xy": xy.astype(float).tolist(),
        "target_yaw": yaw,
        "after_xy": after_xy.astype(float).tolist(),
        "after_yaw": after_yaw,
    }


__all__ = ["apply_manual_path_initial_base_pose"]


def initialize_episode(runtime: Any, episode: Any) -> InitializationReport:
    """Expose reset facts already applied by the task runtime."""
    del episode
    return InitializationReport.applied(facts=dict(getattr(runtime, "initialization", {}) or {}))
