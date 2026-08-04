"""Task-local readers for validated WR01 world-frame calibration records."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from bench.execution.planning import matrix_from_base_obs
from bench.execution.planning.pick_place_targets import pose_pair_from_world


def resolve_anchor_pose_world(env: Any, path: str | Path) -> tuple[np.ndarray, dict[str, Any]]:
    """Resolve an anchor-relative calibrated EEF pose against the live scene."""

    calibration_path = Path(path)
    payload = json.loads(calibration_path.read_text(encoding="utf-8"))
    matrix = np.asarray(payload["pose"]["world"]["matrix"], dtype=float).reshape(4, 4)
    anchor = dict(payload.get("anchor") or {})
    pose = dict(payload.get("pose") or {})
    selector = str(anchor.get("selector", ""))
    reference = np.asarray(anchor.get("world"), dtype=float).reshape(3)
    if selector.startswith("site:"):
        name = selector.removeprefix("site:")
        current = np.asarray(
            env.sim.data.site_xpos[int(env.sim.model.site_name2id(name))],
            dtype=float,
        )
        anchor_kind = "site"
    elif selector.startswith("body:"):
        name = selector.removeprefix("body:")
        current = np.asarray(
            env.sim.data.body_xpos[int(env.sim.model.body_name2id(name))],
            dtype=float,
        )
        anchor_kind = "body"
    else:
        raise ValueError(f"unsupported WR-01 calibration anchor {selector!r}: {path}")
    offset = pose.get("offset_from_anchor_world")
    if offset is None:
        matrix[:3, 3] += current - reference
        resolved_offset = matrix[:3, 3] - current
    else:
        resolved_offset = np.asarray(offset, dtype=float).reshape(3)
        matrix[:3, 3] = current + resolved_offset
    return matrix, {
        "pose_file": str(calibration_path),
        "selector": selector,
        "anchor_kind": anchor_kind,
        "anchor_name": name,
        "reference_anchor_world": reference.tolist(),
        "current_anchor_world": current.tolist(),
        "offset_from_anchor_world": resolved_offset.tolist(),
        "resolved_pose_world": matrix.tolist(),
    }


def target_in_current_base(controller: Any, pose_world: np.ndarray) -> dict[str, Any]:
    """Convert a calibrated world EEF matrix into the shared planner target schema."""

    controller.refresh_obs()
    return pose_pair_from_world(
        matrix_from_base_obs(controller.obs),
        np.asarray(pose_world, dtype=float).reshape(4, 4),
    )["base"]


def matrix_with_pose(position: Any, rotation: Any) -> np.ndarray:
    matrix = np.eye(4, dtype=float)
    matrix[:3, :3] = np.asarray(rotation, dtype=float).reshape(3, 3)
    matrix[:3, 3] = np.asarray(position, dtype=float).reshape(3)
    return matrix


def without_obs(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: without_obs(item) for key, item in value.items() if key != "obs"}
    if isinstance(value, list):
        return [without_obs(item) for item in value]
    if isinstance(value, tuple):
        return [without_obs(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


__all__ = [
    "matrix_with_pose",
    "resolve_anchor_pose_world",
    "target_in_current_base",
    "without_obs",
]
