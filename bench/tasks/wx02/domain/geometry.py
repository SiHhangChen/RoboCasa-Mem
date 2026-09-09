"""Pure deterministic sampling and geometry for WX-02."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from bench.tasks.wx02.domain.contracts import OBJECT_NAMES, InitialObjectPose


def sample_initial_object_poses(
    *,
    episode_seed: int,
    settings: Mapping[str, Any],
) -> dict[str, InitialObjectPose]:
    """Sample all original object poses with independent deterministic streams."""

    result: dict[str, InitialObjectPose] = {}
    for object_name in OBJECT_NAMES:
        raw = settings.get(object_name)
        if not isinstance(raw, Mapping):
            raise ValueError(
                f"WX-02 initial_state.objects.{object_name} must be a mapping"
            )
        allowed = {"center_xy", "size_xy", "rotation_deg", "seed_offset"}
        unknown = set(raw) - allowed
        if unknown:
            raise ValueError(
                f"unknown WX-02 {object_name} initial fields: "
                + ", ".join(sorted(unknown))
            )
        center = _pair(raw.get("center_xy"), f"{object_name}.center_xy")
        size = _pair(raw.get("size_xy"), f"{object_name}.size_xy", non_negative=True)
        rotation = _pair(
            raw.get("rotation_deg", (0.0, 0.0)), f"{object_name}.rotation_deg"
        )
        seed = int(episode_seed) + int(raw.get("seed_offset", 0))
        rng = np.random.RandomState(seed)
        xy = np.asarray(center) + rng.uniform(
            -0.5 * np.asarray(size), 0.5 * np.asarray(size)
        )
        yaw_min, yaw_max = sorted(rotation)
        yaw = (
            yaw_min
            if np.isclose(yaw_min, yaw_max)
            else float(rng.uniform(yaw_min, yaw_max))
        )
        result[object_name] = InitialObjectPose(
            object_name=object_name,
            xy=(float(xy[0]), float(xy[1])),
            yaw_deg=float(yaw),
            region_center_xy=center,
            region_size_xy=size,
            seed=seed,
        )
    return result


def sample_temporary_targets(
    *,
    episode_seed: int,
    settings: Mapping[str, Any],
) -> tuple[dict[str, tuple[float, float]], tuple[float, float], dict[str, Any]]:
    """Sample independent world-frame cup and staged-tool targets."""

    allowed = {
        "object_region_center_xy",
        "object_region_size_xy",
        "object_seed_offsets",
        "staged_tool_region_center_xy",
        "staged_tool_region_size_xy",
        "staged_tool_seed_offset",
    }
    unknown = set(settings) - allowed
    if unknown:
        raise ValueError(
            "unknown WX-02 temporary target settings: " + ", ".join(sorted(unknown))
        )
    object_center = np.asarray(
        _pair(
            settings.get("object_region_center_xy", (2.78, -3.82)),
            "object_region_center_xy",
        ),
        dtype=float,
    )
    object_size = np.asarray(
        _pair(
            settings.get("object_region_size_xy", (0.10, 0.10)),
            "object_region_size_xy",
            non_negative=True,
        ),
        dtype=float,
    )
    raw_seed_offsets = settings.get(
        "object_seed_offsets", {"primary_cup": 500, "secondary_cup": 600}
    )
    if not isinstance(raw_seed_offsets, Mapping):
        raise ValueError("WX-02 object_seed_offsets must be a mapping")
    cup_names = ("primary_cup", "secondary_cup")
    if set(raw_seed_offsets) != set(cup_names):
        raise ValueError(
            "WX-02 object_seed_offsets must define primary_cup and secondary_cup"
        )
    object_targets: dict[str, tuple[float, float]] = {}
    object_metadata: dict[str, Any] = {}
    for object_name in cup_names:
        seed = int(episode_seed) + int(raw_seed_offsets[object_name])
        rng = np.random.RandomState(seed)
        jitter = rng.uniform(-0.5 * object_size, 0.5 * object_size)
        target = object_center + jitter
        object_targets[object_name] = _tuple2(target)
        object_metadata[object_name] = {
            "seed": seed,
            "region_center_xy": object_center.tolist(),
            "region_size_xy": object_size.tolist(),
            "jitter_xy": jitter.tolist(),
        }

    tool_center = np.asarray(
        _pair(
            settings.get("staged_tool_region_center_xy", (2.74, -4.30)),
            "staged_tool_region_center_xy",
        ),
        dtype=float,
    )
    tool_size = np.asarray(
        _pair(
            settings.get("staged_tool_region_size_xy", (0.06, 0.04)),
            "staged_tool_region_size_xy",
            non_negative=True,
        ),
        dtype=float,
    )
    tool_seed = int(episode_seed) + int(
        settings.get("staged_tool_seed_offset", 1000)
    )
    tool_rng = np.random.RandomState(tool_seed)
    tool_jitter = tool_rng.uniform(-0.5 * tool_size, 0.5 * tool_size)
    staged_tool = tool_center + tool_jitter
    return object_targets, _tuple2(staged_tool), {
        "coordinate_frame": "world",
        "objects": object_metadata,
        "staged_tool": {
            "seed": tool_seed,
            "region_center_xy": tool_center.tolist(),
            "region_size_xy": tool_size.tolist(),
            "jitter_xy": tool_jitter.tolist(),
        },
    }


def sample_wipe_offsets(
    *,
    episode_seed: int,
    settings: Mapping[str, Any],
) -> tuple[tuple[tuple[float, float], ...], dict[str, Any]]:
    """Create one deterministic bidirectional wipe sweep."""

    allowed = {
        "seed_offset",
        "half_length",
        "pass_count",
        "angle_range_deg",
        "angle_offset_deg",
    }
    unknown = set(settings) - allowed
    if unknown:
        raise ValueError(
            "unknown WX-02 wipe sampling settings: " + ", ".join(sorted(unknown))
        )
    seed = int(episode_seed) + int(settings.get("seed_offset", 2000))
    half_length = float(settings.get("half_length", 0.04))
    pass_count = int(settings.get("pass_count", 3))
    angle_range = np.asarray(
        settings.get("angle_range_deg", (0.0, 180.0)), dtype=float
    ).reshape(-1)
    angle_offset_deg = float(settings.get("angle_offset_deg", 0.0))
    if half_length <= 0.0 or pass_count < 1:
        raise ValueError("WX-02 wipe half_length and pass_count must be positive")
    if (
        angle_range.size != 2
        or not np.all(np.isfinite(angle_range))
        or angle_range[0] < 0.0
        or angle_range[1] > 180.0
        or angle_range[0] >= angle_range[1]
    ):
        raise ValueError(
            "WX-02 wipe angle_range_deg must be two increasing values within [0, 180]"
        )
    if not np.isfinite(angle_offset_deg):
        raise ValueError("WX-02 wipe angle_offset_deg must be finite")
    rng = np.random.RandomState(seed)
    sampled_angle_deg = float(rng.uniform(angle_range[0], angle_range[1]))
    angle_rad = float(np.deg2rad(sampled_angle_deg + angle_offset_deg))
    direction = np.asarray([np.cos(angle_rad), np.sin(angle_rad)], dtype=float)
    start_sign = -1.0 if rng.rand() < 0.5 else 1.0
    start = start_sign * half_length * direction
    end = -start
    offsets = tuple(
        _tuple2(start if index % 2 == 0 else end) for index in range(pass_count + 1)
    )
    return offsets, {
        "seed": seed,
        "pass_count": pass_count,
        "angle_rad": angle_rad,
        "angle_deg": float(np.rad2deg(angle_rad) % 180.0),
        "angle_range_deg": angle_range.tolist(),
        "angle_offset_deg": angle_offset_deg,
        "direction_xy": direction.tolist(),
        "start_sign": start_sign,
        "half_length": half_length,
        "sweep_length": 2.0 * half_length,
        "offset_count": len(offsets),
    }


def bottom_center_xy_distance(
    object_bottom_center: Any,
    target_pose: Any,
) -> float:
    """Return the XY distance used by the strict origin predicate."""

    current = np.asarray(object_bottom_center, dtype=float).reshape(-1)
    target = np.asarray(target_pose, dtype=float).reshape(-1)
    if (
        current.size < 2
        or target.size < 2
        or not np.all(np.isfinite(current[:2]))
        or not np.all(np.isfinite(target[:2]))
    ):
        raise ValueError("WX-02 origin distance requires finite XY poses")
    return float(np.linalg.norm(current[:2] - target[:2]))


def _pair(value: Any, label: str, *, non_negative: bool = False) -> tuple[float, float]:
    if isinstance(value, (str, bytes)) or value is None:
        raise ValueError(f"{label} must contain two finite values")
    array = np.asarray(value, dtype=float).reshape(-1)
    if array.size != 2 or not np.all(np.isfinite(array)):
        raise ValueError(f"{label} must contain two finite values")
    pair = (float(array[0]), float(array[1]))
    if non_negative and any(item < 0.0 for item in pair):
        raise ValueError(f"{label} values must be non-negative")
    return pair


def _ordered_pair(value: Any, label: str) -> tuple[float, float]:
    low, high = _pair(value, label)
    if high < low:
        raise ValueError(f"{label} must be ordered")
    return low, high


def _tuple2(value: Any) -> tuple[float, float]:
    array = np.asarray(value, dtype=float).reshape(-1)
    return float(array[0]), float(array[1])


__all__ = [
    "bottom_center_xy_distance",
    "sample_initial_object_poses",
    "sample_temporary_targets",
    "sample_wipe_offsets",
]
