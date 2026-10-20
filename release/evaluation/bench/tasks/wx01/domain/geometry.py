"""Pure deterministic sampling and geometry for WX-01."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from bench.tasks.wx01.domain.contracts import InitialObjectPose


def sample_initial_object_poses(
    *,
    episode_seed: int,
    settings: Mapping[str, Any],
) -> dict[str, InitialObjectPose]:
    """Sample both original object poses with independent deterministic streams."""

    result: dict[str, InitialObjectPose] = {}
    for object_name in ("blocking_object", "cleaning_tool"):
        raw = settings.get(object_name)
        if not isinstance(raw, Mapping):
            raise ValueError(
                f"WX-01 initial_state.objects.{object_name} must be a mapping"
            )
        allowed = {"center_xy", "size_xy", "rotation_deg", "seed_offset"}
        unknown = set(raw) - allowed
        if unknown:
            raise ValueError(
                f"unknown WX-01 {object_name} initial fields: "
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


def sample_temporary_offsets(
    *,
    episode_seed: int,
    settings: Mapping[str, Any],
) -> tuple[tuple[float, float], tuple[float, float], dict[str, Any]]:
    """Sample the displaced cup and post-wipe sponge offsets jointly."""

    allowed = {
        "blocking_center_offset_xy",
        "blocking_jitter_x",
        "blocking_jitter_y",
        "tool_center_offset_xy",
        "tool_jitter_x",
        "tool_jitter_y",
        "minimum_tool_to_blocking_distance",
        "minimum_tool_to_origin_distance",
        "maximum_sampling_attempts",
    }
    unknown = set(settings) - allowed
    if unknown:
        raise ValueError(
            "unknown WX-01 temporary target settings: " + ", ".join(sorted(unknown))
        )
    blocking_center = np.asarray(
        _pair(
            settings.get("blocking_center_offset_xy", (0.0, 0.18)),
            "blocking_center_offset_xy",
        )
    )
    blocking_jitter_x = _ordered_pair(
        settings.get("blocking_jitter_x", (-0.06, 0.06)), "blocking_jitter_x"
    )
    blocking_jitter_y = _ordered_pair(
        settings.get("blocking_jitter_y", (-0.015, 0.015)), "blocking_jitter_y"
    )
    blocking_rng = np.random.RandomState(int(episode_seed))
    blocking_jitter = np.asarray(
        [
            blocking_rng.uniform(*blocking_jitter_x),
            blocking_rng.uniform(*blocking_jitter_y),
        ],
        dtype=float,
    )
    blocking = blocking_center + blocking_jitter

    tool_center = np.asarray(
        _pair(
            settings.get("tool_center_offset_xy", (0.0, -0.38)), "tool_center_offset_xy"
        )
    )
    tool_jitter_x = _ordered_pair(
        settings.get("tool_jitter_x", (-0.08, 0.08)), "tool_jitter_x"
    )
    tool_jitter_y = _ordered_pair(
        settings.get("tool_jitter_y", (-0.06, 0.04)), "tool_jitter_y"
    )
    minimum_pair_distance = float(
        settings.get("minimum_tool_to_blocking_distance", 0.32)
    )
    minimum_origin_distance = float(
        settings.get("minimum_tool_to_origin_distance", 0.30)
    )
    maximum_attempts = int(settings.get("maximum_sampling_attempts", 50))
    if (
        minimum_pair_distance < 0.0
        or minimum_origin_distance < 0.0
        or maximum_attempts < 1
    ):
        raise ValueError("WX-01 temporary target constraints must be non-negative")
    tool_rng = np.random.RandomState(int(episode_seed) + 1000)
    for attempt in range(1, maximum_attempts + 1):
        tool_jitter = np.asarray(
            [tool_rng.uniform(*tool_jitter_x), tool_rng.uniform(*tool_jitter_y)],
            dtype=float,
        )
        tool = tool_center + tool_jitter
        pair_distance = float(np.linalg.norm(tool - blocking))
        origin_distance = float(np.linalg.norm(tool))
        if (
            pair_distance < minimum_pair_distance
            or origin_distance < minimum_origin_distance
        ):
            continue
        metadata = {
            "seed": int(episode_seed),
            "blocking": {
                "center_offset_xy": blocking_center.tolist(),
                "jitter_xy": blocking_jitter.tolist(),
            },
            "tool": {
                "center_offset_xy": tool_center.tolist(),
                "jitter_xy": tool_jitter.tolist(),
                "sampling_attempt": attempt,
            },
            "tool_to_blocking_distance": pair_distance,
            "tool_to_origin_distance": origin_distance,
            "minimum_tool_to_blocking_distance": minimum_pair_distance,
            "minimum_tool_to_origin_distance": minimum_origin_distance,
        }
        return _tuple2(blocking), _tuple2(tool), metadata
    raise RuntimeError("could not sample valid WX-01 temporary targets")


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
            "unknown WX-01 wipe sampling settings: " + ", ".join(sorted(unknown))
        )
    seed = int(episode_seed) + int(settings.get("seed_offset", 2000))
    half_length = float(settings.get("half_length", 0.04))
    pass_count = int(settings.get("pass_count", 3))
    angle_range = np.asarray(
        settings.get("angle_range_deg", (0.0, 180.0)), dtype=float
    ).reshape(-1)
    angle_offset_deg = float(settings.get("angle_offset_deg", 0.0))
    if half_length <= 0.0 or pass_count < 1:
        raise ValueError("WX-01 wipe half_length and pass_count must be positive")
    if (
        angle_range.size != 2
        or not np.all(np.isfinite(angle_range))
        or angle_range[0] < 0.0
        or angle_range[1] > 180.0
        or angle_range[0] >= angle_range[1]
    ):
        raise ValueError(
            "WX-01 wipe angle_range_deg must be two increasing values within [0, 180]"
        )
    if not np.isfinite(angle_offset_deg):
        raise ValueError("WX-01 wipe angle_offset_deg must be finite")
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
        raise ValueError("WX-01 origin distance requires finite XY poses")
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
    "sample_temporary_offsets",
    "sample_wipe_offsets",
]
