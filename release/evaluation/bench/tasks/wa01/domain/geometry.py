"""WA-01 餐桌区域几何与确定性目标采样纯函数。"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np


DINING_REGION_SLOTS = ("left", "right")
DINING_REGION_TARGET_SEED_SALT = 0x44494E45
DINING_TARGET_MAX_SAMPLING_ATTEMPTS = 1024


@dataclass(frozen=True)
class DiningPlacementRegionConfig:
    """定义世界坐标系下左右餐桌区域的几何和采样约束。"""

    center_x: float = 1.566
    centerline_y: float = -5.259
    region_size_xy: tuple[float, float] = (0.10, 0.10)
    lateral_center_offset: float = 0.16
    sampling_inset: float = 0.025
    target_separation_range: tuple[float, float] = (0.24, 0.32)
    coordinate_frame: str = "world"


def resolve_dining_region_config(
    value: Mapping[str, Any] | None,
) -> DiningPlacementRegionConfig:
    """校验字典配置并生成不可变的餐桌区域配置。"""

    payload = dict(value or {})
    frame = str(payload.get("coordinate_frame", "world")).strip().lower()
    if frame != "world":
        raise ValueError("WA-01 dining regions require world coordinates")
    raw_size = np.asarray(payload.get("region_size_xy", (0.10, 0.10)), dtype=float).reshape(-1)
    raw_separation = np.asarray(
        payload.get("target_separation_range", (0.24, 0.32)), dtype=float
    ).reshape(-1)
    if raw_size.size != 2 or not np.all(np.isfinite(raw_size)):
        raise ValueError("region_size_xy must contain two finite values")
    if raw_separation.size != 2 or not np.all(np.isfinite(raw_separation)):
        raise ValueError("target_separation_range must contain two finite values")
    size_x, size_y = float(raw_size[0]), float(raw_size[1])
    minimum, maximum = float(raw_separation[0]), float(raw_separation[1])
    center_x = float(payload.get("center_x", 1.566))
    centerline_y = float(payload.get("centerline_y", -5.259))
    lateral_offset = float(payload.get("lateral_center_offset", 0.16))
    inset = float(payload.get("sampling_inset", 0.025))
    if size_x <= 0.0 or size_y <= 0.0:
        raise ValueError("dining-region dimensions must be positive")
    if lateral_offset <= 0.0:
        raise ValueError("lateral_center_offset must be positive")
    if inset < 0.0 or 2.0 * inset >= min(size_x, size_y):
        raise ValueError("sampling_inset must leave a non-empty sample area")
    if minimum < 0.0 or maximum <= minimum:
        raise ValueError("target_separation_range must be ordered and non-negative")
    if not np.all(np.isfinite([center_x, centerline_y, lateral_offset, inset])):
        raise ValueError("dining-region geometry must be finite")
    return DiningPlacementRegionConfig(
        center_x=center_x,
        centerline_y=centerline_y,
        region_size_xy=(size_x, size_y),
        lateral_center_offset=lateral_offset,
        sampling_inset=inset,
        target_separation_range=(minimum, maximum),
        coordinate_frame=frame,
    )


def dining_region_config_summary(config: DiningPlacementRegionConfig) -> dict[str, Any]:
    """将区域配置转换为可写入 episode 元数据的普通字典。"""

    return {
        "coordinate_frame": config.coordinate_frame,
        "center_x": float(config.center_x),
        "centerline_y": float(config.centerline_y),
        "region_size_xy": [float(value) for value in config.region_size_xy],
        "lateral_center_offset": float(config.lateral_center_offset),
        "sampling_inset": float(config.sampling_inset),
        "target_separation_range": [
            float(value) for value in config.target_separation_range
        ],
    }


def compute_dining_regions(
    config: DiningPlacementRegionConfig,
) -> dict[str, dict[str, Any]]:
    """计算左右区域各自的中心、尺寸和闭区间边界。"""

    half_x = 0.5 * config.region_size_xy[0]
    half_y = 0.5 * config.region_size_xy[1]
    regions: dict[str, dict[str, Any]] = {}
    for slot, sign in (("left", 1.0), ("right", -1.0)):
        center_y = config.centerline_y + sign * config.lateral_center_offset
        regions[slot] = {
            "slot": slot,
            "coordinate_frame": config.coordinate_frame,
            "center_xy": [config.center_x, center_y],
            "size_xy": list(config.region_size_xy),
            "bounds": {
                "x": [config.center_x - half_x, config.center_x + half_x],
                "y": [center_y - half_y, center_y + half_y],
            },
        }
    return regions


def sample_dining_targets(
    config: DiningPlacementRegionConfig,
    *,
    episode_seed: int,
) -> dict[str, dict[str, Any]]:
    """按 episode seed 联合采样满足杯间距约束的左右目标。"""

    regions = compute_dining_regions(config)
    samplers: dict[str, dict[str, Any]] = {}
    for slot_index, slot in enumerate(DINING_REGION_SLOTS):
        bounds = regions[slot]["bounds"]
        low = np.asarray(
            [
                bounds["x"][0] + config.sampling_inset,
                bounds["y"][0] + config.sampling_inset,
            ],
            dtype=float,
        )
        high = np.asarray(
            [
                bounds["x"][1] - config.sampling_inset,
                bounds["y"][1] - config.sampling_inset,
            ],
            dtype=float,
        )
        seed_components = [int(episode_seed), DINING_REGION_TARGET_SEED_SALT, slot_index]
        samplers[slot] = {
            "low": low,
            "high": high,
            "seed_components": seed_components,
            "rng": np.random.default_rng(np.random.SeedSequence(seed_components)),
        }
    minimum, maximum = config.target_separation_range
    for attempt in range(1, DINING_TARGET_MAX_SAMPLING_ATTEMPTS + 1):
        sampled = {
            slot: values["rng"].uniform(values["low"], values["high"])
            for slot, values in samplers.items()
        }
        separation = float(np.linalg.norm(sampled["left"] - sampled["right"]))
        if not minimum <= separation <= maximum:
            continue
        return {
            slot: {
                "slot": slot,
                "coordinate_frame": config.coordinate_frame,
                "xy": sampled[slot].astype(float).tolist(),
                "sample_bounds": {
                    "x": [float(samplers[slot]["low"][0]), float(samplers[slot]["high"][0])],
                    "y": [float(samplers[slot]["low"][1]), float(samplers[slot]["high"][1])],
                },
                "episode_seed": int(episode_seed),
                "seed_components": list(samplers[slot]["seed_components"]),
                "sampling_inset": float(config.sampling_inset),
                "joint_sampling_attempt": attempt,
                "paired_target_separation": separation,
                "target_separation_range": [minimum, maximum],
            }
            for slot in DINING_REGION_SLOTS
        }
    raise RuntimeError("could not sample a valid WA-01 dining target pair")


def is_xy_inside_region(
    xy: Any,
    region: Mapping[str, Any],
    *,
    tolerance: float = 1e-9,
) -> bool:
    """判断一个世界坐标 XY 是否位于给定餐桌区域内。"""

    point = np.asarray(xy, dtype=float).reshape(-1)
    bounds = region.get("bounds", {})
    x_bounds = np.asarray(bounds.get("x", []), dtype=float).reshape(-1)
    y_bounds = np.asarray(bounds.get("y", []), dtype=float).reshape(-1)
    if point.size < 2 or x_bounds.size != 2 or y_bounds.size != 2:
        return False
    if not np.all(np.isfinite(point[:2])) or not np.all(np.isfinite(x_bounds)) or not np.all(np.isfinite(y_bounds)):
        return False
    margin = max(0.0, float(tolerance))
    return bool(
        x_bounds[0] - margin <= point[0] <= x_bounds[1] + margin
        and y_bounds[0] - margin <= point[1] <= y_bounds[1] + margin
    )


__all__ = [
    "DINING_REGION_SLOTS",
    "DiningPlacementRegionConfig",
    "compute_dining_regions",
    "dining_region_config_summary",
    "is_xy_inside_region",
    "resolve_dining_region_config",
    "sample_dining_targets",
]
