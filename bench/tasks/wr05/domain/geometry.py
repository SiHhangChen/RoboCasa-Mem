"""Pure seeded geometry decisions for WR-05."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from bench.execution.sampling.rng import SeedStream


def sample_joint_return_offsets(
    *,
    seed: int,
    home_assignment: Mapping[str, str],
    half_extents: Any,
    minimum_separation: float,
    nominal_separation: float,
    max_attempts: int = 256,
) -> tuple[dict[str, tuple[float, float]], dict[str, Any]]:
    """Sample both return offsets from the shared shelf-local XY rectangle."""

    resolved_half_extents = np.asarray(half_extents, dtype=float).reshape(-1)
    if (
        resolved_half_extents.size != 2
        or not np.all(np.isfinite(resolved_half_extents))
        or np.any(resolved_half_extents < 0.0)
    ):
        raise ValueError(
            "WR-05 return_place_xy_half_extents must contain two "
            "finite non-negative values"
        )
    required = max(0.0, float(minimum_separation))
    nominal = float(nominal_separation)
    if not np.isfinite(nominal) or nominal <= 0.0:
        raise ValueError("WR-05 nominal shelf separation must be positive")
    condiment_by_home = {
        str(region): str(condiment)
        for condiment, region in dict(home_assignment).items()
    }
    lower_name = condiment_by_home["shelf_lower"]
    upper_name = condiment_by_home["shelf_upper"]
    if nominal < required:
        raise ValueError("WR-05 shelf-level separation is infeasible")
    rng = SeedStream(int(seed), "wr_05:return_place_joint").rng()

    def rectangle_sample() -> np.ndarray:
        return rng.uniform(-resolved_half_extents, resolved_half_extents)

    selected = {"ketchup": rectangle_sample(), "mustard": rectangle_sample()}
    offsets = {
        name: (float(selected[name][0]), float(selected[name][1]))
        for name in ("ketchup", "mustard")
    }
    metadata = {
        "seed": int(seed),
        "sampling_shape": "rectangle",
        "half_extents": resolved_half_extents.astype(float).tolist(),
        "minimum_separation": required,
        "nominal_separation": nominal,
        "vertical_separation": nominal,
        "lower_condiment": lower_name,
        "upper_condiment": upper_name,
        "attempt": 1,
        "fallback_used": False,
    }
    return offsets, metadata


__all__ = ["sample_joint_return_offsets"]
