"""Pure seeded geometry helpers retained for WR-08 tooling."""

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
    """Sample two separated release offsets in a shared local rectangle."""

    resolved_half_extents = np.asarray(half_extents, dtype=float).reshape(-1)
    if (
        resolved_half_extents.size != 2
        or not np.all(np.isfinite(resolved_half_extents))
        or np.any(resolved_half_extents < 0.0)
    ):
        raise ValueError(
            "WR-08 return_place_xy_half_extents must contain two "
            "finite non-negative values"
        )
    required = max(0.0, float(minimum_separation))
    nominal = float(nominal_separation)
    if not np.isfinite(nominal) or nominal <= 0.0:
        raise ValueError("WR-08 nominal release separation must be positive")
    item_by_region = {
        str(region): str(item)
        for item, region in dict(home_assignment).items()
    }
    if len(item_by_region) != 2:
        raise ValueError("WR-08 home_assignment must contain two distinct zones")
    item_names = tuple(sorted(item_by_region.values()))
    if nominal < required:
        raise ValueError("WR-08 release separation is infeasible")
    rng = SeedStream(int(seed), "wr_08:return_place_joint").rng()

    def rectangle_sample() -> np.ndarray:
        return rng.uniform(-resolved_half_extents, resolved_half_extents)

    selected: dict[str, np.ndarray] = {}
    for item in item_names:
        for attempt in range(1, max(1, int(max_attempts)) + 1):
            candidate = rectangle_sample()
            if not selected or all(
                float(np.linalg.norm(candidate - previous)) >= required
                for previous in selected.values()
            ):
                selected[item] = candidate
                break
        else:
            raise ValueError("WR-08 could not sample separated release offsets")
    offsets = {
        name: (float(selected[name][0]), float(selected[name][1]))
        for name in item_names
    }
    metadata = {
        "seed": int(seed),
        "sampling_shape": "rectangle",
        "half_extents": resolved_half_extents.astype(float).tolist(),
        "minimum_separation": required,
        "nominal_separation": nominal,
        "items": list(item_names),
        "attempt": int(attempt),
        "fallback_used": bool(attempt > 1),
    }
    return offsets, metadata


__all__ = ["sample_joint_return_offsets"]
