"""Translate WR-01 cabinet regions into world targets."""

from __future__ import annotations

from typing import Any

import numpy as np


def storage_target(controller: Any, level: str) -> dict[str, Any]:
    env = controller.env
    position = np.asarray(env.cabinet_for_level(level).interior_floor_pose(), dtype=float)
    return {
        "target_name": f"{level}_cabinet_interior",
        "target_pos_world": position.tolist(),
        "target_info": {"target_kind": "cabinet_interior", "level": str(level)},
    }


__all__ = ["storage_target"]
