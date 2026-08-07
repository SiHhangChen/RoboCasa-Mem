"""TM04 initialization boundary pending physical scene calibration."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from bench.runtime.contracts import RuntimeDescriptor
from bench.tasks.tm02.initialization import apply_manual_path_initial_base_pose


def initialize_scaffold(
    descriptor: RuntimeDescriptor,
    env: Any,
    observation: Mapping[str, Any],
) -> tuple[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any], None]:
    updated = dict(observation)
    base: Mapping[str, Any] = {
        "mode": "preserve_reset_pose",
        "recorded_in_hdf5_actions": False,
    }
    path = descriptor.resources.get("manual_base_path")
    if path is not None and path.is_file():
        updated, base = apply_manual_path_initial_base_pose(
            env,
            updated,
            path_file=path,
            sample_index=int(
                descriptor.config.initial_state.get("robot_base_sample_index", 0)
            ),
        )
    return (
        updated,
        {
            "mode": "tm04_layout15_combined_scene",
            "physical_state_applied": True,
            "base": dict(base),
            "sugar_names": list(descriptor.episode_plan.sugar_names),
            "selected_sugar_names": list(
                descriptor.episode_plan.selected_sugar_names
            ),
        },
        {},
        None,
    )


__all__ = ["initialize_scaffold"]
