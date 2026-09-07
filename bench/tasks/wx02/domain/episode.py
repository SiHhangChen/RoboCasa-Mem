"""Pure, reproducible WX-02 task and episode decisions."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from bench.tasks.wx02.domain.contracts import (
    CUP_NAMES,
    ENV_NAME,
    TASK_ID,
    EpisodePlan,
    SceneConfig,
    TaskDefinition,
)
from bench.tasks.wx02.domain.geometry import (
    sample_initial_object_poses,
    sample_temporary_targets,
    sample_wipe_offsets,
)


def resolve_task_definition(**values: Any) -> TaskDefinition:
    """Resolve and validate the stable WX-02 identity."""

    task_id = str(values.get("id", TASK_ID))
    if task_id != TASK_ID:
        raise ValueError(f"WX-02 config must use task.id={TASK_ID!r}")
    conditions = tuple(str(item) for item in values.get("success_conditions", ()))
    if not conditions:
        raise ValueError("WX-02 must declare at least one success condition")
    return TaskDefinition(
        task_id=task_id,
        display_name=str(values.get("display_name", "Clean under a cup and return it")),
        description=str(
            values.get(
                "description",
                "Fetch a sponge, clean the stain under a cup, and return both objects.",
            )
        ),
        success_conditions=conditions,
    )


def resolve_scene_config(*, seed: int, env_split: str, **values: Any) -> SceneConfig:
    """Resolve the fixed layout-38 scene used by production WX-02."""

    fixed_scene = bool(values.get("fixed_scene", True))
    rotation_deg = _pair(
        values.get("cup_rotation_deg", (0.0, 0.0)),
        "cup_rotation_deg",
    )
    rotation = tuple(
        float(item) for item in np.deg2rad(np.asarray(rotation_deg, dtype=float))
    )
    return SceneConfig(
        env_name=str(values.get("env_name", ENV_NAME)),
        env_split=str(env_split),
        layout_id=38 if fixed_scene else _optional_int(values.get("layout_id")),
        style_id=38 if fixed_scene else _optional_int(values.get("style_id")),
        scene_seed=int(values.get("scene_seed", 0)) if fixed_scene else int(seed),
        fixed_scene=fixed_scene,
        cup_scale=float(values.get("cup_scale", 0.9)),
        cup_placement_size=_pair(
            values.get("cup_placement_size", (0.16, 0.20)),
            "cup_placement_size",
        ),
        primary_cup_placement_pos=_pair(
            values.get("primary_cup_placement_pos", (1.0, 1.0)),
            "primary_cup_placement_pos",
        ),
        secondary_cup_placement_pos=_pair(
            values.get("secondary_cup_placement_pos", (1.0, 0.55)),
            "secondary_cup_placement_pos",
        ),
        cup_rotation=rotation,
        cleaning_tool_return_region_center_xy=_pair(
            values.get("cleaning_tool_return_region_center_xy", (2.50, -4.22)),
            "cleaning_tool_return_region_center_xy",
        ),
        cleaning_tool_return_region_size_xy=_positive_pair(
            values.get("cleaning_tool_return_region_size_xy", (0.22, 0.20)),
            "cleaning_tool_return_region_size_xy",
        ),
        cleaning_tool_return_region_rgba=_rgba(
            values.get(
                "cleaning_tool_return_region_rgba",
                (0.96, 0.72, 0.16, 0.72),
            ),
            "cleaning_tool_return_region_rgba",
        ),
    )


def resolve_episode_plan(
    *,
    seed: int,
    episode: Mapping[str, Any],
    object_settings: Mapping[str, Any],
) -> EpisodePlan:
    """Freeze original regions, temporary targets and wipe trajectory for one seed."""

    values = dict(episode)
    start_mode = str(
        values.pop("start_mode", "cup_front_unheld")
    )
    if start_mode != "cup_front_unheld":
        raise ValueError(
            "WX-02 production architecture supports the fourteen-stage cup-front start only"
        )
    temporary = values.pop("temporary_targets", {})
    wipe = values.pop("wipe_sampling", {})
    max_horizon = int(values.pop("max_horizon", 3200))
    if values:
        raise ValueError("unknown WX-02 episode settings: " + ", ".join(sorted(values)))
    if not isinstance(temporary, Mapping) or not isinstance(wipe, Mapping):
        raise ValueError("WX-02 episode target and wipe settings must be mappings")
    object_targets, staged_tool_xy, temporary_metadata = sample_temporary_targets(
        episode_seed=int(seed),
        settings=temporary,
    )
    wipe_offsets_by_object: dict[str, tuple[tuple[float, float], ...]] = {}
    wipe_metadata_by_object: dict[str, Mapping[str, Any]] = {}
    for index, object_name in enumerate(CUP_NAMES):
        object_wipe = dict(wipe)
        object_wipe["seed_offset"] = int(wipe.get("seed_offset", 2000)) + index * 1000
        offsets, metadata = sample_wipe_offsets(
            episode_seed=int(seed),
            settings=object_wipe,
        )
        wipe_offsets_by_object[object_name] = offsets
        wipe_metadata_by_object[object_name] = metadata
    if max_horizon <= 0:
        raise ValueError("WX-02 max_horizon must be positive")
    return EpisodePlan(
        seed=int(seed),
        initial_object_poses=sample_initial_object_poses(
            episode_seed=int(seed),
            settings=object_settings,
        ),
        temporary_object_xy=object_targets,
        staged_tool_xy=staged_tool_xy,
        temporary_target_metadata=temporary_metadata,
        wipe_offsets_xy_by_object=wipe_offsets_by_object,
        wipe_metadata_by_object=wipe_metadata_by_object,
        max_horizon=max_horizon,
    )


def _pair(value: Any, label: str) -> tuple[float, float]:
    array = np.asarray(value, dtype=float).reshape(-1)
    if array.size != 2 or not np.all(np.isfinite(array)):
        raise ValueError(f"WX-02 {label} must contain two finite values")
    return float(array[0]), float(array[1])


def _positive_pair(value: Any, label: str) -> tuple[float, float]:
    pair = _pair(value, label)
    if any(item <= 0.0 for item in pair):
        raise ValueError(f"WX-02 {label} must contain positive values")
    return pair


def _rgba(value: Any, label: str) -> tuple[float, float, float, float]:
    array = np.asarray(value, dtype=float).reshape(-1)
    if (
        array.size != 4
        or not np.all(np.isfinite(array))
        or np.any(array < 0.0)
        or np.any(array > 1.0)
    ):
        raise ValueError(f"WX-02 {label} must contain four values in [0, 1]")
    return tuple(float(item) for item in array)


def _optional_int(value: Any) -> int | None:
    return None if value is None else int(value)


__all__ = [
    "resolve_episode_plan",
    "resolve_scene_config",
    "resolve_task_definition",
]
