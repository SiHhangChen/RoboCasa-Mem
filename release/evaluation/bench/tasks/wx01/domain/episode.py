"""Pure, reproducible WX-01 task and episode decisions."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from bench.tasks.wx01.domain.contracts import (
    ENV_NAME,
    TASK_ID,
    EpisodePlan,
    SceneConfig,
    TaskDefinition,
)
from bench.tasks.wx01.domain.geometry import (
    sample_initial_object_poses,
    sample_temporary_offsets,
    sample_wipe_offsets,
)
from bench.tasks.wx01.object_randomization import (
    CUP_MATERIAL_IDS,
    SPONGE_TYPES,
)


CUP_MATERIAL_SEED_SALT = 0x5758434D
SPONGE_TYPE_SEED_SALT = 0x57585350


def resolve_task_definition(**values: Any) -> TaskDefinition:
    """Resolve and validate the stable WX-01 identity."""

    task_id = str(values.get("id", TASK_ID))
    if task_id != TASK_ID:
        raise ValueError(f"WX-01 config must use task.id={TASK_ID!r}")
    conditions = tuple(str(item) for item in values.get("success_conditions", ()))
    if not conditions:
        raise ValueError("WX-01 must declare at least one success condition")
    return TaskDefinition(
        task_id=task_id,
        display_name=str(values.get("display_name", "Return object to original place")),
        description=str(
            values.get("description", "Wipe a remembered spot and return its object.")
        ),
        success_conditions=conditions,
    )


def resolve_scene_config(*, seed: int, env_split: str, **values: Any) -> SceneConfig:
    """Resolve the fixed layout-38 scene used by production WX-01."""

    fixed_scene = bool(values.get("fixed_scene", True))
    rotation_deg = _pair(
        values.get("blocking_object_rotation_deg", (0.0, 0.0)),
        "blocking_object_rotation_deg",
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
        blocking_object_scale=float(values.get("blocking_object_scale", 0.9)),
        blocking_object_placement_size=_pair(
            values.get("blocking_object_placement_size", (0.16, 0.20)),
            "blocking_object_placement_size",
        ),
        blocking_object_placement_pos=_pair(
            values.get("blocking_object_placement_pos", (1.0, 1.0)),
            "blocking_object_placement_pos",
        ),
        blocking_object_rotation=rotation,
    )


def resolve_episode_plan(
    *,
    seed: int,
    episode: Mapping[str, Any],
    object_settings: Mapping[str, Any],
) -> EpisodePlan:
    """Freeze original regions, temporary targets, wipe trajectory and appearance for one seed."""

    values = dict(episode)
    start_mode = str(
        values.pop("start_mode", "short_preplaced_blocking_pregrasped_tool")
    )
    if start_mode != "short_preplaced_blocking_pregrasped_tool":
        raise ValueError(
            "WX-01 production architecture supports the six-stage short start only"
        )
    temporary = values.pop("temporary_targets", {})
    wipe = values.pop("wipe_sampling", {})
    cup_material_candidates = _candidate_pool(
        values.pop("cup_material_candidates", CUP_MATERIAL_IDS),
        allowed=CUP_MATERIAL_IDS,
        label="cup_material_candidates",
    )
    sponge_type_candidates = _candidate_pool(
        values.pop("sponge_type_candidates", SPONGE_TYPES),
        allowed=SPONGE_TYPES,
        label="sponge_type_candidates",
    )
    max_horizon = int(values.pop("max_horizon", 3200))
    if values:
        raise ValueError("unknown WX-01 episode settings: " + ", ".join(sorted(values)))
    if not isinstance(temporary, Mapping) or not isinstance(wipe, Mapping):
        raise ValueError("WX-01 episode target and wipe settings must be mappings")
    blocking_offset, tool_offset, temporary_metadata = sample_temporary_offsets(
        episode_seed=int(seed),
        settings=temporary,
    )
    wipe_offsets, wipe_metadata = sample_wipe_offsets(
        episode_seed=int(seed),
        settings=wipe,
    )
    if max_horizon <= 0:
        raise ValueError("WX-01 max_horizon must be positive")
    return EpisodePlan(
        seed=int(seed),
        cup_material_id=str(
            _episode_rng(seed, CUP_MATERIAL_SEED_SALT).choice(
                cup_material_candidates
            )
        ),
        sponge_type=str(
            _episode_rng(seed, SPONGE_TYPE_SEED_SALT).choice(
                sponge_type_candidates
            )
        ),
        initial_object_poses=sample_initial_object_poses(
            episode_seed=int(seed),
            settings=object_settings,
        ),
        temporary_blocking_offset_xy=blocking_offset,
        temporary_tool_offset_xy=tool_offset,
        temporary_target_metadata=temporary_metadata,
        wipe_offsets_xy=wipe_offsets,
        wipe_metadata=wipe_metadata,
        max_horizon=max_horizon,
    )


def _pair(value: Any, label: str) -> tuple[float, float]:
    array = np.asarray(value, dtype=float).reshape(-1)
    if array.size != 2 or not np.all(np.isfinite(array)):
        raise ValueError(f"WX-01 {label} must contain two finite values")
    return float(array[0]), float(array[1])


def _optional_int(value: Any) -> int | None:
    return None if value is None else int(value)


def _episode_rng(seed: int, salt: int) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence([int(seed), salt]))


def _candidate_pool(
    value: Any,
    *,
    allowed: Sequence[str],
    label: str,
) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"WX-01 {label} must be a non-empty sequence")
    candidates = tuple(str(item) for item in value)
    if not candidates or len(set(candidates)) != len(candidates):
        raise ValueError(f"WX-01 {label} must be non-empty and unique")
    unknown = sorted(set(candidates) - set(allowed))
    if unknown:
        raise ValueError(f"unknown WX-01 {label}: " + ", ".join(unknown))
    return candidates


__all__ = [
    "resolve_episode_plan",
    "resolve_scene_config",
    "resolve_task_definition",
]
