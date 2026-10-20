"""Pure TS03 identity resolution and full K/N episode sampling."""

from __future__ import annotations

from typing import Any

import numpy as np

from bench.tasks.ts03.domain.contracts import (
    DISPLAY_NAME,
    ENV_NAME,
    RECEPTACLE_NAME,
    SUGAR_PLATE_NAME,
    TASK_ID,
    EpisodePlan,
    SceneConfig,
    SugarCountProfile,
    TaskDefinition,
    profile_for_counts,
    profile_for_id,
    profile_for_seed,
)
from bench.tasks.ts03.domain.geometry import SUGAR_RELEASE_REGION_NAMES
from bench.tasks.ts03.materials import (
    SAUCEPAN_ASSET_BY_TYPE,
    SAUCEPAN_MATERIALS,
    SAUCEPAN_TYPES,
    SUGAR_CUBE_MATERIALS,
    SUGAR_PLATE_MATERIALS,
)


# Keep these streams identical to the calibrated TS01 physical task. TS03 owns
# its implementation; the six K/N profiles only specialize count sampling.
TARGET_ORDER_SALT = 0x54415247
LAYOUT_SALT = 0x504C4154
RELEASE_REGION_ORDER_SALT = 0x5245474E
SAUCEPAN_TYPE_SALT = 0x53505459
SAUCEPAN_MATERIAL_SALT = 0x53504D41
SUGAR_PLATE_MATERIAL_SALT = 0x53504C54
SUGAR_CUBE_MATERIAL_SALT = 0x53504355


def normalize_target_sugar_count(value: Any | None) -> int | None:
    if value is None or value == "":
        return None
    result = int(value)
    if result not in (1, 2, 3):
        raise ValueError("TS03 target_sugar_count must be 1, 2, or 3")
    return result


def normalize_total_sugar_count(
    value: Any | None,
    *,
    target_sugar_count: int | None = None,
) -> int | None:
    if value is None or value == "":
        return None
    result = int(value)
    if result < 1 or result > 3 or (
        target_sugar_count is not None and result < int(target_sugar_count)
    ):
        raise ValueError("TS03 total_sugar_count must satisfy K <= N <= 3")
    return result


def resolve_task_definition(**values: Any) -> TaskDefinition:
    task_id = str(values.get("id", TASK_ID))
    if task_id != TASK_ID:
        raise ValueError(f"TS03 config must use task.id={TASK_ID!r}")
    conditions = tuple(
        str(item)
        for item in values.get(
            "success_conditions",
            (
                "exact_target_sugar_count",
                "no_extra_sugar_added",
                "sugars_released",
                "completion_button_pressed",
            ),
        )
    )
    if not conditions:
        raise ValueError("TS03 must declare at least one success condition")
    return TaskDefinition(
        task_id=TASK_ID,
        display_name=str(values.get("display_name", DISPLAY_NAME)),
        description=str(
            values.get(
                "description",
                "Complete one of six K/N sugar-count profiles.",
            )
        ),
        success_conditions=conditions,
    )


def resolve_scene_config(*, seed: int, env_split: str, **values: Any) -> SceneConfig:
    del seed
    if not bool(values.get("fixed_scene", True)):
        raise ValueError("TS03 requires fixed_scene=true")
    env_name = str(values.get("env_name", ENV_NAME))
    if env_name != ENV_NAME:
        raise ValueError(f"TS03 config must use scene.env_name={ENV_NAME!r}")
    if int(values.get("layout_id", 39)) != 39:
        raise ValueError("TS03 is fixed to layout_id=39")
    if int(values.get("style_id", 37)) != 37:
        raise ValueError("TS03 is fixed to style_id=37")
    if int(values.get("scene_seed", 0)) != 0:
        raise ValueError("TS03 is fixed to scene_seed=0")
    return SceneConfig(
        env_name=env_name,
        env_split=str(env_split),
        layout_id=39,
        style_id=37,
        scene_seed=0,
        fixed_scene=True,
        sugar_plate_name=str(values.get("sugar_plate_name", SUGAR_PLATE_NAME)),
        receptacle_name=str(values.get("receptacle_name", RECEPTACLE_NAME)),
    )


def resolve_profile_selection(
    *,
    seed: int,
    profile_id: Any | None = None,
    target_sugar_count: Any | None = None,
    total_sugar_count: Any | None = None,
) -> SugarCountProfile:
    has_profile = profile_id is not None and str(profile_id).strip() != ""
    has_target = target_sugar_count is not None and target_sugar_count != ""
    has_total = total_sugar_count is not None and total_sugar_count != ""
    if has_target != has_total:
        raise ValueError("TS03 K and N overrides must be provided together")
    selected = profile_for_id(profile_id) if has_profile else None
    if has_target:
        counts_profile = profile_for_counts(target_sugar_count, total_sugar_count)
        if selected is not None and selected != counts_profile:
            raise ValueError(
                f"TS03 profile_id={selected.profile_id!r} conflicts with "
                f"K={target_sugar_count}, N={total_sugar_count}"
            )
        selected = counts_profile
    return profile_for_seed(seed) if selected is None else selected


def resolve_episode_plan(
    *,
    seed: int,
    profile_id: Any | None = None,
    target_sugar_count: Any | None = None,
    total_sugar_count: Any | None = None,
    max_total_sugar_count: Any = 3,
    sugar_name_prefix: str = "sugar",
    target_sugar_names: Any | None = None,
    max_horizon: Any = 2800,
    saucepan_type_candidates: Any = SAUCEPAN_TYPES,
    saucepan_material_candidates: Any = SAUCEPAN_MATERIALS,
    sugar_plate_material_candidates: Any = SUGAR_PLATE_MATERIALS,
    sugar_cube_material_candidates: Any = SUGAR_CUBE_MATERIALS,
    **unknown: Any,
) -> EpisodePlan:
    if unknown:
        raise ValueError("unknown TS03 episode settings: " + ", ".join(sorted(unknown)))
    episode_seed = int(seed)
    if episode_seed < 0:
        raise ValueError("TS03 seed must be non-negative")
    if int(max_total_sugar_count) != 3:
        raise ValueError("TS03 requires max_total_sugar_count=3")
    profile = resolve_profile_selection(
        seed=episode_seed,
        profile_id=profile_id,
        target_sugar_count=target_sugar_count,
        total_sugar_count=total_sugar_count,
    )
    total = profile.total_sugar_count
    prefix = str(sugar_name_prefix).strip()
    if not prefix:
        raise ValueError("TS03 sugar name prefix must not be empty")
    names = tuple(f"{prefix}{index}" for index in range(total))
    supplied_targets = _normalize_target_names(
        target_sugar_names,
        sugar_names=names,
        target_count=profile.target_sugar_count,
    )
    targets = (
        supplied_targets
        if supplied_targets is not None
        else _select_target_sugar_names(
            names,
            target_count=profile.target_sugar_count,
            seed=episode_seed,
        )
    )
    distractors = tuple(name for name in names if name not in set(targets))
    release_region_orders = tuple(
        _select_sugar_release_region_order(seed=episode_seed, target_index=index)
        for index in range(1, profile.target_sugar_count + 1)
    )
    layout_seed = int(_rng(episode_seed, LAYOUT_SALT).integers(0, 2**31 - 1))
    horizon = int(max_horizon)
    if horizon <= 0:
        raise ValueError("TS03 max_horizon must be positive")
    saucepan_types = _candidate_pool(
        saucepan_type_candidates, allowed=SAUCEPAN_TYPES, label="saucepan_type_candidates"
    )
    saucepan_materials = _candidate_pool(
        saucepan_material_candidates, allowed=SAUCEPAN_MATERIALS, label="saucepan_material_candidates"
    )
    plate_materials = _candidate_pool(
        sugar_plate_material_candidates, allowed=SUGAR_PLATE_MATERIALS, label="sugar_plate_material_candidates"
    )
    cube_materials = _candidate_pool(
        sugar_cube_material_candidates, allowed=SUGAR_CUBE_MATERIALS, label="sugar_cube_material_candidates"
    )
    saucepan_type = str(_rng(episode_seed, SAUCEPAN_TYPE_SALT).choice(saucepan_types))
    saucepan_material = str(_rng(episode_seed, SAUCEPAN_MATERIAL_SALT).choice(saucepan_materials))
    sugar_plate_material = str(_rng(episode_seed, SUGAR_PLATE_MATERIAL_SALT).choice(plate_materials))
    sugar_cube_material = str(_rng(episode_seed, SUGAR_CUBE_MATERIAL_SALT).choice(cube_materials))
    return EpisodePlan(
        seed=episode_seed,
        target_sugar_count=profile.target_sugar_count,
        total_sugar_count=total,
        sugar_names=names,
        target_sugar_names=tuple(targets),
        distractor_sugar_names=distractors,
        sugar_release_region_orders=release_region_orders,
        sugar_plate_layout_seed=layout_seed,
        max_horizon=horizon,
        saucepan_type=saucepan_type,
        saucepan_asset=SAUCEPAN_ASSET_BY_TYPE[saucepan_type],
        saucepan_material=saucepan_material,
        sugar_plate_material=sugar_plate_material,
        sugar_cube_material=sugar_cube_material,
    )


def _select_target_sugar_names(
    sugar_names: tuple[str, ...],
    *,
    target_count: int,
    seed: int,
) -> tuple[str, ...]:
    rng = _rng(seed, TARGET_ORDER_SALT)
    indices = np.asarray(rng.permutation(len(sugar_names)), dtype=int)[:target_count]
    return tuple(sugar_names[int(index)] for index in indices)


def _select_sugar_release_region_order(
    *,
    seed: int,
    target_index: int,
) -> tuple[str, ...]:
    rng = np.random.default_rng(
        np.random.SeedSequence(
            [int(seed), int(RELEASE_REGION_ORDER_SALT), int(target_index)]
        )
    )
    permutation = np.asarray(
        rng.permutation(len(SUGAR_RELEASE_REGION_NAMES)),
        dtype=int,
    )
    return tuple(SUGAR_RELEASE_REGION_NAMES[int(index)] for index in permutation)


def _normalize_target_names(
    value: Any | None,
    *,
    sugar_names: tuple[str, ...],
    target_count: int,
) -> tuple[str, ...] | None:
    if value is None or value == "":
        return None
    if isinstance(value, str):
        raw = value.replace(";", ",").split(",")
    else:
        raw = list(value)
    names = tuple(str(item).strip() for item in raw if str(item).strip())
    if len(names) != int(target_count) or len(set(names)) != len(names):
        raise ValueError("TS03 target_sugar_names must contain exactly K unique names")
    unknown = set(names) - set(sugar_names)
    if unknown:
        raise ValueError(
            f"TS03 target_sugar_names contains unknown names: {sorted(unknown)}"
        )
    return names


def _rng(seed: int, salt: int) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence([int(seed), int(salt)]))


def _candidate_pool(value: Any, *, allowed: tuple[str, ...], label: str) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not hasattr(value, "__iter__"):
        raise ValueError(f"TS03 {label} must be a non-empty sequence")
    candidates = tuple(str(item).strip() for item in value)
    if not candidates or len(set(candidates)) != len(candidates):
        raise ValueError(f"TS03 {label} must be non-empty and unique")
    unknown = sorted(set(candidates) - set(allowed))
    if unknown:
        raise ValueError(f"unknown TS03 {label}: {', '.join(unknown)}")
    return candidates


__all__ = [
    "normalize_target_sugar_count",
    "normalize_total_sugar_count",
    "resolve_episode_plan",
    "resolve_profile_selection",
    "resolve_scene_config",
    "resolve_task_definition",
]
