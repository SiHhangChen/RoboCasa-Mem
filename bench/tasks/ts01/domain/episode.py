"""Pure, reproducible TS01 task and episode decisions."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from bench.tasks.ts01.domain.contracts import (
    DEFAULT_MAX_TOTAL_SUGAR_COUNT,
    DISPLAY_NAME,
    ENV_NAME,
    MAX_SUPPORTED_TOTAL_SUGAR_COUNT,
    RECEPTACLE_NAME,
    SUGAR_PLATE_NAME,
    TASK_ID,
    VALID_TARGET_SUGAR_COUNTS,
    EpisodePlan,
    SceneConfig,
    TaskDefinition,
)
from bench.tasks.ts01.domain.geometry import SUGAR_RELEASE_REGION_NAMES


K_SALT = 0x54533031
N_SALT = 0x4E534330
TARGET_ORDER_SALT = 0x54415247
LAYOUT_SALT = 0x504C4154
RELEASE_REGION_ORDER_SALT = 0x5245474E


def normalize_target_sugar_count(value: Any | None) -> int | None:
    """Normalize an optional K override."""

    if value is None or value == "":
        return None
    try:
        count = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"target_sugar_count must be an integer, got {value!r}") from exc
    if count not in VALID_TARGET_SUGAR_COUNTS:
        raise ValueError(
            "target_sugar_count must be one of "
            f"{VALID_TARGET_SUGAR_COUNTS}, got {value!r}"
        )
    return count


def normalize_total_sugar_count(
    value: Any | None,
    *,
    target_sugar_count: int | None = None,
) -> int | None:
    """Normalize an optional N override and enforce N >= K."""

    if value is None or value == "":
        return None
    try:
        count = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"total_sugar_count must be an integer, got {value!r}") from exc
    if count < 1 or count > MAX_SUPPORTED_TOTAL_SUGAR_COUNT:
        raise ValueError(
            "total_sugar_count must be between 1 and "
            f"{MAX_SUPPORTED_TOTAL_SUGAR_COUNT}, got {value!r}"
        )
    if target_sugar_count is not None and count < int(target_sugar_count):
        raise ValueError(
            "total_sugar_count must satisfy N >= K; "
            f"got N={count}, K={int(target_sugar_count)}"
        )
    return count


def sugar_names_for_count(count: int, *, prefix: str = "sugar") -> tuple[str, ...]:
    """Return deterministic names for the N visible sugar cubes."""

    normalized = normalize_total_sugar_count(count)
    if normalized is None:
        raise ValueError("total_sugar_count is required")
    prefix = str(prefix).strip()
    if not prefix:
        raise ValueError("sugar name prefix must not be empty")
    return tuple(f"{prefix}{index}" for index in range(normalized))


def select_target_sugar_names(
    sugar_names: tuple[str, ...] | list[str],
    *,
    target_count: int,
    seed: int,
) -> tuple[str, ...]:
    """Select an ordered K-subset independently of simulator RNG state."""

    names = tuple(str(name) for name in sugar_names)
    count = int(target_count)
    if count < 1 or count > len(names):
        raise ValueError(f"target_count must satisfy 1 <= K <= N; got K={count}, N={len(names)}")
    if len(set(names)) != len(names):
        raise ValueError("sugar_names must be unique")
    rng = _rng(seed, TARGET_ORDER_SALT)
    indices = np.asarray(rng.permutation(len(names)), dtype=int)[:count]
    return tuple(names[int(index)] for index in indices)


def select_sugar_release_region_order(
    *,
    seed: int,
    target_index: int,
) -> tuple[str, ...]:
    """Return one independently sampled, reproducible release-region order.

    Each target sugar owns a separate seed stream. Therefore regions are
    sampled with replacement across sugars, while every sugar still receives
    a complete permutation for deterministic fallback.
    """

    index = int(target_index)
    if index < 1:
        raise ValueError("target_index must be positive")
    rng = np.random.default_rng(
        np.random.SeedSequence(
            [int(seed), int(RELEASE_REGION_ORDER_SALT), index]
        )
    )
    permutation = np.asarray(rng.permutation(len(SUGAR_RELEASE_REGION_NAMES)), dtype=int)
    return tuple(SUGAR_RELEASE_REGION_NAMES[int(item)] for item in permutation)


def resolve_task_definition(**values: Any) -> TaskDefinition:
    """Resolve and validate the stable TS01 identity."""

    task_id = str(values.get("id", TASK_ID))
    if task_id != TASK_ID:
        raise ValueError(f"TS01 config must use task.id={TASK_ID!r}")
    conditions = tuple(
        str(item)
        for item in values.get(
            "success_conditions",
            ("exact_target_sugar_count", "no_extra_sugar_added", "sugars_released"),
        )
    )
    if not conditions:
        raise ValueError("TS01 must declare at least one success condition")
    return TaskDefinition(
        task_id=task_id,
        display_name=str(values.get("display_name", DISPLAY_NAME)),
        description=str(
            values.get(
                "description",
                "Add exactly K sugar cubes to the hot chocolate saucepan.",
            )
        ),
        success_conditions=conditions,
    )


def resolve_scene_config(*, seed: int, env_split: str, **values: Any) -> SceneConfig:
    """Resolve the fixed layout-39/style-37 scene used by production TS01."""

    fixed_scene = bool(values.get("fixed_scene", True))
    if not fixed_scene:
        raise ValueError("TS01 requires fixed_scene=true")
    env_name = str(values.get("env_name", ENV_NAME))
    if env_name != ENV_NAME:
        raise ValueError(f"TS01 config must use scene.env_name={ENV_NAME!r}")
    if int(values.get("layout_id", 39)) != 39:
        raise ValueError("TS01 is fixed to layout_id=39")
    if int(values.get("style_id", 37)) != 37:
        raise ValueError("TS01 is fixed to style_id=37")
    if int(values.get("scene_seed", 0)) != 0:
        raise ValueError("TS01 is fixed to scene_seed=0")
    return SceneConfig(
        env_name=env_name,
        env_split=str(env_split),
        layout_id=39,
        style_id=37,
        scene_seed=0,
        fixed_scene=fixed_scene,
        sugar_plate_name=str(values.get("sugar_plate_name", SUGAR_PLATE_NAME)),
        receptacle_name=str(values.get("receptacle_name", RECEPTACLE_NAME)),
    )


def resolve_episode_plan(
    *,
    seed: int,
    target_sugar_count: Any | None = None,
    total_sugar_count: Any | None = None,
    max_total_sugar_count: Any = DEFAULT_MAX_TOTAL_SUGAR_COUNT,
    sugar_name_prefix: str = "sugar",
    target_sugar_names: Any | None = None,
    max_horizon: Any = 2400,
    **unknown: Any,
) -> EpisodePlan:
    """Freeze K, N, target order and all episode-level random decisions."""

    if unknown:
        raise ValueError("unknown TS01 episode settings: " + ", ".join(sorted(unknown)))
    episode_seed = int(seed)
    if episode_seed < 0:
        raise ValueError("TS01 seed must be non-negative")
    normalized_max = normalize_total_sugar_count(max_total_sugar_count)
    if normalized_max is None:
        raise ValueError("max_total_sugar_count is required")

    target_override = normalize_target_sugar_count(target_sugar_count)
    total_override = normalize_total_sugar_count(
        total_sugar_count,
        target_sugar_count=target_override,
    )
    if total_override is not None:
        valid_targets = [k for k in VALID_TARGET_SUGAR_COUNTS if k <= total_override]
    else:
        valid_targets = [k for k in VALID_TARGET_SUGAR_COUNTS if k <= normalized_max]
    if not valid_targets:
        raise ValueError("no valid target K is available for the configured maximum N")

    target = target_override
    if target is None:
        target_rng = _rng(episode_seed, K_SALT)
        target = int(target_rng.choice(valid_targets))
    if total_override is None:
        if normalized_max < target:
            raise ValueError(f"max_total_sugar_count must satisfy max N >= K; got {normalized_max}, {target}")
        total_rng = _rng(episode_seed, N_SALT)
        total = int(total_rng.integers(int(target), normalized_max + 1))
    else:
        total = total_override
    normalize_total_sugar_count(total, target_sugar_count=target)

    sugar_names = sugar_names_for_count(total, prefix=sugar_name_prefix)
    supplied_targets = _normalize_target_names(target_sugar_names, sugar_names, target)
    targets = (
        supplied_targets
        if supplied_targets is not None
        else select_target_sugar_names(sugar_names, target_count=target, seed=episode_seed)
    )
    distractors = tuple(name for name in sugar_names if name not in set(targets))
    release_region_orders = tuple(
        select_sugar_release_region_order(seed=episode_seed, target_index=index)
        for index in range(1, int(target) + 1)
    )
    layout_seed = int(_rng(episode_seed, LAYOUT_SALT).integers(0, 2**31 - 1))
    horizon = int(max_horizon)
    if horizon <= 0:
        raise ValueError("TS01 max_horizon must be positive")
    return EpisodePlan(
        seed=episode_seed,
        target_sugar_count=int(target),
        total_sugar_count=int(total),
        sugar_names=sugar_names,
        target_sugar_names=tuple(targets),
        distractor_sugar_names=distractors,
        sugar_release_region_orders=release_region_orders,
        sugar_plate_layout_seed=layout_seed,
        max_horizon=horizon,
    )


def _normalize_target_names(
    value: Any | None,
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
        raise ValueError("target_sugar_names must contain exactly K unique names")
    unknown = set(names) - set(sugar_names)
    if unknown:
        raise ValueError(f"target_sugar_names contains unknown names: {sorted(unknown)}")
    return names


def _rng(seed: int, salt: int) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence([int(seed), int(salt)]))


def _optional_int(value: Any) -> int | None:
    return None if value is None else int(value)


__all__ = [
    "K_SALT",
    "LAYOUT_SALT",
    "N_SALT",
    "RELEASE_REGION_ORDER_SALT",
    "TARGET_ORDER_SALT",
    "normalize_target_sugar_count",
    "normalize_total_sugar_count",
    "resolve_episode_plan",
    "resolve_scene_config",
    "resolve_task_definition",
    "select_target_sugar_names",
    "select_sugar_release_region_order",
    "sugar_names_for_count",
]
