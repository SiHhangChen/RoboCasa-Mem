"""Pure, reproducible episode decisions for WR-01."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from bench.tasks.wr01.cabinet_materials import (
    CABINET_MATERIAL_TEXTURES,
    normalize_cabinet_material_candidates,
    sample_cabinet_material,
)

from bench.tasks.wr01.domain.contracts import (
    CABINET_LEVELS,
    ENV_NAME,
    ITEM_IDS,
    TASK_ID,
    EpisodePlan,
    SceneConfig,
    TaskDefinition,
)


_PLACEMENT_SALT = 0x57523150
_TARGET_SALT = 0x57523154
_CLOSE_ORDER_SALT = 0x57523143
_CLOSE_ORDER_PERIOD = 200


def validate_item_id(value: Any) -> str:
    item_id = str(value)
    if item_id not in ITEM_IDS:
        raise ValueError(f"invalid WR-01 item id: {item_id!r}")
    return item_id


def validate_level(value: Any) -> str:
    level = str(value)
    if level not in CABINET_LEVELS:
        raise ValueError(f"invalid WR-01 cabinet level: {level!r}")
    return level


def storage_assignment_for_seed(seed: int) -> dict[str, str]:
    """Independently sample the visible item-to-cabinet relation."""

    rng = np.random.default_rng(np.random.SeedSequence([int(seed), _PLACEMENT_SALT]))
    items = rng.permutation(np.asarray(ITEM_IDS))
    return {level: str(items[index]) for index, level in enumerate(CABINET_LEVELS)}


def target_item_for_seed(seed: int) -> str:
    """Sample the delayed retrieval request independently of item placement."""

    rng = np.random.default_rng(np.random.SeedSequence([int(seed), _TARGET_SALT]))
    return ITEM_IDS[int(rng.integers(0, len(ITEM_IDS)))]


def close_order_for_seed(seed: int) -> tuple[str, str]:
    """Sample the expert's door-close order; it is not an evaluation constraint."""
    # Use a dedicated seed stream rather than the item/target streams.  The
    # affine permutation makes the first 200 release seeds exactly balanced
    # while avoiding the direct seed-parity correlation used by older drafts.
    bucket = (int(seed) * 73 + _CLOSE_ORDER_SALT) % _CLOSE_ORDER_PERIOD
    return ("lower", "upper") if bucket < _CLOSE_ORDER_PERIOD // 2 else ("upper", "lower")


def resolve_task_definition(**values: Any) -> TaskDefinition:
    task_id = str(values.get("id", TASK_ID))
    if task_id != TASK_ID:
        raise ValueError(f"WR-01 config must use task.id={TASK_ID!r}")
    conditions = tuple(str(item) for item in values.get("success_conditions", ()))
    if not conditions:
        raise ValueError("WR-01 must declare final success conditions")
    return TaskDefinition(
        task_id=task_id,
        display_name=str(values.get("display_name", "Retrieve remembered colored can")),
        description=str(
            values.get(
                "description",
                "Observe two pre-filled open cabinets, close both doors, then "
                "retrieve the requested can by memory.",
            )
        ),
        success_conditions=conditions,
    )


def resolve_scene_config(*, seed: int, env_split: str, **values: Any) -> SceneConfig:
    fixed_scene = bool(values.get("fixed_scene", True))
    return SceneConfig(
        env_name=str(values.get("env_name", ENV_NAME)),
        env_split=str(env_split),
        layout_id=24 if fixed_scene else _optional_int(values.get("layout_id")),
        style_id=24 if fixed_scene else _optional_int(values.get("style_id")),
        scene_seed=int(values.get("scene_seed", 0)) if fixed_scene else int(seed),
        fixed_scene=fixed_scene,
    )


def resolve_episode_plan(*, seed: int, **values: Any) -> EpisodePlan:
    """Freeze the relation and the later request before runtime construction."""

    assignment = storage_assignment_for_seed(seed)
    configured_assignment = values.get("item_by_level")
    if configured_assignment is not None:
        if not isinstance(configured_assignment, Mapping):
            raise ValueError("episode.item_by_level must be a mapping")
        assignment = {
            level: validate_item_id(configured_assignment[level])
            for level in CABINET_LEVELS
        }
        if set(assignment.values()) != set(ITEM_IDS):
            raise ValueError("episode.item_by_level must assign each item exactly once")
    level_by_item = {item: level for level, item in assignment.items()}
    configured_target = values.get("target_item_id")
    target = (
        target_item_for_seed(seed)
        if configured_target is None
        else validate_item_id(configured_target)
    )
    configured_order = values.get("close_order")
    if configured_order is None:
        close_order = close_order_for_seed(seed)
    else:
        if isinstance(configured_order, (str, bytes)) or len(configured_order) != 2:
            raise ValueError("episode.close_order must contain lower and upper")
        close_order = tuple(validate_level(v) for v in configured_order)
        if set(close_order) != set(CABINET_LEVELS):
            raise ValueError("episode.close_order must contain each level exactly once")
    horizon = int(values.get("max_horizon", 15000))
    if horizon <= 0:
        raise ValueError("episode.max_horizon must be positive")
    material_candidates = normalize_cabinet_material_candidates(
        values.get("cabinet_material_candidates", tuple(CABINET_MATERIAL_TEXTURES))
    )
    return EpisodePlan(
        seed=int(seed),
        item_by_level=dict(assignment),
        level_by_item=level_by_item,
        target_item_id=target,
        target_level=level_by_item[target],
        close_order=close_order,
        max_horizon=horizon,
        cabinet_material_id=sample_cabinet_material(seed, material_candidates),
    )


def public_retrieval_prompt(target_item_id: str) -> str:
    """Return the policy-visible delayed request without leaking the cabinet level."""

    item_id = validate_item_id(target_item_id)
    color = "red" if item_id == "red_can" else "blue"
    return f"Both cabinet doors are closed. Retrieve the {color} can."


def _optional_int(value: Any) -> int | None:
    return None if value is None else int(value)


__all__ = [
    "close_order_for_seed",
    "public_retrieval_prompt",
    "resolve_episode_plan",
    "resolve_scene_config",
    "resolve_task_definition",
    "storage_assignment_for_seed",
    "target_item_for_seed",
    "validate_item_id",
    "validate_level",
]
