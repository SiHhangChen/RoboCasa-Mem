"""Pure, reproducible episode decisions for WR-01."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

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
                "Store red and blue cans in separate cabinets, then retrieve the "
                "requested can by memory.",
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
    horizon = int(values.get("max_horizon", 5000))
    if horizon <= 0:
        raise ValueError("episode.max_horizon must be positive")
    return EpisodePlan(
        seed=int(seed),
        item_by_level=dict(assignment),
        level_by_item=level_by_item,
        target_item_id=target,
        target_level=level_by_item[target],
        storage_order=tuple(assignment[level] for level in CABINET_LEVELS),
        max_horizon=horizon,
    )


def public_retrieval_prompt(target_item_id: str) -> str:
    """Return the policy-visible delayed request without leaking the cabinet level."""

    item_id = validate_item_id(target_item_id)
    color = "red" if item_id == "red_can" else "blue"
    return f"Both cabinet doors are closed. Retrieve the {color} can."


def _optional_int(value: Any) -> int | None:
    return None if value is None else int(value)


__all__ = [
    "public_retrieval_prompt",
    "resolve_episode_plan",
    "resolve_scene_config",
    "resolve_task_definition",
    "storage_assignment_for_seed",
    "target_item_for_seed",
    "validate_item_id",
    "validate_level",
]
