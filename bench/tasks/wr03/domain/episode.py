"""Pure, deterministic episode decisions for WR-03."""

from __future__ import annotations

from typing import Any

from bench.tasks.wr03.domain.contracts import SceneConfig, TaskDefinition
from bench.tasks.wr03.domain.contracts import (
    CABINET_LEVELS,
    DISPLAY_NAME,
    ENV_NAME,
    EpisodePlan,
    TASK_ID,
)


def validate_level(value: Any) -> str:
    level = str(value)
    if level not in CABINET_LEVELS:
        raise ValueError(f"invalid WR-03 cabinet level: {level!r}")
    return level


def demo_storage_level_for_seed(seed: int) -> str:
    episode_seed = int(seed)
    if episode_seed < 0:
        raise ValueError("WR-03 seed must be non-negative")
    return "lower" if episode_seed % 2 == 0 else "upper"


def resolve_task_definition(**values: Any) -> TaskDefinition:
    if str(values.get("id", TASK_ID)) != TASK_ID:
        raise ValueError(f"WR-03 config must use task.id={TASK_ID!r}")
    conditions = tuple(str(item) for item in values.get("success_conditions", ()))
    if not conditions:
        raise ValueError("WR-03 must declare final success conditions")
    return TaskDefinition(
        task_id=TASK_ID,
        display_name=str(values.get("display_name", DISPLAY_NAME)),
        description=str(
            values.get(
                "description",
                "Store one can in either cabinet, close both doors, then retrieve "
                "the can from memory.",
            )
        ),
        success_conditions=conditions,
    )


def resolve_scene_config(*, seed: int, env_split: str, **values: Any) -> SceneConfig:
    del seed
    if str(values.get("env_name", ENV_NAME)) != ENV_NAME:
        raise ValueError(f"WR-03 config must use scene.env_name={ENV_NAME!r}")
    if not bool(values.get("fixed_scene", True)):
        raise ValueError("WR-03 requires fixed_scene=true")
    layout_id = int(values.get("layout_id", 24))
    style_id = int(values.get("style_id", 24))
    scene_seed = int(values.get("scene_seed", 0))
    if (layout_id, style_id, scene_seed) != (24, 24, 0):
        raise ValueError("WR-03 is fixed to layout/style/scene-seed 24/24/0")
    return SceneConfig(
        env_name=ENV_NAME,
        env_split=str(env_split),
        layout_id=layout_id,
        style_id=style_id,
        scene_seed=scene_seed,
        fixed_scene=True,
    )


def resolve_episode_plan(*, seed: int, **values: Any) -> EpisodePlan:
    episode_seed = int(seed)
    if episode_seed < 0:
        raise ValueError("WR-03 seed must be non-negative")
    configured = values.get("demo_storage_level")
    level = (
        demo_storage_level_for_seed(episode_seed)
        if configured is None
        else validate_level(configured)
    )
    horizon = int(values.get("max_horizon", 15000))
    if horizon <= 0:
        raise ValueError("WR-03 max_horizon must be positive")
    return EpisodePlan(
        seed=episode_seed,
        demo_storage_level=level,
        placement_pair_seed=episode_seed // 2,
        max_horizon=horizon,
    )


def public_memory_prompt() -> str:
    """Describe the delayed request without exposing the stored level."""

    return (
        "Place the can into one of the two cabinets. Close both cabinet doors, "
        "remember where the can was stored, then retrieve it."
    )


__all__ = [
    "demo_storage_level_for_seed",
    "public_memory_prompt",
    "resolve_episode_plan",
    "resolve_scene_config",
    "resolve_task_definition",
    "validate_level",
]
