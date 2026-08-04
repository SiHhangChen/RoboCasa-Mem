"""Pure deterministic episode decisions for WR-04."""

from __future__ import annotations

from typing import Any

from bench.tasks.wr04.domain.contracts import (
    DISPLAY_NAME,
    ENV_NAME,
    HOME_LEVELS,
    EpisodePlan,
    SceneConfig,
    TASK_ID,
    TaskDefinition,
)


def validate_level(value: Any) -> str:
    level = str(value)
    if level not in HOME_LEVELS:
        raise ValueError(f"invalid WR-04 shelf level: {level!r}")
    return level


def demo_original_level_for_seed(seed: int) -> str:
    episode_seed = int(seed)
    if episode_seed < 0:
        raise ValueError("WR-04 seed must be non-negative")
    return "shelf_lower" if episode_seed % 2 == 0 else "shelf_upper"


def resolve_task_definition(**values: Any) -> TaskDefinition:
    if str(values.get("id", TASK_ID)) != TASK_ID:
        raise ValueError(f"WR-04 config must use task.id={TASK_ID!r}")
    conditions = tuple(str(value) for value in values.get("success_conditions", ()))
    if not conditions:
        raise ValueError("WR-04 must declare final success conditions")
    return TaskDefinition(
        task_id=TASK_ID,
        display_name=str(values.get("display_name", DISPLAY_NAME)),
        description=str(
            values.get(
                "description",
                "Stage one sponge beside the sink, then return it to its original "
                "shelf level from memory.",
            )
        ),
        success_conditions=conditions,
    )


def resolve_scene_config(*, seed: int, env_split: str, **values: Any) -> SceneConfig:
    del seed
    if str(values.get("env_name", ENV_NAME)) != ENV_NAME:
        raise ValueError(f"WR-04 config must use scene.env_name={ENV_NAME!r}")
    if not bool(values.get("fixed_scene", True)):
        raise ValueError("WR-04 requires fixed_scene=true")
    layout_id = int(values.get("layout_id", 32))
    style_id = int(values.get("style_id", 32))
    scene_seed = int(values.get("scene_seed", 0))
    if (layout_id, style_id, scene_seed) != (32, 32, 0):
        raise ValueError("WR-04 is fixed to layout/style/scene-seed 32/32/0")
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
        raise ValueError("WR-04 seed must be non-negative")
    configured = values.get("demo_original_level")
    level = (
        demo_original_level_for_seed(episode_seed)
        if configured is None
        else validate_level(configured)
    )
    horizon = int(values.get("max_horizon", 3200))
    if horizon <= 0:
        raise ValueError("WR-04 max_horizon must be positive")
    return EpisodePlan(
        seed=episode_seed,
        demo_original_level=level,
        placement_pair_seed=episode_seed // 2,
        max_horizon=horizon,
    )


def public_initial_prompt() -> str:
    return (
        "Remember whether the sponge starts on the lower or upper shelf. Move it "
        "to the fixed sink staging spot, then return it to its original shelf level."
    )


def public_return_prompt() -> str:
    return "Return the sponge to its original shelf level."


__all__ = [
    "demo_original_level_for_seed",
    "public_initial_prompt",
    "public_return_prompt",
    "resolve_episode_plan",
    "resolve_scene_config",
    "resolve_task_definition",
    "validate_level",
]
