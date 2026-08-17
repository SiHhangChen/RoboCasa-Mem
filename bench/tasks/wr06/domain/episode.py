"""Pure deterministic episode decisions for WR-06.

The bottle's original shelf level and its staged plate side are seed-decided:
even seeds place it on the lower shelf and odd seeds on the upper shelf, while
the plate staging side alternates with the placement-pair seed (even pair ->
plate_left, odd pair -> plate_right).  Together the four (level, side) cases
are balanced and cover every WR-05 pairing once the mustard is removed.  The
scene is fixed to layout/style/scene-seed 32/32/0, and placement randomness is
owned by ``placement_pair_seed`` so the lower/upper cases draw identical
placements.
"""

from __future__ import annotations

from typing import Any

from bench.tasks.wr06.domain.contracts import (
    DISPLAY_NAME,
    ENV_NAME,
    HOME_LEVELS,
    PLATE_SIDES,
    EpisodePlan,
    SceneConfig,
    TASK_ID,
    TaskDefinition,
)


def validate_level(value: Any) -> str:
    level = str(value)
    if level not in HOME_LEVELS:
        raise ValueError(f"invalid WR-06 shelf level: {level!r}")
    return level


def validate_plate_stage_region(value: Any) -> str:
    region = str(value)
    if region not in PLATE_SIDES:
        raise ValueError(f"invalid WR-06 plate stage region: {region!r}")
    return region


def demo_original_level_for_seed(seed: int) -> str:
    episode_seed = int(seed)
    if episode_seed < 0:
        raise ValueError("WR-06 seed must be non-negative")
    return "shelf_lower" if episode_seed % 2 == 0 else "shelf_upper"


def plate_stage_region_for_seed(seed: int) -> str:
    """Alternate the staged plate side with the placement-pair seed."""
    episode_seed = int(seed)
    if episode_seed < 0:
        raise ValueError("WR-06 seed must be non-negative")
    return "plate_left" if (episode_seed // 2) % 2 == 0 else "plate_right"


def resolve_task_definition(**values: Any) -> TaskDefinition:
    if str(values.get("id", TASK_ID)) != TASK_ID:
        raise ValueError(f"WR-06 config must use task.id={TASK_ID!r}")
    conditions = tuple(str(value) for value in values.get("success_conditions", ()))
    if not conditions:
        raise ValueError("WR-06 must declare final success conditions")
    return TaskDefinition(
        task_id=TASK_ID,
        display_name=str(values.get("display_name", DISPLAY_NAME)),
        description=str(
            values.get(
                "description",
                "Stage one ketchup bottle beside the fixed serving plate, then "
                "return it to its remembered original shelf level.",
            )
        ),
        success_conditions=conditions,
    )


def resolve_scene_config(*, seed: int, env_split: str, **values: Any) -> SceneConfig:
    del seed
    if str(values.get("env_name", ENV_NAME)) != ENV_NAME:
        raise ValueError(f"WR-06 config must use scene.env_name={ENV_NAME!r}")
    if not bool(values.get("fixed_scene", True)):
        raise ValueError("WR-06 requires fixed_scene=true")
    layout_id = int(values.get("layout_id", 32))
    style_id = int(values.get("style_id", 32))
    scene_seed = int(values.get("scene_seed", 0))
    if (layout_id, style_id, scene_seed) != (32, 32, 0):
        raise ValueError("WR-06 is fixed to layout/style/scene-seed 32/32/0")
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
        raise ValueError("WR-06 seed must be non-negative")
    configured = values.get("demo_original_level")
    level = (
        demo_original_level_for_seed(episode_seed)
        if configured is None
        else validate_level(configured)
    )
    configured_side = values.get("demo_plate_stage_region")
    plate_side = (
        plate_stage_region_for_seed(episode_seed)
        if configured_side is None
        else validate_plate_stage_region(configured_side)
    )
    horizon = int(values.get("max_horizon", 3200))
    if horizon <= 0:
        raise ValueError("WR-06 max_horizon must be positive")
    return EpisodePlan(
        seed=episode_seed,
        demo_original_level=level,
        plate_stage_region=plate_side,
        placement_pair_seed=episode_seed // 2,
        max_horizon=horizon,
    )


def public_initial_prompt() -> str:
    return (
        "Remember whether the ketchup bottle starts on the lower or upper shelf. "
        "Move it to the plate staging spot beside the serving plate, then return "
        "it to its original shelf level."
    )


def public_return_prompt() -> str:
    return "Return the ketchup bottle to its original shelf level."


__all__ = [
    "demo_original_level_for_seed",
    "plate_stage_region_for_seed",
    "public_initial_prompt",
    "public_return_prompt",
    "resolve_episode_plan",
    "resolve_scene_config",
    "resolve_task_definition",
    "validate_level",
    "validate_plate_stage_region",
]
