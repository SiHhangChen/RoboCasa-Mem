"""Pure, deterministic episode decisions for WR-03."""

from __future__ import annotations

from typing import Any

from bench.tasks.wr03.cabinet_materials import (
    CABINET_MATERIAL_TEXTURES,
    normalize_cabinet_material_candidates,
    sample_cabinet_material,
)

from bench.tasks.wr03.domain.contracts import SceneConfig, TaskDefinition
from bench.tasks.wr03.domain.contracts import (
    CABINET_LEVELS,
    DISPLAY_NAME,
    ENV_NAME,
    EpisodePlan,
    TASK_ID,
)

_CLOSE_ORDER_SALT = 0x57523343
_CLOSE_ORDER_PERIOD = 200
_STORAGE_LEVEL_SALT = 51
_STORAGE_LEVEL_MULTIPLIER = 49


def validate_level(value: Any) -> str:
    level = str(value)
    if level not in CABINET_LEVELS:
        raise ValueError(f"invalid WR-03 cabinet level: {level!r}")
    return level


def demo_storage_level_for_seed(seed: int) -> str:
    episode_seed = int(seed)
    if episode_seed < 0:
        raise ValueError("WR-03 seed must be non-negative")
    # Use an independent affine permutation rather than seed parity. It remains
    # exactly balanced over every 200-seed release window without exposing the
    # cabinet choice through an obvious even/odd rule.
    bucket = (
        episode_seed * _STORAGE_LEVEL_MULTIPLIER + _STORAGE_LEVEL_SALT
    ) % _CLOSE_ORDER_PERIOD
    return "lower" if bucket < _CLOSE_ORDER_PERIOD // 2 else "upper"


def close_order_for_seed(seed: int) -> tuple[str, str]:
    # The close order uses a stream independent of the stored level and is
    # exactly balanced over every contiguous 200-seed release window.
    bucket = (int(seed) * 73 + _CLOSE_ORDER_SALT) % _CLOSE_ORDER_PERIOD
    return ("lower", "upper") if bucket < _CLOSE_ORDER_PERIOD // 2 else ("upper", "lower")


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
                "Observe a pre-filled open cabinet, close both doors, reset to "
                "neutral, then retrieve the can from memory.",
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
    configured = values.get("stored_level", values.get("demo_storage_level"))
    level = (
        demo_storage_level_for_seed(episode_seed)
        if configured is None
        else validate_level(configured)
    )
    configured_order = values.get("close_order")
    if configured_order is None:
        close_order = close_order_for_seed(episode_seed)
    else:
        if isinstance(configured_order, (str, bytes)) or len(configured_order) != 2:
            raise ValueError("episode.close_order must contain lower and upper")
        close_order = tuple(validate_level(v) for v in configured_order)
        if set(close_order) != set(CABINET_LEVELS):
            raise ValueError("episode.close_order must contain each level exactly once")
    horizon = int(values.get("max_horizon", 8000))
    if horizon <= 0:
        raise ValueError("WR-03 max_horizon must be positive")
    material_candidates = normalize_cabinet_material_candidates(
        values.get("cabinet_material_candidates", tuple(CABINET_MATERIAL_TEXTURES))
    )
    return EpisodePlan(
        seed=episode_seed,
        stored_level=level,
        close_order=close_order,
        max_horizon=horizon,
        cabinet_material_id=sample_cabinet_material(episode_seed, material_candidates),
    )


def public_memory_prompt() -> str:
    """Describe the delayed request without exposing the stored level."""

    return (
        "Retrieve the can. Observe which open cabinet contains it, close both cabinet "
        "doors in any order, return to the neutral position, then open the remembered "
        "cabinet and bring the can back to the table."
    )


__all__ = [
    "close_order_for_seed",
    "demo_storage_level_for_seed",
    "public_memory_prompt",
    "resolve_episode_plan",
    "resolve_scene_config",
    "resolve_task_definition",
    "validate_level",
]
