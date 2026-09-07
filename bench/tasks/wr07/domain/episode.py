"""Deterministic WR-07 color and placement sampling for single transport."""

from __future__ import annotations

import random
from math import cos, isfinite, pi, sin, sqrt
from typing import Any

from bench.tasks.wr07.domain.contracts import (
    COLOR_NAMES, DISPLAY_NAME, ENV_NAME, EpisodePlan, ITEM_ID, ITEM_IDS,
    SOURCE_SLOTS, TARGET_SLOTS, TASK_ID, TaskDefinition, SceneConfig,
)

def validate_source_slot(value: Any) -> str:
    slot = str(value)
    if slot not in SOURCE_SLOTS:
        raise ValueError(f"invalid WR-07 source slot: {slot!r}")
    return slot

def validate_target_slot(value: Any) -> str:
    slot = str(value)
    if slot not in TARGET_SLOTS:
        raise ValueError(f"invalid WR-07 target slot: {slot!r}")
    return slot

def validate_color(value: Any) -> str:
    color = str(value)
    if color not in COLOR_NAMES:
        raise ValueError(f"invalid WR-07 color: {color!r}")
    return color

def resolve_task_definition(**values: Any) -> TaskDefinition:
    if str(values.get("id", TASK_ID)) != TASK_ID:
        raise ValueError(f"WR-07 config must use task.id={TASK_ID!r}")
    conditions = tuple(str(value) for value in values.get("success_conditions", ()))
    if not conditions:
        raise ValueError("WR-07 must declare success conditions")
    return TaskDefinition(
        TASK_ID,
        str(values.get("display_name", DISPLAY_NAME)),
        str(values.get("description", "Move the milk carton to the sugar plate with the same color.")),
        conditions,
    )

def resolve_scene_config(*, seed: int, env_split: str, **values: Any) -> SceneConfig:
    del seed
    if str(values.get("env_name", ENV_NAME)) != ENV_NAME:
        raise ValueError(f"WR-07 config must use scene.env_name={ENV_NAME!r}")
    identity = (int(values.get("layout_id", 32)), int(values.get("style_id", 32)), int(values.get("scene_seed", 0)))
    if not bool(values.get("fixed_scene", True)) or identity != (32, 32, 0):
        raise ValueError("WR-07 is fixed to layout/style/scene-seed 32/32/0")
    return SceneConfig(ENV_NAME, str(env_split), *identity, fixed_scene=True)

def resolve_episode_plan(*, seed: int, **values: Any) -> EpisodePlan:
    episode_seed = int(seed)
    if episode_seed < 0:
        raise ValueError("WR-07 seed must be non-negative")
    if int(values.pop("color_count", 3)) != 3:
        raise ValueError("WR-07 requires exactly three visible colors")
    item_radius = _nonnegative_scalar(values.pop("item_sample_radius_m", 0.045), label="item_sample_radius_m")
    horizon = int(values.pop("max_horizon", 3600))
    values.pop("color_pool", None)
    if values:
        raise ValueError("unknown WR-07 episode settings: " + ", ".join(sorted(values)))
    rng = random.Random(episode_seed)
    selected = rng.sample(list(COLOR_NAMES), 3)
    source_colors = list(selected)
    rng.shuffle(source_colors)
    source_by_slot = dict(zip(SOURCE_SLOTS, source_colors, strict=True))
    shift = rng.choice((1, 2))
    target_colors = [source_colors[(index + shift) % 3] for index in range(3)]
    target_by_slot = dict(zip(TARGET_SLOTS, target_colors, strict=True))
    target_by_color = {color: slot for slot, color in target_by_slot.items()}
    source_slot = rng.choice(SOURCE_SLOTS)
    source_color = source_by_slot[source_slot]
    return EpisodePlan(
        seed=episode_seed,
        source_plate_color_by_slot=source_by_slot,
        target_plate_color_by_slot=target_by_slot,
        source_slot_by_item={ITEM_ID: source_slot},
        source_color_by_item={ITEM_ID: source_color},
        target_slot_by_color=target_by_color,
        source_item_offset_xy_by_item={ITEM_ID: _sample_disk(rng, item_radius)},
        source_item_yaw_by_item={ITEM_ID: rng.uniform(-pi, pi)},
        target_item_offset_xy_by_item={ITEM_ID: _sample_disk(rng, item_radius)},
        target_item_yaw_by_item={ITEM_ID: rng.uniform(-pi, pi)},
        item_sample_radius_m=item_radius,
        transport_order=ITEM_IDS,
        max_horizon=horizon,
    )

def _sample_disk(rng: random.Random, radius: float) -> tuple[float, float]:
    if radius <= 0.0:
        return (0.0, 0.0)
    angle = rng.uniform(-pi, pi)
    distance = radius * sqrt(rng.random())
    return (float(distance * cos(angle)), float(distance * sin(angle)))

def _nonnegative_scalar(value: Any, *, label: str) -> float:
    try:
        resolved = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"WR-07 {label} must be finite and non-negative") from exc
    if not isfinite(resolved) or resolved < 0.0:
        raise ValueError(f"WR-07 {label} must be finite and non-negative")
    return resolved

def public_task_prompt() -> str:
    return (
        "Remember the color of the sugar plate holding the milk carton. "
        "Move the milk carton from the coffee-machine-side sugar plate to "
        "the matching colored sugar plate on the other side."
    )

def public_initial_prompt() -> str:
    return public_task_prompt()

def public_return_prompt() -> str:
    return public_task_prompt()

__all__ = [
    "public_initial_prompt", "public_return_prompt", "public_task_prompt",
    "resolve_episode_plan", "resolve_scene_config", "resolve_task_definition",
    "validate_color", "validate_source_slot", "validate_target_slot",
]
