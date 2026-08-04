"""Deterministic TM03 route and frozen-scene resolution."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from bench.tasks.tm03.domain.contracts import (
    BURNER_ID,
    BURNER_SITE,
    COMPLETION_BUTTON_NAMES,
    DRAWER_NAMES,
    ENV_NAME,
    EXPECTED_TOASTER_OVEN_ASSET,
    KNOB_COLLISION_GEOM,
    KNOB_ID,
    KNOB_JOINT,
    TOASTER_OVEN_FOOD_NAME,
    TOASTER_OVEN_TIME_COLLISION_GEOM,
    TOASTER_OVEN_TIME_JOINT,
    PREFIX_ORDERS,
    TAIL_ORDERS,
    TASK_ID,
    EpisodePlan,
    PrefixOrder,
    SceneConfig,
    TailOrder,
    TaskDefinition,
)


DEFAULT_LAYOUT_ID = 39
DEFAULT_STYLE_ID = 24
DEFAULT_DRAWER_WORLD_XYZS = (
    (4.10, -2.37, 1.1363333333333332),
    (4.60, -2.37, 1.1363333333333332),
)
DEFAULT_COMPLETION_BUTTON_WORLD_XYZS = (
    (COMPLETION_BUTTON_NAMES[0], (3.55, -2.40, 0.92)),
    (COMPLETION_BUTTON_NAMES[1], (4.30, -3.30, 0.92)),
)
DEFAULT_TOASTER_OVEN_FOOD_XML = (
    "objects/objaverse/waffle/waffle_0/model.xml"
)


def sample_episode_plan(seed: int, *, max_horizon: int = 4800) -> EpisodePlan:
    """Cover 6 tails x 2 prefixes x 2 drawer orders every 24 seeds."""

    seed_value = int(seed)
    horizon = int(max_horizon)
    if horizon <= 0:
        raise ValueError("TM03 max_horizon must be positive")
    cell = seed_value % 24
    drawer_order = (
        DRAWER_NAMES if cell % 2 == 0 else tuple(reversed(DRAWER_NAMES))
    )
    prefix = PREFIX_ORDERS[(cell // 2) % len(PREFIX_ORDERS)]
    tail = TAIL_ORDERS[cell // 4]
    return EpisodePlan(
        seed=seed_value,
        prefix=prefix,
        tail=tail,
        drawer_order=(str(drawer_order[0]), str(drawer_order[1])),
        max_horizon=horizon,
    )


def resolve_episode_plan(
    *,
    seed: int,
    max_horizon: int = 4800,
    prefix: str | PrefixOrder | None = None,
    tail: str | TailOrder | None = None,
    drawer_order: Sequence[str] | str | None = None,
    **_: Any,
) -> EpisodePlan:
    sampled = sample_episode_plan(seed, max_horizon=max_horizon)
    return EpisodePlan(
        seed=int(seed),
        prefix=(
            sampled.prefix
            if prefix is None
            else prefix if isinstance(prefix, PrefixOrder) else PrefixOrder(str(prefix))
        ),
        tail=(
            sampled.tail
            if tail is None
            else tail if isinstance(tail, TailOrder) else TailOrder(str(tail))
        ),
        drawer_order=(
            sampled.drawer_order
            if drawer_order is None
            else normalize_drawer_order(drawer_order)
        ),
        max_horizon=int(max_horizon),
    )


def normalize_drawer_order(value: Sequence[str] | str) -> tuple[str, str]:
    if isinstance(value, str):
        items = tuple(item.strip() for item in value.split(",") if item.strip())
    else:
        items = tuple(str(item).strip() for item in value)
    aliases = {"left": "drawer_left", "right": "drawer_right"}
    result = tuple(aliases.get(item, item) for item in items)
    if len(result) != 2 or set(result) != set(DRAWER_NAMES):
        raise ValueError(f"TM03 drawer_order must permute {DRAWER_NAMES}")
    return (result[0], result[1])


def resolve_task_definition(**values: Any) -> TaskDefinition:
    task_id = str(values.get("id", TASK_ID))
    if task_id != TASK_ID:
        raise ValueError(f"TM03 config must use task.id={TASK_ID!r}")
    return TaskDefinition(
        task_id=task_id,
        display_name=str(
            values.get(
                "display_name",
                "Boil Water, Inspect Two Drawers, and Run Toaster Oven",
            )
        ),
        description=str(
            values.get(
                "description",
                "Interleave stove, drawer, and toaster-oven task threads.",
            )
        ),
        success_conditions=tuple(
            str(item) for item in values.get("success_conditions", ())
        ),
    )


def resolve_scene_config(*, env_split: str, **values: Any) -> SceneConfig:
    env_name = str(values.get("env_name", ENV_NAME))
    if env_name != ENV_NAME:
        raise ValueError(f"TM03 config must use scene.env_name={ENV_NAME!r}")
    layout_id = int(values.get("layout_id", DEFAULT_LAYOUT_ID))
    style_id = int(values.get("style_id", DEFAULT_STYLE_ID))
    if (layout_id, style_id) != (DEFAULT_LAYOUT_ID, DEFAULT_STYLE_ID):
        raise ValueError("TM03 is fixed to layout_id=39 and style_id=24")
    names = tuple(str(name) for name in values.get("drawer_names", DRAWER_NAMES))
    if names != DRAWER_NAMES:
        raise ValueError(f"TM03 drawer_names must equal {DRAWER_NAMES}")
    expected_toaster_oven = str(
        values.get("expected_toaster_oven_asset", EXPECTED_TOASTER_OVEN_ASSET)
    )
    if expected_toaster_oven != EXPECTED_TOASTER_OVEN_ASSET:
        raise ValueError(
            "TM03 requires expected_toaster_oven_asset="
            f"{EXPECTED_TOASTER_OVEN_ASSET!r}"
        )
    return SceneConfig(
        env_name=env_name,
        env_split=str(env_split),
        layout_id=layout_id,
        style_id=style_id,
        scene_seed=int(values.get("scene_seed", 0)),
        drawer_names=(names[0], names[1]),
        drawer_world_xyzs=_xyzs(
            values.get("drawer_world_xyzs", DEFAULT_DRAWER_WORLD_XYZS)
        ),
        drawer_world_yaw_deg=float(values.get("drawer_world_yaw_deg", 180.0)),
        completion_button_world_xyzs=_named_xyzs(
            values.get(
                "completion_button_world_xyzs",
                dict(DEFAULT_COMPLETION_BUTTON_WORLD_XYZS),
            ),
            expected_names=COMPLETION_BUTTON_NAMES,
        ),
        expected_stove_asset=str(values.get("expected_stove_asset", "Stovetop057")),
        knob_id=str(values.get("stove_knob_id", KNOB_ID)),
        burner_id=str(values.get("burner_id", BURNER_ID)),
        knob_joint=str(values.get("stove_knob_joint", KNOB_JOINT)),
        knob_collision_geom=str(
            values.get("stove_knob_collision_geom", KNOB_COLLISION_GEOM)
        ),
        burner_site=str(values.get("stove_burner_site", BURNER_SITE)),
        pot_name=str(values.get("pot_name", "pot")),
        pot_xml=str(
            values.get(
                "pot_xml", "objects/lightwheel/saucepan/Saucepan002/model.xml"
            )
        ),
        expected_toaster_oven_asset=expected_toaster_oven,
        toaster_oven_food_name=str(
            values.get("toaster_oven_food_name", TOASTER_OVEN_FOOD_NAME)
        ),
        toaster_oven_food_xml=str(
            values.get("toaster_oven_food_xml", DEFAULT_TOASTER_OVEN_FOOD_XML)
        ),
        toaster_oven_time_joint=str(
            values.get("toaster_oven_time_joint", TOASTER_OVEN_TIME_JOINT)
        ),
        toaster_oven_time_collision_geom=str(
            values.get(
                "toaster_oven_time_collision_geom",
                TOASTER_OVEN_TIME_COLLISION_GEOM,
            )
        ),
    )


def _xyz(value: Any) -> tuple[float, float, float]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError("TM03 xyz must contain three numbers")
    result = tuple(float(item) for item in value)
    if len(result) != 3 or not np.all(np.isfinite(result)):
        raise ValueError("TM03 xyz must contain three finite numbers")
    return (result[0], result[1], result[2])


def _xyzs(value: Any) -> tuple[tuple[float, float, float], ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError("TM03 drawer_world_xyzs must be a sequence")
    result = tuple(_xyz(item) for item in value)
    if len(result) != 2:
        raise ValueError("TM03 requires exactly two drawer xyz values")
    return result


def _named_xyzs(
    value: Any, *, expected_names: tuple[str, ...]
) -> tuple[tuple[str, tuple[float, float, float]], ...]:
    if not isinstance(value, Mapping) or tuple(value) != expected_names:
        raise ValueError(
            f"TM03 completion button names must equal {expected_names}"
        )
    return tuple((name, _xyz(value[name])) for name in expected_names)


__all__ = [
    "DEFAULT_LAYOUT_ID",
    "DEFAULT_TOASTER_OVEN_FOOD_XML",
    "DEFAULT_STYLE_ID",
    "normalize_drawer_order",
    "resolve_episode_plan",
    "resolve_scene_config",
    "resolve_task_definition",
    "sample_episode_plan",
]
