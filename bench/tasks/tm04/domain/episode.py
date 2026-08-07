"""Deterministic TM04 route sampling and frozen-scene resolution."""

from __future__ import annotations

import random
from collections.abc import Sequence
from typing import Any

import numpy as np

from .contracts import (
    CAN_NAME,
    ENV_NAME,
    HOT_CHOCOLATE_NAME,
    PREFIX_ORDERS,
    SUGAR_NAMES,
    SUGAR_PLATE_NAME,
    TARGET_SUGAR_COUNT,
    TASK_ID,
    TAIL_ORDERS,
    TOTAL_SUGAR_COUNT,
    WATER_POT_NAME,
    EpisodePlan,
    PrefixOrder,
    SceneConfig,
    TailOrder,
    TaskDefinition,
)


DEFAULT_LAYOUT_ID = 15
DEFAULT_STYLE_ID = 34
DEFAULT_WATER_BUTTON_WORLD_XYZ = (2.06, -2.56, 0.93)
DEFAULT_COMPLETION_BUTTON_WORLD_XYZ = (2.06, -2.42, 0.93)
DEFAULT_CABINET_WORLD_XYZ = (0.32, -3.93, 1.12)
DEFAULT_CABINET_YAW_DEG = 90.0
DEFAULT_CAN_PLACE_REGION_WORLD_XYZ = (2.72, -1.95, 0.925)
DEFAULT_CAN_PLACE_REGION_SIZE_XY = (0.22, 0.20)
SUGAR_ORDER_SALT = 0x544D3034


def sample_episode_plan(seed: int, *, max_horizon: int = 5200) -> EpisodePlan:
    """Cover two prefixes by six tails exactly once every twelve seeds."""

    seed_value = int(seed)
    horizon = int(max_horizon)
    if seed_value < 0:
        raise ValueError("TM04 seed must be non-negative")
    if horizon <= 0:
        raise ValueError("TM04 max_horizon must be positive")
    cell = seed_value % 12
    prefix = PREFIX_ORDERS[cell % len(PREFIX_ORDERS)]
    tail = TAIL_ORDERS[cell // len(PREFIX_ORDERS)]
    sugar_order = list(SUGAR_NAMES)
    random.Random(seed_value ^ SUGAR_ORDER_SALT).shuffle(sugar_order)
    selected = tuple(sugar_order[:TARGET_SUGAR_COUNT])
    distractors = tuple(name for name in SUGAR_NAMES if name not in selected)
    return EpisodePlan(
        seed=seed_value,
        prefix=prefix,
        tail=tail,
        target_sugar_count=TARGET_SUGAR_COUNT,
        total_sugar_count=TOTAL_SUGAR_COUNT,
        sugar_names=SUGAR_NAMES,
        selected_sugar_names=selected,
        distractor_sugar_names=distractors,
        max_horizon=horizon,
    )


def resolve_episode_plan(
    *,
    seed: int,
    max_horizon: int = 5200,
    prefix: str | PrefixOrder | None = None,
    tail: str | TailOrder | None = None,
    target_sugar_count: int = TARGET_SUGAR_COUNT,
    total_sugar_count: int = TOTAL_SUGAR_COUNT,
    **_: Any,
) -> EpisodePlan:
    if (int(target_sugar_count), int(total_sugar_count)) != (
        TARGET_SUGAR_COUNT,
        TOTAL_SUGAR_COUNT,
    ):
        raise ValueError("TM04 is frozen to K=2, N=3")
    sampled = sample_episode_plan(seed, max_horizon=max_horizon)
    return EpisodePlan(
        seed=sampled.seed,
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
        target_sugar_count=sampled.target_sugar_count,
        total_sugar_count=sampled.total_sugar_count,
        sugar_names=sampled.sugar_names,
        selected_sugar_names=sampled.selected_sugar_names,
        distractor_sugar_names=sampled.distractor_sugar_names,
        max_horizon=sampled.max_horizon,
    )


def resolve_task_definition(**values: Any) -> TaskDefinition:
    task_id = str(values.get("id", TASK_ID))
    if task_id != TASK_ID:
        raise ValueError(f"TM04 config must use task.id={TASK_ID!r}")
    return TaskDefinition(
        task_id=task_id,
        display_name=str(
            values.get("display_name", "Fill Pot, Retrieve Can, and Add Two Sugars")
        ),
        description=str(
            values.get(
                "description",
                "Interleave pot filling, can retrieval, and exactly-two-of-three sugar counting.",
            )
        ),
        success_conditions=tuple(str(item) for item in values.get("success_conditions", ())),
    )


def resolve_scene_config(*, env_split: str, **values: Any) -> SceneConfig:
    env_name = str(values.get("env_name", ENV_NAME))
    if env_name != ENV_NAME:
        raise ValueError(f"TM04 config must use scene.env_name={ENV_NAME!r}")
    layout_id = int(values.get("layout_id", DEFAULT_LAYOUT_ID))
    style_id = int(values.get("style_id", DEFAULT_STYLE_ID))
    if (layout_id, style_id) != (DEFAULT_LAYOUT_ID, DEFAULT_STYLE_ID):
        raise ValueError("TM04 is fixed to TM02 layout 15 / style 34")
    names = tuple(str(name) for name in values.get("sugar_names", SUGAR_NAMES))
    if names != SUGAR_NAMES:
        raise ValueError(f"TM04 sugar_names must equal {SUGAR_NAMES}")
    return SceneConfig(
        env_name=env_name,
        env_split=str(env_split),
        layout_id=layout_id,
        style_id=style_id,
        scene_seed=int(values.get("scene_seed", 0)),
        water_button_world_xyz=_xyz(values.get("water_button_world_xyz", DEFAULT_WATER_BUTTON_WORLD_XYZ)),
        completion_button_world_xyz=_xyz(values.get("completion_button_world_xyz", DEFAULT_COMPLETION_BUTTON_WORLD_XYZ)),
        cabinet_world_xyz=_xyz(values.get("cabinet_world_xyz", DEFAULT_CABINET_WORLD_XYZ)),
        cabinet_yaw_deg=_finite(values.get("cabinet_yaw_deg", DEFAULT_CABINET_YAW_DEG), "cabinet_yaw_deg"),
        can_place_region_world_xyz=_xyz(values.get("can_place_region_world_xyz", DEFAULT_CAN_PLACE_REGION_WORLD_XYZ)),
        can_place_region_size_xy=_xy(values.get("can_place_region_size_xy", DEFAULT_CAN_PLACE_REGION_SIZE_XY)),
        water_pot_name=str(values.get("water_pot_name", WATER_POT_NAME)),
        can_name=str(values.get("can_name", CAN_NAME)),
        sugar_plate_name=str(values.get("sugar_plate_name", SUGAR_PLATE_NAME)),
        hot_chocolate_name=str(values.get("hot_chocolate_name", HOT_CHOCOLATE_NAME)),
        sugar_names=names,
        sugar_plate_world_xyz=_optional_xyz(values.get("sugar_plate_world_xyz")),
        hot_chocolate_world_xyz=_optional_xyz(values.get("hot_chocolate_world_xyz")),
    )


def _finite(value: Any, label: str) -> float:
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"TM04 {label} must be finite")
    return result


def _xyz(value: Any) -> tuple[float, float, float]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError("TM04 xyz must contain three numbers")
    result = tuple(float(item) for item in value)
    if len(result) != 3 or not np.all(np.isfinite(result)):
        raise ValueError("TM04 xyz must contain three finite numbers")
    return (result[0], result[1], result[2])


def _optional_xyz(value: Any) -> tuple[float, float, float] | None:
    return None if value is None else _xyz(value)


def _xy(value: Any) -> tuple[float, float]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError("TM04 xy must contain two numbers")
    result = tuple(float(item) for item in value)
    if len(result) != 2 or not np.all(np.isfinite(result)) or any(item <= 0 for item in result):
        raise ValueError("TM04 xy size must contain two positive finite numbers")
    return (result[0], result[1])


__all__ = [
    "DEFAULT_LAYOUT_ID", "DEFAULT_STYLE_ID", "resolve_episode_plan",
    "resolve_scene_config", "resolve_task_definition", "sample_episode_plan",
]
