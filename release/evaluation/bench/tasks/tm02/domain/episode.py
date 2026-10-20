"""Deterministic TM02 episode planning and frozen scene resolution."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from bench.tasks.tm02.cabinet_materials import (
    CABINET_MATERIAL_TEXTURES,
    normalize_cabinet_material_candidates,
    sample_cabinet_material,
)

from bench.tasks.tm02.domain.contracts import (
    CAN_NAME,
    ENV_NAME,
    POT_NAME,
    TASK_ID,
    EpisodePlan,
    Route,
    SceneConfig,
    TaskDefinition,
)


DEFAULT_LAYOUT_ID = 15
DEFAULT_STYLE_ID = 34
DEFAULT_WATER_BUTTON_WORLD_XYZ = (2.06, -2.56, 0.93)
# The nominal button station faces +X, so +Y is the robot's left. Keep 0.14 m
# center spacing so the two 0.12 m-wide button bases do not overlap.
DEFAULT_COMPLETION_BUTTON_WORLD_XYZ = (2.06, -2.42, 0.93)
# When facing layout 15's wall-mounted stove, the WR01 cabinet occupies the
# free counter patch on its left. At 90 degrees its door faces the center aisle.
DEFAULT_CABINET_WORLD_XYZ = (0.32, -3.93, 1.12)
DEFAULT_CABINET_YAW_DEG = 90.0
# The marked release region occupies the free counter patch right of the stove.
# The user-calibrated place station at (0.880, -2.518) faces this target.
DEFAULT_CAN_PLACE_REGION_WORLD_XYZ = (0.30, -2.42, 0.925)
DEFAULT_CAN_PLACE_REGION_SIZE_XY = (0.22, 0.20)


def sample_episode_plan(seed: int, *, max_horizon: int = 3600) -> EpisodePlan:
    """Alternate L1/L2 demonstrations without changing the success target."""

    horizon = int(max_horizon)
    if horizon <= 0:
        raise ValueError("TM02 max_horizon must be positive")
    seed_value = int(seed)
    return EpisodePlan(
        seed=seed_value,
        route=Route.L1 if seed_value % 2 == 0 else Route.L2,
        max_horizon=horizon,
    )


def resolve_episode_plan(
    *,
    seed: int,
    max_horizon: int = 3600,
    route: str | Route | None = None,
    **values: Any,
) -> EpisodePlan:
    sampled = sample_episode_plan(seed, max_horizon=max_horizon)
    resolved_route = (
        sampled.route
        if route is None
        else route
        if isinstance(route, Route)
        else Route(str(route))
    )
    candidates = normalize_cabinet_material_candidates(
        values.get("cabinet_material_candidates", tuple(CABINET_MATERIAL_TEXTURES))
    )
    return EpisodePlan(
        seed=int(seed),
        route=resolved_route,
        max_horizon=int(max_horizon),
        cabinet_material_id=sample_cabinet_material(seed, candidates),
    )


def resolve_task_definition(**values: Any) -> TaskDefinition:
    task_id = str(values.get("id", TASK_ID))
    if task_id != TASK_ID:
        raise ValueError(f"TM02 config must use task.id={TASK_ID!r}")
    return TaskDefinition(
        task_id=task_id,
        display_name=str(values.get("display_name", "Fill Pot and Retrieve Can")),
        description=str(
            values.get(
                "description",
                "Fill a sink-side pot while retrieving a can, then finish both threads.",
            )
        ),
        success_conditions=tuple(
            str(item)
            for item in values.get(
                "success_conditions",
                (
                    "legal_event_route",
                    "faucet_on_once",
                    "faucet_off_once",
                    "faucet_off",
                    "pot_filled_and_stable_in_sink",
                    "can_retrieved_and_released_in_place_region",
                    "cabinet_closed",
                    "completion_button_pressed",
                    "no_absorbing_failure",
                    "fixed_prompt",
                ),
            )
        ),
    )


def resolve_scene_config(*, env_split: str, **values: Any) -> SceneConfig:
    env_name = str(values.get("env_name", ENV_NAME))
    if env_name != ENV_NAME:
        raise ValueError(f"TM02 config must use scene.env_name={ENV_NAME!r}")
    layout_id = int(values.get("layout_id", DEFAULT_LAYOUT_ID))
    style_id = int(values.get("style_id", DEFAULT_STYLE_ID))
    if (layout_id, style_id) != (DEFAULT_LAYOUT_ID, DEFAULT_STYLE_ID):
        raise ValueError("TM02 is fixed to WA02 layout_id=15 and style_id=34")
    can_scale = float(values.get("can_scale", 0.90))
    if not np.isfinite(can_scale) or can_scale <= 0.0:
        raise ValueError("TM02 can_scale must be positive and finite")
    return SceneConfig(
        env_name=env_name,
        env_split=str(env_split),
        layout_id=layout_id,
        style_id=style_id,
        scene_seed=int(values.get("scene_seed", 0)),
        water_button_world_xyz=_xyz(
            values.get("water_button_world_xyz", DEFAULT_WATER_BUTTON_WORLD_XYZ)
        ),
        completion_button_world_xyz=_xyz(
            values.get(
                "completion_button_world_xyz",
                DEFAULT_COMPLETION_BUTTON_WORLD_XYZ,
            )
        ),
        cabinet_world_xyz=_xyz(
            values.get("cabinet_world_xyz", DEFAULT_CABINET_WORLD_XYZ)
        ),
        cabinet_yaw_deg=_finite(
            values.get("cabinet_yaw_deg", DEFAULT_CABINET_YAW_DEG),
            "cabinet_yaw_deg",
        ),
        can_place_region_world_xyz=_xyz(
            values.get(
                "can_place_region_world_xyz",
                DEFAULT_CAN_PLACE_REGION_WORLD_XYZ,
            )
        ),
        can_place_region_size_xy=_xy(
            values.get(
                "can_place_region_size_xy",
                DEFAULT_CAN_PLACE_REGION_SIZE_XY,
            ),
            positive=True,
        ),
        pot_name=str(values.get("pot_name", POT_NAME)),
        pot_xml=str(
            values.get(
                "pot_xml",
                "objects/lightwheel/saucepan/Saucepan002/model.xml",
            )
        ),
        can_name=str(values.get("can_name", CAN_NAME)),
        can_xml=str(
            values.get(
                "can_xml",
                "objects/objaverse/canned_food/canned_food_15/model.xml",
            )
        ),
        can_scale=can_scale,
    )


def _finite(value: Any, label: str) -> float:
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"TM02 {label} must be finite")
    return result


def _xyz(value: Any) -> tuple[float, float, float]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError("TM02 xyz value must contain three numbers")
    result = tuple(float(item) for item in value)
    if len(result) != 3 or not np.all(np.isfinite(result)):
        raise ValueError("TM02 xyz value must contain three finite numbers")
    return (result[0], result[1], result[2])


def _xy(value: Any, *, positive: bool = False) -> tuple[float, float]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError("TM02 xy value must contain two numbers")
    result = tuple(float(item) for item in value)
    if len(result) != 2 or not np.all(np.isfinite(result)):
        raise ValueError("TM02 xy value must contain two finite numbers")
    if positive and any(item <= 0.0 for item in result):
        raise ValueError("TM02 xy size must be positive")
    return (result[0], result[1])


__all__ = [
    "DEFAULT_CABINET_WORLD_XYZ",
    "DEFAULT_CABINET_YAW_DEG",
    "DEFAULT_COMPLETION_BUTTON_WORLD_XYZ",
    "DEFAULT_CAN_PLACE_REGION_SIZE_XY",
    "DEFAULT_CAN_PLACE_REGION_WORLD_XYZ",
    "DEFAULT_LAYOUT_ID",
    "DEFAULT_STYLE_ID",
    "DEFAULT_WATER_BUTTON_WORLD_XYZ",
    "resolve_episode_plan",
    "resolve_scene_config",
    "resolve_task_definition",
    "sample_episode_plan",
]
