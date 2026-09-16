"""Deterministic TM01 episode planning and immutable config resolution."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from bench.tasks.families.drawer_inspection.materials import (
    normalize_drawer_material_candidates,
)
from bench.tasks.tm01.domain.contracts import (
    BURNER_ID,
    BURNER_SITE,
    COMPLETION_BUTTON_NAMES,
    DRAWER_NAMES,
    ENV_NAME,
    KNOB_COLLISION_GEOM,
    KNOB_ID,
    KNOB_JOINT,
    TASK_ID,
    EpisodePlan,
    Route,
    SceneConfig,
    TaskDefinition,
)


DEFAULT_LAYOUT_ID = 39
DEFAULT_STYLE_ID = 24
DRAWER_MATERIAL_SEED_SALT = 0x544D4D41
DEFAULT_DRAWER_WORLD_XYZS = (
    (4.60, -2.37, 1.1363333333333332),
    (4.85, -2.37, 1.1363333333333332),
)
DEFAULT_COMPLETION_BUTTON_WORLD_XYZS = (
    (COMPLETION_BUTTON_NAMES[0], (3.55, -2.40, 0.92)),
    (COMPLETION_BUTTON_NAMES[1], (4.30, -3.30, 0.92)),
)


def sample_episode_plan(
    seed: int,
    *,
    max_horizon: int = 3200,
    drawer_material_candidates: Sequence[str] = ("original",),
) -> EpisodePlan:
    """Balance route and drawer order over every consecutive four seeds."""

    seed_value = int(seed)
    horizon = int(max_horizon)
    if horizon <= 0:
        raise ValueError("TM01 max_horizon must be positive")
    index = seed_value % 4
    route = Route.L1 if index in (0, 2) else Route.L2
    order = DRAWER_NAMES if index in (0, 1) else tuple(reversed(DRAWER_NAMES))
    material_candidates = normalize_drawer_material_candidates(
        drawer_material_candidates
    )
    materials = {
        drawer: str(
            np.random.default_rng(
                np.random.SeedSequence(
                    [seed_value, DRAWER_MATERIAL_SEED_SALT, fixture_index]
                )
            ).choice(material_candidates)
        )
        for fixture_index, drawer in enumerate(DRAWER_NAMES)
    }
    return EpisodePlan(
        seed=seed_value,
        route=route,
        drawer_order=(str(order[0]), str(order[1])),
        max_horizon=horizon,
        drawer_material_by_fixture=materials,
    )


def resolve_episode_plan(
    *,
    seed: int,
    max_horizon: int = 3200,
    route: str | Route | None = None,
    drawer_order: Sequence[str] | str | None = None,
    drawer_material_candidates: Sequence[str] = ("original",),
    **_: Any,
) -> EpisodePlan:
    sampled = sample_episode_plan(
        seed,
        max_horizon=max_horizon,
        drawer_material_candidates=drawer_material_candidates,
    )
    resolved_route = (
        sampled.route
        if route is None
        else route if isinstance(route, Route) else Route(str(route))
    )
    resolved_order = (
        sampled.drawer_order
        if drawer_order is None
        else normalize_drawer_order(drawer_order)
    )
    return EpisodePlan(
        seed=int(seed),
        route=resolved_route,
        drawer_order=resolved_order,
        max_horizon=int(max_horizon),
        drawer_material_by_fixture=dict(sampled.drawer_material_by_fixture),
    )


def normalize_drawer_order(value: Sequence[str] | str) -> tuple[str, str]:
    if isinstance(value, str):
        items = tuple(item.strip() for item in value.split(",") if item.strip())
    else:
        items = tuple(str(item).strip() for item in value)
    aliases = {"left": "drawer_left", "right": "drawer_right"}
    result = tuple(aliases.get(item, item) for item in items)
    if len(result) != 2 or set(result) != set(DRAWER_NAMES):
        raise ValueError(f"TM01 drawer_order must permute {DRAWER_NAMES}")
    return (result[0], result[1])


def resolve_task_definition(**values: Any) -> TaskDefinition:
    task_id = str(values.get("id", TASK_ID))
    if task_id != TASK_ID:
        raise ValueError(f"TM01 config must use task.id={TASK_ID!r}")
    return TaskDefinition(
        task_id=task_id,
        display_name=str(
            values.get("display_name", "Boil Water and Inspect Two Drawers")
        ),
        description=str(
            values.get(
                "description",
                "Turn on a stove, inspect two drawers, turn it off, and submit.",
            )
        ),
        success_conditions=tuple(
            str(item)
            for item in values.get(
                "success_conditions",
                (
                    "legal_event_route",
                    "native_knob_on_once",
                    "native_knob_off_once",
                    "burner_off",
                    "both_drawers_checked_once",
                    "all_drawers_closed",
                    "no_invalid_open",
                    "no_invalid_inspect",
                    "no_invalid_close",
                    "no_duplicate_check",
                    "no_close_before_inspect",
                    "pot_stable",
                    "either_completion_button_pressed",
                    "completion_button_pressed_once",
                    "no_invalid_completion_button_press",
                    "no_absorbing_failure",
                    "fixed_prompt",
                ),
            )
        ),
    )


def resolve_scene_config(*, env_split: str, **values: Any) -> SceneConfig:
    env_name = str(values.get("env_name", ENV_NAME))
    if env_name != ENV_NAME:
        raise ValueError(f"TM01 config must use scene.env_name={ENV_NAME!r}")
    layout_id = int(values.get("layout_id", DEFAULT_LAYOUT_ID))
    style_id = int(values.get("style_id", DEFAULT_STYLE_ID))
    if layout_id != DEFAULT_LAYOUT_ID or style_id != DEFAULT_STYLE_ID:
        raise ValueError("TM01 is fixed to layout_id=39 and style_id=24")
    names = tuple(str(name) for name in values.get("drawer_names", DRAWER_NAMES))
    if names != DRAWER_NAMES:
        raise ValueError(f"TM01 drawer_names must equal {DRAWER_NAMES}")
    xyzs = _xyzs(values.get("drawer_world_xyzs", DEFAULT_DRAWER_WORLD_XYZS))
    expected_stove_asset = str(values.get("expected_stove_asset", "Stovetop057"))
    knob_id = str(values.get("stove_knob_id", KNOB_ID))
    burner_id = str(values.get("burner_id", BURNER_ID))
    knob_joint = str(values.get("stove_knob_joint", KNOB_JOINT))
    knob_collision_geom = str(
        values.get("stove_knob_collision_geom", KNOB_COLLISION_GEOM)
    )
    burner_site = str(values.get("stove_burner_site", BURNER_SITE))
    frozen_appliance = (
        expected_stove_asset,
        knob_id,
        burner_id,
        knob_joint,
        knob_collision_geom,
        burner_site,
    )
    if frozen_appliance != (
        "Stovetop057",
        KNOB_ID,
        BURNER_ID,
        KNOB_JOINT,
        KNOB_COLLISION_GEOM,
        BURNER_SITE,
    ):
        raise ValueError(
            "TM01 is fixed to the Stovetop057 front_right-knob / "
            "rear_left-burner binding"
        )
    return SceneConfig(
        env_name=env_name,
        env_split=str(env_split),
        layout_id=layout_id,
        style_id=style_id,
        scene_seed=int(values.get("scene_seed", 0)),
        drawer_names=(names[0], names[1]),
        drawer_world_xyzs=xyzs,
        drawer_world_yaw_deg=float(values.get("drawer_world_yaw_deg", 180.0)),
        completion_button_world_xyzs=_named_xyzs(
            values.get(
                "completion_button_world_xyzs",
                dict(DEFAULT_COMPLETION_BUTTON_WORLD_XYZS),
            ),
            expected_names=COMPLETION_BUTTON_NAMES,
        ),
        stove_fixture=str(values.get("stove_fixture", "stove")),
        expected_stove_asset=expected_stove_asset,
        knob_id=knob_id,
        burner_id=burner_id,
        knob_joint=knob_joint,
        knob_collision_geom=knob_collision_geom,
        burner_site=burner_site,
        pot_name=str(values.get("pot_name", "pot")),
        pot_xml=str(
            values.get(
                "pot_xml",
                "objects/lightwheel/saucepan/Saucepan002/model.xml",
            )
        ),
    )


def _xyz(value: Any) -> tuple[float, float, float]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError("TM01 xyz value must contain three numbers")
    result = tuple(float(item) for item in value)
    if len(result) != 3 or not np.all(np.isfinite(result)):
        raise ValueError("TM01 xyz value must contain three finite numbers")
    return (result[0], result[1], result[2])


def _xyzs(value: Any) -> tuple[tuple[float, float, float], ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError("TM01 drawer_world_xyzs must be a sequence")
    result = tuple(_xyz(item) for item in value)
    if len(result) != len(DRAWER_NAMES):
        raise ValueError("TM01 requires exactly two drawer xyz values")
    return result


def _named_xyzs(
    value: Any,
    *,
    expected_names: tuple[str, ...],
) -> tuple[tuple[str, tuple[float, float, float]], ...]:
    if not isinstance(value, Mapping):
        raise ValueError("TM01 completion_button_world_xyzs must be a mapping")
    names = tuple(str(name) for name in value)
    if names != expected_names:
        raise ValueError(
            "TM01 completion button names must equal "
            f"{expected_names} in frozen order"
        )
    return tuple((name, _xyz(value[name])) for name in expected_names)


__all__ = [
    "DEFAULT_COMPLETION_BUTTON_WORLD_XYZS",
    "DEFAULT_DRAWER_WORLD_XYZS",
    "DEFAULT_LAYOUT_ID",
    "DEFAULT_STYLE_ID",
    "normalize_drawer_order",
    "resolve_episode_plan",
    "resolve_scene_config",
    "resolve_task_definition",
    "sample_episode_plan",
]
