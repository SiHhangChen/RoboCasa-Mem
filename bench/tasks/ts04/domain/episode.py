"""Deterministic TS04 two-drawer episode decisions."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from bench.tasks.ts04.domain.contracts import EpisodePlan, SceneConfig, TaskDefinition, FIXTURE_NAMES, FIXTURE_DISPLAY_NAMES
from bench.tasks.ts04.domain.contracts import (
    COMPLETION_BUTTON_WORLD_XYZ,
    DISPLAY_NAME,
    DRAWER_WORLD_XYZS,
    ENV_NAME,
    FIXTURE_DISPLAY_NAMES,
    FIXTURE_NAMES,
    TASK_ID,
)

def normalize_drawer_world_xyzs(value: Any, *, expected_count: int) -> tuple[tuple[float, float, float], ...]:
    rows = tuple(tuple(float(v) for v in row) for row in value)
    if len(rows) != expected_count or any(len(row) != 3 for row in rows):
        raise ValueError("invalid TS04 drawer positions")
    return rows


def normalize_fixture_order(value: Any | None) -> tuple[str, str] | None:
    if value is None or value == "":
        return None
    if isinstance(value, str):
        raw = [item.strip() for item in value.replace(";", ",").split(",")]
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        raw = [str(item).strip() for item in value]
    else:
        raise ValueError("TS04 demo_fixture_order must be a sequence")
    aliases = {display: fixture for fixture, display in FIXTURE_DISPLAY_NAMES.items()}
    items = tuple(aliases.get(item.lower(), item) for item in raw if item)
    if len(items) != 2 or set(items) != set(FIXTURE_NAMES):
        raise ValueError(f"TS04 demo_fixture_order must permute {FIXTURE_NAMES}")
    return (items[0], items[1])


def sample_demo_fixture_order(seed: int) -> tuple[str, str]:
    episode_seed = int(seed)
    if episode_seed < 0:
        raise ValueError("TS04 seed must be non-negative")
    return FIXTURE_NAMES if episode_seed % 2 == 0 else tuple(reversed(FIXTURE_NAMES))


def resolve_task_definition(**values: Any) -> TaskDefinition:
    if str(values.get("id", TASK_ID)) != TASK_ID:
        raise ValueError(f"TS04 config must use task.id={TASK_ID!r}")
    conditions = tuple(str(item) for item in values.get("success_conditions", ()))
    if not conditions:
        raise ValueError("TS04 must declare success conditions")
    return TaskDefinition(
        task_id=TASK_ID,
        display_name=str(values.get("display_name", DISPLAY_NAME)),
        description=str(values.get("description", "Inspect both drawers exactly once.")),
        success_conditions=conditions,
    )


def resolve_scene_config(*, seed: int, env_split: str, **values: Any) -> SceneConfig:
    del seed
    if str(values.get("env_name", ENV_NAME)) != ENV_NAME:
        raise ValueError(f"TS04 config must use scene.env_name={ENV_NAME!r}")
    if not bool(values.get("fixed_scene", True)):
        raise ValueError("TS04 requires fixed_scene=true")
    if int(values.get("layout_id", 24)) != 24 or int(values.get("style_id", 24)) != 24:
        raise ValueError("TS04 is fixed to layout/style 24/24")
    names = tuple(str(name) for name in values.get("fixture_names", FIXTURE_NAMES))
    if names != FIXTURE_NAMES:
        raise ValueError(f"TS04 fixture_names must equal {FIXTURE_NAMES}")
    xyzs = normalize_drawer_world_xyzs(
        values.get("drawer_world_xyzs", DRAWER_WORLD_XYZS), expected_count=2
    )
    open_threshold = float(values.get("open_threshold", 0.70))
    closed_threshold = float(values.get("closed_threshold", 0.08))
    if not 0 <= closed_threshold < open_threshold <= 1:
        raise ValueError("TS04 requires closed_threshold < open_threshold")
    button = tuple(float(v) for v in values.get("completion_button_world_xyz", COMPLETION_BUTTON_WORLD_XYZ))
    if len(button) != 3 or not np.all(np.isfinite(button)):
        raise ValueError("TS04 completion button position must be finite xyz")
    return SceneConfig(
        env_name=ENV_NAME,
        env_split=str(env_split),
        layout_id=24,
        style_id=24,
        scene_seed=0,
        fixed_scene=True,
        fixture_names=names,
        drawer_world_xyzs=xyzs,
        drawer_world_yaw_deg=float(values.get("drawer_world_yaw_deg", 180.0)),
        completion_button_world_xyz=(button[0], button[1], button[2]),
        open_threshold=open_threshold,
        closed_threshold=closed_threshold,
    )


def resolve_episode_plan(*, seed: int, **values: Any) -> EpisodePlan:
    order = normalize_fixture_order(values.get("demo_fixture_order"))
    horizon = int(values.get("max_horizon", 1800))
    if horizon <= 0:
        raise ValueError("TS04 max_horizon must be positive")
    return EpisodePlan(
        seed=int(seed),
        demo_fixture_order=sample_demo_fixture_order(seed) if order is None else order,
        max_horizon=horizon,
    )


__all__ = [
    "normalize_fixture_order",
    "resolve_episode_plan",
    "resolve_scene_config",
    "resolve_task_definition",
    "sample_demo_fixture_order",
]
