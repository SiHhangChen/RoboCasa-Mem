"""Deterministic TS04 two-drawer episode decisions."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from bench.tasks.ts04.domain.contracts import (
    COMPLETION_BUTTON_WORLD_XYZ,
    DISPLAY_NAME,
    DRAWER_WORLD_XYZS,
    DrawerOpenVariation,
    EpisodePlan,
    ENV_NAME,
    FIXTURE_DISPLAY_NAMES,
    FIXTURE_NAMES,
    SceneConfig,
    TaskDefinition,
    TASK_ID,
)
from bench.tasks.families.drawer_inspection.materials import (
    normalize_drawer_material_candidates,
)


TARGET_FIXTURE_SEED_SALT = 0x54533454
DRAWER_MATERIAL_SEED_SALT = 0x54534D34
PREHOOK_SEED_SALT = 0x54535034
HANDLE_OFFSET_SEED_SALT = 0x54534834


def normalize_drawer_world_xyzs(
    value: Any, *, expected_count: int
) -> tuple[tuple[float, float, float], ...]:
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


def normalize_target_fixture(value: Any | None) -> str | None:
    """Resolve an optional red-cube drawer for deterministic debugging."""

    if value is None or value == "":
        return None
    name = str(value).strip().lower()
    aliases = {display: fixture for fixture, display in FIXTURE_DISPLAY_NAMES.items()}
    resolved = aliases.get(name, name)
    if resolved not in FIXTURE_NAMES:
        raise ValueError(
            f"TS04 target_fixture must be one of {FIXTURE_NAMES}, got {value!r}"
        )
    return resolved


def sample_target_fixture(seed: int) -> str:
    """Sample the red-cube location from a stable seed stream."""

    episode_seed = int(seed)
    if episode_seed < 0:
        raise ValueError("TS04 seed must be non-negative")
    rng = np.random.default_rng(
        np.random.SeedSequence([episode_seed, TARGET_FIXTURE_SEED_SALT])
    )
    return str(rng.choice(FIXTURE_NAMES))


def resolve_task_definition(**values: Any) -> TaskDefinition:
    if str(values.get("id", TASK_ID)) != TASK_ID:
        raise ValueError(f"TS04 config must use task.id={TASK_ID!r}")
    conditions = tuple(str(item) for item in values.get("success_conditions", ()))
    if not conditions:
        raise ValueError("TS04 must declare success conditions")
    return TaskDefinition(
        task_id=TASK_ID,
        display_name=str(values.get("display_name", DISPLAY_NAME)),
        description=str(
            values.get(
                "description",
                "Search two drawers for a red cube and report the search length.",
            )
        ),
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
    button = tuple(
        float(v)
        for v in values.get(
            "completion_button_world_xyz", COMPLETION_BUTTON_WORLD_XYZ
        )
    )
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
    target = normalize_target_fixture(values.get("target_fixture"))
    horizon = int(values.get("max_horizon", 1800))
    if horizon <= 0:
        raise ValueError("TS04 max_horizon must be positive")
    material_candidates, prehook_candidates, side_half_range = (
        normalize_drawer_randomization(values)
    )
    materials: dict[str, str] = {}
    variations: dict[str, DrawerOpenVariation] = {}
    for index, fixture in enumerate(FIXTURE_NAMES):
        materials[fixture] = str(
            _fixture_rng(seed, DRAWER_MATERIAL_SEED_SALT, index).choice(
                material_candidates
            )
        )
        variant = str(
            _fixture_rng(seed, PREHOOK_SEED_SALT, index).choice(
                tuple(prehook_candidates)
            )
        )
        variations[fixture] = DrawerOpenVariation(
            prehook_variant=variant,
            prehook_rotation_offset_deg=prehook_candidates[variant],
            handle_side_offset_m=float(
                _fixture_rng(seed, HANDLE_OFFSET_SEED_SALT, index).uniform(
                    -side_half_range, side_half_range
                )
            ),
        )
    return EpisodePlan(
        seed=int(seed),
        demo_fixture_order=sample_demo_fixture_order(seed) if order is None else order,
        target_fixture=sample_target_fixture(seed) if target is None else target,
        max_horizon=horizon,
        drawer_material_by_fixture=materials,
        drawer_open_variation_by_fixture=variations,
    )


def _fixture_rng(seed: int, salt: int, index: int) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence([int(seed), salt, index]))


def normalize_drawer_randomization(
    values: Mapping[str, Any],
) -> tuple[tuple[str, ...], dict[str, tuple[float, float, float]], float]:
    candidates = normalize_drawer_material_candidates(
        values.get("drawer_material_candidates", ("original",))
    )
    rotations = values.get(
        "prehook_rotation_candidates_deg", {"baseline": (0.0, 0.0, 0.0)}
    )
    if not isinstance(rotations, Mapping) or not rotations:
        raise ValueError(
            "TS04 prehook_rotation_candidates_deg must be a non-empty mapping"
        )
    normalized: dict[str, tuple[float, float, float]] = {}
    for name, rotation in rotations.items():
        if not isinstance(name, str) or not name.strip():
            raise ValueError("TS04 prehook candidate names must be non-empty strings")
        normalized[name] = _finite_xyz(rotation, f"prehook candidate {name!r}")
    half_range = float(values.get("handle_side_offset_half_range_m", 0.0))
    if not np.isfinite(half_range) or not 0.0 <= half_range <= 0.04:
        raise ValueError("TS04 handle_side_offset_half_range_m must be in [0, 0.04]")
    return candidates, dict(sorted(normalized.items())), half_range


def _finite_xyz(value: Any, label: str) -> tuple[float, float, float]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"TS04 {label} must contain three values")
    result = tuple(float(item) for item in value)
    if len(result) != 3 or not np.all(np.isfinite(result)):
        raise ValueError(f"TS04 {label} must be a finite xyz value")
    return result


__all__ = [
    "normalize_fixture_order",
    "normalize_drawer_randomization",
    "normalize_target_fixture",
    "resolve_episode_plan",
    "resolve_scene_config",
    "resolve_task_definition",
    "sample_demo_fixture_order",
    "sample_target_fixture",
]
