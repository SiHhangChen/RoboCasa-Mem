"""Deterministic TS02 episode decisions and top-level domain resolvers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from bench.tasks.ts02.domain.contracts import (
    ENV_NAME,
    FIXTURE_DISPLAY_NAMES,
    FIXTURE_NAMES,
    TASK_ID,
    DrawerOpenVariation,
    EpisodePlan,
    SceneConfig,
    TaskDefinition,
)
from bench.tasks.ts02.domain.geometry import (
    DEFAULT_DRAWER_WORLD_XYZS,
    DEFAULT_DRAWER_WORLD_YAW_DEG,
    normalize_drawer_world_xyzs,
)
from bench.tasks.ts02.drawer_materials import normalize_drawer_material_candidates


DEMO_ORDER_SEED_SALT = 0x54533032
TARGET_FIXTURE_SEED_SALT = 0x54534754
DRAWER_MATERIAL_SEED_SALT = 0x54534D41
PREHOOK_SEED_SALT = 0x54535048
HANDLE_OFFSET_SEED_SALT = 0x5453484F
DEFAULT_LAYOUT_ID = 24
DEFAULT_STYLE_ID = 24
DEFAULT_OPEN_THRESHOLD = 0.70
DEFAULT_CLOSED_THRESHOLD = 0.08
DEFAULT_COMPLETION_BUTTON_WORLD_XYZ = (3.85, -2.06, 0.92)


def normalize_fixture_order(value: Any | None) -> tuple[str, ...] | None:
    """Return a fixture permutation or ``None`` for seeded sampling."""

    if value is None or value == "":
        return None
    if isinstance(value, str):
        raw = [item.strip() for item in value.replace(";", ",").split(",")]
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        raw = [str(item).strip() for item in value]
    else:
        raise ValueError(
            "demo_fixture_order must be a sequence or comma-separated string"
        )
    aliases = {display: fixture for fixture, display in FIXTURE_DISPLAY_NAMES.items()}
    items = tuple(aliases.get(item, item) for item in raw if item)
    if len(items) != len(FIXTURE_NAMES) or set(items) != set(FIXTURE_NAMES):
        raise ValueError(
            "demo_fixture_order must be a permutation of "
            f"{FIXTURE_NAMES}, got {value!r}"
        )
    return items


def sample_demo_fixture_order(seed: int) -> tuple[str, ...]:
    """Sample the expert order only; it is not a success target."""

    rng = np.random.default_rng(
        np.random.SeedSequence([int(seed), DEMO_ORDER_SEED_SALT])
    )
    return tuple(str(value) for value in rng.permutation(FIXTURE_NAMES))


def normalize_target_fixture(value: Any | None) -> str | None:
    """Resolve an optional fixture name used for deterministic debugging."""

    if value is None or value == "":
        return None
    name = str(value).strip().lower()
    aliases = {display: fixture for fixture, display in FIXTURE_DISPLAY_NAMES.items()}
    resolved = aliases.get(name, name)
    if resolved not in FIXTURE_NAMES:
        raise ValueError(f"target_fixture must be one of {FIXTURE_NAMES}, got {value!r}")
    return resolved


def sample_target_fixture(seed: int) -> str:
    """Sample the hidden red-cube location independently of expert order."""

    rng = np.random.default_rng(
        np.random.SeedSequence([int(seed), TARGET_FIXTURE_SEED_SALT])
    )
    return str(rng.choice(FIXTURE_NAMES))


def resolve_task_definition(**values: Any) -> TaskDefinition:
    task_id = str(values.get("id", TASK_ID))
    if task_id != TASK_ID:
        raise ValueError(f"TS02 config must use task.id={TASK_ID!r}")
    return TaskDefinition(
        task_id=task_id,
        display_name=str(values.get("display_name", "Inspect Three Cabinets")),
        description=str(
            values.get(
                "description",
                "Search three drawers for a red cube and report the search length.",
            )
        ),
        success_conditions=tuple(
            str(item)
            for item in values.get(
                "success_conditions",
                (
                    "target_found",
                    "search_stopped_at_target",
                    "at_least_one_fixture_checked",
                    "no_duplicate_completion",
                    "no_active_fixture",
                    "all_fixtures_closed",
                    "no_invalid_open",
                    "no_invalid_inspect",
                    "no_invalid_close",
                    "no_duplicate_check",
                    "no_close_before_inspect",
                    "no_search_after_target",
                    "completion_button_pressed",
                    "completion_button_count_matches_search_count",
                    "no_invalid_completion_button_press",
                ),
            )
        ),
    )


def resolve_scene_config(*, seed: int, env_split: str, **values: Any) -> SceneConfig:
    """Resolve the one calibrated layout and its three drawer poses."""

    env_name = str(values.get("env_name", ENV_NAME))
    if env_name != ENV_NAME:
        raise ValueError(f"TS02 config must use scene.env_name={ENV_NAME!r}")
    fixed = bool(values.get("fixed_scene", True))
    requested_layout = values.get("layout_id", DEFAULT_LAYOUT_ID)
    requested_style = values.get("style_id", DEFAULT_STYLE_ID)
    if fixed and int(requested_layout) != DEFAULT_LAYOUT_ID:
        raise ValueError(f"TS02 is fixed to layout_id={DEFAULT_LAYOUT_ID}")
    if fixed and int(requested_style) != DEFAULT_STYLE_ID:
        raise ValueError(f"TS02 is fixed to style_id={DEFAULT_STYLE_ID}")
    fixture_names = tuple(
        str(name) for name in values.get("fixture_names", FIXTURE_NAMES)
    )
    if fixture_names != FIXTURE_NAMES:
        raise ValueError(f"TS02 fixture_names must equal {FIXTURE_NAMES}")
    open_threshold = float(values.get("open_threshold", DEFAULT_OPEN_THRESHOLD))
    closed_threshold = float(values.get("closed_threshold", DEFAULT_CLOSED_THRESHOLD))
    if not 0.0 <= closed_threshold < open_threshold <= 1.0:
        raise ValueError("TS02 requires 0 <= closed_threshold < open_threshold <= 1")
    layout_id = DEFAULT_LAYOUT_ID if fixed else requested_layout
    style_id = DEFAULT_STYLE_ID if fixed else requested_style
    return SceneConfig(
        env_name=env_name,
        env_split=str(env_split),
        layout_id=None if layout_id is None else int(layout_id),
        style_id=None if style_id is None else int(style_id),
        scene_seed=int(values.get("scene_seed", 0)) if fixed else int(seed),
        fixed_scene=fixed,
        fixture_names=fixture_names,
        drawer_world_xyzs=normalize_drawer_world_xyzs(
            values.get("drawer_world_xyzs", DEFAULT_DRAWER_WORLD_XYZS)
        ),
        drawer_world_yaw_deg=float(
            values.get("drawer_world_yaw_deg", DEFAULT_DRAWER_WORLD_YAW_DEG)
        ),
        completion_button_world_xyz=_finite_xyz(
            values.get(
                "completion_button_world_xyz",
                DEFAULT_COMPLETION_BUTTON_WORLD_XYZ,
            ),
            "completion_button_world_xyz",
        ),
        open_threshold=open_threshold,
        closed_threshold=closed_threshold,
    )


def resolve_episode_plan(*, seed: int, **values: Any) -> EpisodePlan:
    configured = normalize_fixture_order(values.get("demo_fixture_order"))
    configured_target = normalize_target_fixture(values.get("target_fixture"))
    max_horizon = int(values.get("max_horizon", 2800))
    if max_horizon <= 0:
        raise ValueError("TS02 episode.max_horizon must be positive")
    material_candidates, prehook_candidates, side_half_range = (
        normalize_drawer_randomization(values)
    )
    materials: dict[str, str] = {}
    variations: dict[str, DrawerOpenVariation] = {}
    for index, fixture in enumerate(FIXTURE_NAMES):
        materials[fixture] = str(
            _fixture_rng(seed, DRAWER_MATERIAL_SEED_SALT, index).choice(material_candidates)
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
        demo_fixture_order=(
            sample_demo_fixture_order(seed) if configured is None else configured
        ),
        target_fixture=(
            sample_target_fixture(seed) if configured_target is None else configured_target
        ),
        max_horizon=max_horizon,
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
    rotations = values.get("prehook_rotation_candidates_deg", {"baseline": (0.0, 0.0, 0.0)})
    if not isinstance(rotations, Mapping) or not rotations:
        raise ValueError("TS02 prehook_rotation_candidates_deg must be a non-empty mapping")
    normalized: dict[str, tuple[float, float, float]] = {}
    for name, rotation in rotations.items():
        if not isinstance(name, str) or not name.strip():
            raise ValueError("TS02 prehook candidate names must be non-empty strings")
        normalized[name] = _finite_xyz(rotation, f"prehook candidate {name!r}")
    half_range = float(values.get("handle_side_offset_half_range_m", 0.0))
    if not np.isfinite(half_range) or not 0.0 <= half_range <= 0.04:
        raise ValueError("TS02 handle_side_offset_half_range_m must be in [0, 0.04]")
    return candidates, dict(sorted(normalized.items())), half_range


def _finite_xyz(value: Any, label: str) -> tuple[float, float, float]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"TS02 {label} must contain three values")
    result = tuple(float(item) for item in value)
    if len(result) != 3 or not np.all(np.isfinite(result)):
        raise ValueError(f"TS02 {label} must be a finite xyz value")
    return result


__all__ = [
    "DEFAULT_CLOSED_THRESHOLD",
    "DEFAULT_COMPLETION_BUTTON_WORLD_XYZ",
    "DEFAULT_LAYOUT_ID",
    "DEFAULT_OPEN_THRESHOLD",
    "DEFAULT_STYLE_ID",
    "normalize_fixture_order",
    "normalize_target_fixture",
    "normalize_drawer_randomization",
    "resolve_episode_plan",
    "resolve_scene_config",
    "resolve_task_definition",
    "sample_demo_fixture_order",
    "sample_target_fixture",
]
