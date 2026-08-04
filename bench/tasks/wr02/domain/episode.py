"""Pure, reproducible WR-02 task and episode decisions."""

from __future__ import annotations

from typing import Any

from bench.tasks.wr02.domain.contracts import (
    CONDIMENT_IDS,
    ENV_NAME,
    HOME_REGIONS,
    TASK_ID,
    USE_REGIONS,
    EpisodePlan,
    SceneConfig,
    TaskDefinition,
)
from bench.tasks.wr02.domain.geometry import sample_joint_return_offsets


EPISODE_CASES = (
    ({"ketchup": "shelf_upper", "mustard": "shelf_lower"}, {"ketchup": "sink_left", "mustard": "sink_right"}),
    ({"ketchup": "shelf_upper", "mustard": "shelf_lower"}, {"ketchup": "sink_right", "mustard": "sink_left"}),
    ({"ketchup": "shelf_lower", "mustard": "shelf_upper"}, {"ketchup": "sink_left", "mustard": "sink_right"}),
    ({"ketchup": "shelf_lower", "mustard": "shelf_upper"}, {"ketchup": "sink_right", "mustard": "sink_left"}),
)


def resolve_task_definition(**values: Any) -> TaskDefinition:
    task_id = str(values.get("id", TASK_ID))
    if task_id != TASK_ID:
        raise ValueError(f"WR-02 config must use task.id={TASK_ID!r}")
    conditions = tuple(str(value) for value in values.get("success_conditions", ()))
    if not conditions:
        raise ValueError("WR-02 must declare success conditions")
    return TaskDefinition(
        task_id=task_id,
        display_name=str(values.get("display_name", "Return sponge and soap from the sink")),
        description=str(values.get("description", "Stage both items beside the sink, then restore their remembered shelf order.")),
        success_conditions=conditions,
    )


def resolve_scene_config(*, seed: int, env_split: str, **values: Any) -> SceneConfig:
    fixed = bool(values.get("fixed_scene", True))
    return SceneConfig(
        env_name=str(values.get("env_name", ENV_NAME)),
        env_split=str(env_split),
        layout_id=32 if fixed else _optional_int(values.get("layout_id")),
        style_id=32 if fixed else _optional_int(values.get("style_id")),
        scene_seed=int(values.get("scene_seed", 0)) if fixed else int(seed),
        fixed_scene=fixed,
    )


def resolve_episode_plan(*, seed: int, **values: Any) -> EpisodePlan:
    settings = dict(values)
    max_horizon = int(settings.pop("max_horizon", 5000))
    half_extents = settings.pop(
        "return_place_xy_half_extents", (0.3331, 0.0388)
    )
    minimum_separation = float(settings.pop("return_level_min_separation", 0.20))
    nominal_separation = float(
        settings.pop("return_nominal_level_separation", 0.2966666666666667)
    )
    transport_order = tuple(settings.pop("transport_order", CONDIMENT_IDS))
    if settings:
        raise ValueError("unknown WR-02 episode settings: " + ", ".join(sorted(settings)))
    if transport_order != CONDIMENT_IDS:
        raise ValueError(f"WR-02 transport_order must be {CONDIMENT_IDS!r}")
    if max_horizon <= 0:
        raise ValueError("WR-02 max_horizon must be positive")
    case_id = int(seed) % len(EPISODE_CASES)
    home, use = EPISODE_CASES[case_id]
    if set(home.values()) != set(HOME_REGIONS) or set(use.values()) != set(USE_REGIONS):
        raise ValueError("WR-02 episode case must be bijective")
    offsets, sampling = sample_joint_return_offsets(
        seed=int(seed),
        home_assignment=home,
        half_extents=half_extents,
        minimum_separation=minimum_separation,
        nominal_separation=nominal_separation,
    )
    return EpisodePlan(
        seed=int(seed),
        case_id=case_id,
        home_assignment=dict(home),
        use_assignment=dict(use),
        transport_order=CONDIMENT_IDS,
        return_sampled_offsets_xy=offsets,
        return_sampling_metadata=sampling,
        max_horizon=max_horizon,
    )


def public_return_prompt() -> str:
    """Return instruction that never reveals the private lower/upper binding."""

    return "Return the sponge and soap to their original open-shelf levels."


def _optional_int(value: Any) -> int | None:
    return None if value is None else int(value)


__all__ = [
    "EPISODE_CASES",
    "public_return_prompt",
    "resolve_episode_plan",
    "resolve_scene_config",
    "resolve_task_definition",
]
