"""Pure, reproducible WR-05 task and episode decisions."""

from __future__ import annotations

from typing import Any

from bench.tasks.wr05.domain.contracts import (
    CONDIMENT_IDS,
    ENV_NAME,
    HOME_REGIONS,
    TASK_ID,
    USE_REGIONS,
    EpisodePlan,
    SceneConfig,
    TaskDefinition,
)
from bench.tasks.wr05.domain.geometry import sample_joint_return_offsets


EPISODE_CASES = (
    ({"ketchup": "shelf_upper", "mustard": "shelf_lower"}, {"ketchup": "plate_left", "mustard": "plate_right"}),
    ({"ketchup": "shelf_upper", "mustard": "shelf_lower"}, {"ketchup": "plate_right", "mustard": "plate_left"}),
    ({"ketchup": "shelf_lower", "mustard": "shelf_upper"}, {"ketchup": "plate_left", "mustard": "plate_right"}),
    ({"ketchup": "shelf_lower", "mustard": "shelf_upper"}, {"ketchup": "plate_right", "mustard": "plate_left"}),
)


def resolve_task_definition(**values: Any) -> TaskDefinition:
    task_id = str(values.get("id", TASK_ID))
    if task_id != TASK_ID:
        raise ValueError(f"WR-05 config must use task.id={TASK_ID!r}")
    conditions = tuple(str(value) for value in values.get("success_conditions", ()))
    if not conditions:
        raise ValueError("WR-05 must declare success conditions")
    return TaskDefinition(
        task_id=task_id,
        display_name=str(values.get("display_name", "Return ketchup and mustard from the plate")),
        description=str(values.get("description", "Stage both bottles beside the serving plate, then restore their remembered shelf levels.")),
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
        raise ValueError("unknown WR-05 episode settings: " + ", ".join(sorted(settings)))
    if transport_order != CONDIMENT_IDS:
        raise ValueError(f"WR-05 transport_order must be {CONDIMENT_IDS!r}")
    if max_horizon <= 0:
        raise ValueError("WR-05 max_horizon must be positive")
    case_id = int(seed) % len(EPISODE_CASES)
    home, use = EPISODE_CASES[case_id]
    if set(home.values()) != set(HOME_REGIONS) or set(use.values()) != set(USE_REGIONS):
        raise ValueError("WR-05 episode case must be bijective")
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


def public_task_prompt() -> str:
    """Complete public instruction without private shelf or plate assignments."""

    return (
        "Remember whether the ketchup and mustard bottles start on the lower or "
        "upper shelf. First move the ketchup beside the serving plate, then move "
        "the mustard beside it. After both bottles are staged, return the ketchup "
        "and then the mustard to their original shelf levels."
    )


def public_return_prompt() -> str:
    """Backward-compatible alias for the complete public task prompt."""

    return public_task_prompt()


def _optional_int(value: Any) -> int | None:
    return None if value is None else int(value)


__all__ = [
    "EPISODE_CASES",
    "public_return_prompt",
    "resolve_episode_plan",
    "resolve_scene_config",
    "resolve_task_definition",
]
