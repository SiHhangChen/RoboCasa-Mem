"""Deterministic WA-02 episode decisions, independent of the simulator."""

from __future__ import annotations

from typing import Any

from bench.tasks.wa02.domain.contracts import (
    CUP_IDS,
    DISPLAY_NAME,
    ENV_NAME,
    TASK_ID,
    TEMPERATURES,
    EpisodePlan,
    SceneConfig,
    TaskDefinition,
)
from bench.tasks.wa02.domain.geometry import (
    other_cup_id,
    service_slot_assignment_for_seed,
    validate_service_slot_assignment,
)


def balanced_axes_for_seed(seed: int) -> dict[str, Any]:
    """Return the balanced 2 x 2 x 2 x 2 label grid entry for one seed.

    Across every consecutive block of sixteen seeds, hot-cup identity, requested
    temperature, service-slot permutation, and execution sequence are balanced.
    """

    index = int(seed) % 16
    return {
        "hot_cup_id": CUP_IDS[(index >> 2) & 0x1],
        "requested_temperature": TEMPERATURES[(index >> 1) & 0x1],
        "service_slot_swap": bool(index & 0x1),
        "fill_temperature_order": (
            ("hot", "cold")
            if ((index >> 3) & 0x1) == 0
            else ("cold", "hot")
        ),
    }


def validate_cup_id(value: Any) -> str:
    cup = str(value)
    if cup not in CUP_IDS:
        raise ValueError(f"invalid WA-02 cup id: {value!r}")
    return cup


def validate_temperature(value: Any) -> str:
    temperature = str(value).strip().lower()
    if temperature not in TEMPERATURES:
        raise ValueError(f"invalid WA-02 temperature: {value!r}")
    return temperature


def resolve_task_definition(**values: Any) -> TaskDefinition:
    task_id = str(values.get("id", TASK_ID))
    if task_id != TASK_ID:
        raise ValueError(f"WA-02 config must use task.id={TASK_ID!r}")
    return TaskDefinition(
        task_id=task_id,
        display_name=str(values.get("display_name", DISPLAY_NAME)),
        description=str(
            values.get(
                "description",
                "Fill identical cups with hot and cold water, then serve the requested one by memory.",
            )
        ),
        success_conditions=tuple(
            str(item)
            for item in values.get(
                "success_conditions",
                (
                    "hot_water_filled",
                    "cold_water_filled",
                    "both_cups_staged_before_final_pick",
                    "requested_cup_picked_up",
                    "correct_temperature_selected",
                ),
            )
        ),
    )


def resolve_scene_config(*, seed: int, env_split: str, **values: Any) -> SceneConfig:
    fixed_scene = bool(values.get("fixed_scene", True))
    return SceneConfig(
        env_name=str(values.get("env_name", ENV_NAME)),
        env_split=str(env_split),
        layout_id=(15 if fixed_scene else _optional_int(values.get("layout_id"))),
        style_id=(34 if fixed_scene else _optional_int(values.get("style_id"))),
        scene_seed=int(values.get("scene_seed", 0)) if fixed_scene else int(seed),
        fixed_scene=fixed_scene,
    )


def resolve_episode_plan(*, seed: int, **values: Any) -> EpisodePlan:
    """Freeze every hidden WA-02 decision before environment construction."""

    if "fill_order" in values:
        raise ValueError(
            "WA-02 no longer supports fill_order overrides; "
            "the execution sequence is seed-balanced"
        )
    balanced = balanced_axes_for_seed(seed)
    configured_hot = values.get("hot_cup_id")
    hot_cup = (
        str(balanced["hot_cup_id"])
        if _is_unspecified(configured_hot)
        else validate_cup_id(configured_hot)
    )
    cold_cup = other_cup_id(hot_cup)
    configured_request = values.get("requested_temperature")
    requested_temperature = (
        str(balanced["requested_temperature"])
        if _is_unspecified(configured_request)
        else validate_temperature(configured_request)
    )
    requested_cup = hot_cup if requested_temperature == "hot" else cold_cup
    configured_slots = values.get("service_slot_by_cup")
    if configured_slots is None:
        assignment = service_slot_assignment_for_seed(
            seed,
            swap=bool(values.get("service_slot_swap", balanced["service_slot_swap"])),
        )
    else:
        if not isinstance(configured_slots, dict):
            raise ValueError("WA-02 episode.service_slot_by_cup must be a mapping")
        assignment = validate_service_slot_assignment(configured_slots)
    return EpisodePlan(
        seed=int(seed),
        hot_cup_id=hot_cup,
        cold_cup_id=cold_cup,
        requested_temperature=requested_temperature,
        requested_cup_id=requested_cup,
        fill_temperature_order=tuple(balanced["fill_temperature_order"]),
        service_slot_by_cup=assignment,
        max_horizon=int(values.get("max_horizon", 6000)),
    )


def _optional_int(value: Any) -> int | None:
    return None if value is None else int(value)


def _is_unspecified(value: Any) -> bool:
    """Recognize scalar YAML placeholders without hashing arbitrary input."""

    if value is None:
        return True
    return isinstance(value, str) and value.strip().lower() in {"", "none", "random"}


__all__ = [
    "balanced_axes_for_seed",
    "resolve_episode_plan",
    "resolve_scene_config",
    "resolve_task_definition",
    "validate_cup_id",
    "validate_temperature",
]
