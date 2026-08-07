"""Deterministic WA-04 episode decisions, independent of the simulator."""

from __future__ import annotations

from itertools import permutations
from typing import Any

from bench.tasks.wa04.domain.contracts import (
    CUP_IDS,
    DISPLAY_NAME,
    ENV_NAME,
    TASK_ID,
    TEMPERATURES,
    EpisodePlan,
    SceneConfig,
    TaskDefinition,
)
from bench.tasks.wa04.domain.geometry import (
    service_slot_assignment_for_seed,
    validate_service_slot_assignment,
)


_CUP_PERMUTATIONS = tuple(permutations(CUP_IDS))


def balanced_axes_for_seed(seed: int) -> dict[str, Any]:
    """Return one entry of the balanced 2 x 3 x 6 x 6 WA-04 grid."""

    index = int(seed) % 216
    return {
        "requested_temperature": TEMPERATURES[index % 2],
        "requested_cup_id": CUP_IDS[(index // 2) % 3],
        "fill_cup_order": _CUP_PERMUTATIONS[(index // 6) % 6],
        "service_slot_permutation_index": (index // 36) % 6,
    }


def validate_cup_id(value: Any) -> str:
    cup = str(value)
    if cup not in CUP_IDS:
        raise ValueError(f"invalid WA-04 cup id: {value!r}")
    return cup


def validate_temperature(value: Any) -> str:
    temperature = str(value).strip().lower()
    if temperature not in TEMPERATURES:
        raise ValueError(f"invalid WA-04 temperature: {value!r}")
    return temperature


def resolve_fill_order(value: Any, *, seed: int) -> tuple[str, ...]:
    """Resolve a permutation of all three cup identities."""

    if _is_unspecified(value) or (
        isinstance(value, str)
        and value.strip().lower().replace("_", "-") in {"balanced", "seed-balanced"}
    ):
        return tuple(balanced_axes_for_seed(seed)["fill_cup_order"])
    if isinstance(value, str):
        values = tuple(item.strip() for item in value.split(",") if item.strip())
    else:
        try:
            values = tuple(str(item) for item in value)
        except TypeError as exc:
            raise ValueError("WA-04 fill_order must be balanced or a cup permutation") from exc
    if len(values) != len(CUP_IDS) or set(values) != set(CUP_IDS):
        raise ValueError("WA-04 fill_order must contain cup_A, cup_B, and cup_C once")
    return values


def resolve_task_definition(**values: Any) -> TaskDefinition:
    task_id = str(values.get("id", TASK_ID))
    if task_id != TASK_ID:
        raise ValueError(f"WA-04 config must use task.id={TASK_ID!r}")
    return TaskDefinition(
        task_id=task_id,
        display_name=str(values.get("display_name", DISPLAY_NAME)),
        description=str(
            values.get(
                "description",
                "Fill three identical cups so one has the requested temperature, then select it by memory.",
            )
        ),
        success_conditions=tuple(
            str(item)
            for item in values.get(
                "success_conditions",
                (
                    "all_three_cups_filled_as_planned",
                    "all_three_cups_staged_before_final_pick",
                    "requested_cup_picked_up",
                    "unique_requested_temperature_selected",
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
    """Freeze every hidden WA-04 decision before environment construction."""

    balanced = balanced_axes_for_seed(seed)
    configured_target = values.get("requested_cup_id")
    requested_cup = (
        str(balanced["requested_cup_id"])
        if _is_unspecified(configured_target)
        else validate_cup_id(configured_target)
    )
    configured_temperature = values.get("requested_temperature")
    requested_temperature = (
        str(balanced["requested_temperature"])
        if _is_unspecified(configured_temperature)
        else validate_temperature(configured_temperature)
    )
    opposite = "cold" if requested_temperature == "hot" else "hot"
    temperature_by_cup = {
        cup: requested_temperature if cup == requested_cup else opposite
        for cup in CUP_IDS
    }

    configured_slots = values.get("service_slot_by_cup")
    if configured_slots is None:
        assignment = service_slot_assignment_for_seed(
            seed,
            permutation_index=int(balanced["service_slot_permutation_index"]),
        )
    else:
        if not isinstance(configured_slots, dict):
            raise ValueError("WA-04 episode.service_slot_by_cup must be a mapping")
        assignment = validate_service_slot_assignment(configured_slots)
    return EpisodePlan(
        seed=int(seed),
        temperature_by_cup=temperature_by_cup,
        requested_temperature=requested_temperature,
        requested_cup_id=requested_cup,
        fill_cup_order=resolve_fill_order(values.get("fill_order"), seed=seed),
        service_slot_by_cup=assignment,
        max_horizon=int(values.get("max_horizon", 8500)),
    )


def _optional_int(value: Any) -> int | None:
    return None if value is None else int(value)


def _is_unspecified(value: Any) -> bool:
    if value is None:
        return True
    return isinstance(value, str) and value.strip().lower() in {"", "none", "random"}


__all__ = [
    "balanced_axes_for_seed",
    "resolve_episode_plan",
    "resolve_fill_order",
    "resolve_scene_config",
    "resolve_task_definition",
    "validate_cup_id",
    "validate_temperature",
]
