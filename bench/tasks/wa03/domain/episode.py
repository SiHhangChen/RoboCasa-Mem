"""WA-03 纯函数形式且可复现的 episode 决策。"""

from __future__ import annotations

from typing import Any

import numpy as np

from bench.tasks.wa03.materials import (
    normalize_coffee_material_candidates,
    normalize_sugar_cube_material_candidates,
    sample_coffee_materials,
    sample_sugar_cube_material,
)

from bench.tasks.wa03.domain.contracts import (
    COFFEE_IDS,
    ENV_NAME,
    TASK_ID,
    EpisodePlan,
    SceneConfig,
    TaskDefinition,
)


COFFEE_INITIAL_REGION_BOUNDS = {
    "x": (-0.38, -0.12),
    "y": (-5.414, -5.134),
}
COFFEE_INITIAL_REACH_CENTER_XY = (0.244, -5.274)
MAX_INITIAL_CUP_REACH_DISTANCE = 0.68
COFFEE_INITIAL_REGION_SLOTS = ("left", "center", "right")
COFFEE_INITIAL_SLOT_BY_ID = dict(zip(COFFEE_IDS, COFFEE_INITIAL_REGION_SLOTS, strict=True))
COFFEE_INITIAL_YAW_BOUNDS = {
    "coffee_A": (169.75, 177.75),
    "coffee_B": (141.0, 142.0),
    "coffee_C": (155.0, 163.0),
}
COFFEE_INITIAL_SAMPLING_INSET = 0.01
MIN_INITIAL_CUP_SEPARATION = 0.10
MAX_INITIAL_CUP_SAMPLING_ATTEMPTS = 1024


def target_sugared_coffee_for_seed(seed: int) -> str:
    """根据 seed 确定性选择 expert 要加糖的杯子。"""

    rng = np.random.default_rng(np.random.SeedSequence([int(seed), 0x57413033]))
    return COFFEE_IDS[int(rng.integers(0, len(COFFEE_IDS)))]


def validate_coffee_id(value: Any) -> str:
    """校验并规范化 WA-03 使用的杯子标识。"""

    coffee_id = str(value)
    if coffee_id not in COFFEE_IDS:
        raise ValueError(f"invalid WA-03 coffee id: {coffee_id!r}")
    return coffee_id


def coffee_initial_poses_for_seed(seed: int) -> dict[str, dict[str, Any]]:
    """在原双杯总区域等分出的三条平行区域内联合采样三只杯子。"""
    x_min, x_max = COFFEE_INITIAL_REGION_BOUNDS["x"]
    y_min, y_max = COFFEE_INITIAL_REGION_BOUNDS["y"]
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), 0x434F4646]))
    y_edges = np.linspace(y_min, y_max, len(COFFEE_IDS) + 1)
    region_bounds = {
        "left": (float(y_edges[2]), float(y_edges[3])),
        "center": (float(y_edges[1]), float(y_edges[2])),
        "right": (float(y_edges[0]), float(y_edges[1])),
    }
    reach_center = np.asarray(COFFEE_INITIAL_REACH_CENTER_XY, dtype=float)
    for attempt in range(1, MAX_INITIAL_CUP_SAMPLING_ATTEMPTS + 1):
        sampled_xy = []
        for coffee_id in COFFEE_IDS:
            slot = COFFEE_INITIAL_SLOT_BY_ID[coffee_id]
            slot_y_min, slot_y_max = region_bounds[slot]
            low = np.asarray(
                [x_min + COFFEE_INITIAL_SAMPLING_INSET, slot_y_min + COFFEE_INITIAL_SAMPLING_INSET],
                dtype=float,
            )
            high = np.asarray(
                [x_max - COFFEE_INITIAL_SAMPLING_INSET, slot_y_max - COFFEE_INITIAL_SAMPLING_INSET],
                dtype=float,
            )
            sampled_xy.append(rng.uniform(low, high))
        sampled_xy = np.asarray(sampled_xy, dtype=float)
        if np.any(
            np.linalg.norm(sampled_xy - reach_center, axis=1)
            > MAX_INITIAL_CUP_REACH_DISTANCE
        ):
            continue
        pairwise = [
            float(np.linalg.norm(sampled_xy[first] - sampled_xy[second]))
            for first in range(len(COFFEE_IDS))
            for second in range(first + 1, len(COFFEE_IDS))
        ]
        if min(pairwise) < MIN_INITIAL_CUP_SEPARATION:
            continue
        result: dict[str, dict[str, Any]] = {}
        for index, coffee_id in enumerate(COFFEE_IDS):
            yaw_min, yaw_max = COFFEE_INITIAL_YAW_BOUNDS[coffee_id]
            result[coffee_id] = {
                "xy": sampled_xy[index].tolist(),
                "yaw_deg": float(rng.uniform(yaw_min, yaw_max)),
                "sampling_attempt": attempt,
                "initial_region_slot": COFFEE_INITIAL_SLOT_BY_ID[coffee_id],
            }
        return result
    raise RuntimeError("could not sample a reachable WA-03 three-coffee arrangement")


def delivery_order_for_seed(seed: int) -> tuple[str, ...]:
    """根据 seed 确定三杯咖啡的运送先后顺序。"""

    rng = np.random.default_rng(np.random.SeedSequence([int(seed), 0x44454C49]))
    order = rng.permutation(COFFEE_IDS)
    return tuple(str(value) for value in order)


def final_slot_assignment_for_seed(seed: int) -> dict[str, str]:
    """根据 seed 将三杯咖啡一一分配到左、中、右放置区域。"""

    rng = np.random.default_rng(np.random.SeedSequence([int(seed), 0x46494E41]))
    slots = rng.permutation(np.asarray(["left", "center", "right"]))
    return {coffee_id: str(slots[index]) for index, coffee_id in enumerate(COFFEE_IDS)}


def resolve_task_definition(**values: Any) -> TaskDefinition:
    """把任务配置解析为稳定且经过 ID 校验的任务定义。"""

    task_id = str(values.get("id", TASK_ID))
    if task_id != TASK_ID:
        raise ValueError(f"WA-03 config must use task.id={TASK_ID!r}")
    return TaskDefinition(
        task_id=task_id,
        display_name=str(values.get("display_name", "Identify sugared coffee among three cups")),
        description=str(
            values.get(
                "description",
                "Deliver all three coffees and select the one that received sugar.",
            )
        ),
        success_conditions=tuple(
            str(item)
            for item in values.get(
                "success_conditions",
                (
                    "sugar_consumed",
                    "all_cups_ready_before_selection",
                    "correct_sugared_selection",
                ),
            )
        ),
    )


def resolve_scene_config(*, seed: int, env_split: str, **values: Any) -> SceneConfig:
    """解析场景配置，并在固定场景模式下锁定布局和场景 seed。"""

    fixed_scene = bool(values.get("fixed_scene", True))
    layout_id = 29 if fixed_scene else values.get("layout_id")
    style_id = 29 if fixed_scene else values.get("style_id")
    return SceneConfig(
        env_name=str(values.get("env_name", ENV_NAME)),
        env_split=str(env_split),
        layout_id=None if layout_id is None else int(layout_id),
        style_id=None if style_id is None else int(style_id),
        scene_seed=int(values.get("scene_seed", 0)) if fixed_scene else int(seed),
        fixed_scene=fixed_scene,
        sugar_name=str(values.get("sugar_name", "sugar_cube")),
    )


def resolve_episode_plan(*, seed: int, **values: Any) -> EpisodePlan:
    """一次性解析加糖模式、三杯运送顺序和最终槽位。"""

    mode = str(values.get("sugar_target_mode", "unmarked")).strip().lower()
    if mode not in {"unmarked", "marked"}:
        raise ValueError(f"unsupported WA-03 sugar_target_mode: {mode!r}")
    configured_target = values.get("target_sugared_coffee_id")
    if mode == "marked":
        target = (
            target_sugared_coffee_for_seed(seed)
            if configured_target is None
            else validate_coffee_id(configured_target)
        )
    else:
        if configured_target is not None:
            raise ValueError(
                "target_sugared_coffee_id is only valid when sugar_target_mode='marked'"
            )
        target = None
    seeded_order = delivery_order_for_seed(seed)
    configured_order = values.get("delivery_order")
    if configured_order is None:
        configured_first = values.get("first_delivery_coffee")
        if configured_first is None:
            order = seeded_order
        else:
            first = validate_coffee_id(configured_first)
            order = (first, *(coffee_id for coffee_id in seeded_order if coffee_id != first))
    else:
        order = tuple(validate_coffee_id(value) for value in configured_order)
        if len(order) != len(COFFEE_IDS) or set(order) != set(COFFEE_IDS):
            raise ValueError("WA-03 episode.delivery_order must contain every coffee exactly once")
    return EpisodePlan(
        seed=int(seed),
        sugar_target_mode=mode,
        target_sugared_coffee_id=target,
        delivery_order=tuple(order),
        final_slot_assignment=final_slot_assignment_for_seed(seed),
        coffee_material_by_id=sample_coffee_materials(
            seed,
            normalize_coffee_material_candidates(values.get("coffee_material_candidates")),
        ),
        sugar_cube_material=sample_sugar_cube_material(
            seed,
            normalize_sugar_cube_material_candidates(
                values.get("sugar_cube_material_candidates")
            ),
        ),
        max_horizon=int(values.get("max_horizon", 5000)),
    )


__all__ = [
    "COFFEE_INITIAL_REACH_CENTER_XY",
    "COFFEE_INITIAL_REGION_BOUNDS",
    "COFFEE_INITIAL_REGION_SLOTS",
    "COFFEE_INITIAL_SLOT_BY_ID",
    "COFFEE_INITIAL_YAW_BOUNDS",
    "MAX_INITIAL_CUP_REACH_DISTANCE",
    "MIN_INITIAL_CUP_SEPARATION",
    "coffee_initial_poses_for_seed",
    "delivery_order_for_seed",
    "final_slot_assignment_for_seed",
    "resolve_episode_plan",
    "resolve_scene_config",
    "resolve_task_definition",
    "target_sugared_coffee_for_seed",
    "validate_coffee_id",
]
