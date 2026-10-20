"""WA-01 纯函数形式且可复现的 episode 决策。"""

from __future__ import annotations

from typing import Any

import numpy as np

from bench.tasks.wa01.domain.contracts import (
    COFFEE_IDS,
    ENV_NAME,
    TASK_ID,
    EpisodePlan,
    SceneConfig,
    TaskDefinition,
)
from bench.tasks.wa01.materials import (
    normalize_coffee_material_candidates,
    normalize_sugar_cube_material_candidates,
    sample_coffee_materials,
    sample_sugar_cube_material,
)


COFFEE_INITIAL_REGION_BOUNDS = {
    "x": (-0.27, -0.12),
    "y": (-5.414, -5.264),
}
COFFEE_INITIAL_REACH_CENTER_XY = (0.244, -5.274)
MAX_INITIAL_CUP_REACH_DISTANCE = 0.68
COFFEE_INITIAL_YAW_BOUNDS = {
    "coffee_A": (169.75, 177.75),
    "coffee_B": (141.0, 142.0),
}
MIN_INITIAL_CUP_SEPARATION = 0.14
MAX_INITIAL_CUP_SAMPLING_ATTEMPTS = 1024


def sugared_coffee_for_seed(seed: int) -> str:
    """根据 seed 确定性选择 expert 要加糖的杯子。"""

    rng = np.random.default_rng(np.random.SeedSequence([int(seed), 0x57413033]))
    return COFFEE_IDS[int(rng.integers(0, len(COFFEE_IDS)))]


def validate_coffee_id(value: Any) -> str:
    """校验并规范化 WA-01 使用的杯子标识。"""

    coffee_id = str(value)
    if coffee_id not in COFFEE_IDS:
        raise ValueError(f"invalid WA-01 coffee id: {coffee_id!r}")
    return coffee_id


def coffee_initial_poses_for_seed(seed: int) -> dict[str, dict[str, Any]]:
    """在共同可达区域内按 seed 联合采样两只杯子的初始位姿。"""
    x_min, x_max = COFFEE_INITIAL_REGION_BOUNDS["x"]
    y_min, y_max = COFFEE_INITIAL_REGION_BOUNDS["y"]
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), 0x434F4646]))
    low = np.asarray([x_min, y_min], dtype=float)
    high = np.asarray([x_max, y_max], dtype=float)
    reach_center = np.asarray(COFFEE_INITIAL_REACH_CENTER_XY, dtype=float)
    for attempt in range(1, MAX_INITIAL_CUP_SAMPLING_ATTEMPTS + 1):
        sampled_xy = rng.uniform(low, high, size=(len(COFFEE_IDS), 2))
        if np.any(
            np.linalg.norm(sampled_xy - reach_center, axis=1)
            > MAX_INITIAL_CUP_REACH_DISTANCE
        ):
            continue
        if float(np.linalg.norm(sampled_xy[0] - sampled_xy[1])) < MIN_INITIAL_CUP_SEPARATION:
            continue
        result: dict[str, dict[str, Any]] = {}
        for index, coffee_id in enumerate(COFFEE_IDS):
            yaw_min, yaw_max = COFFEE_INITIAL_YAW_BOUNDS[coffee_id]
            result[coffee_id] = {
                "xy": sampled_xy[index].tolist(),
                "yaw_deg": float(rng.uniform(yaw_min, yaw_max)),
                "sampling_attempt": attempt,
            }
        return result
    raise RuntimeError("could not sample a reachable WA-01 coffee pair")


def delivery_order_for_seed(seed: int) -> tuple[str, str]:
    """根据 seed 确定两杯咖啡的运送先后顺序。"""

    rng = np.random.default_rng(np.random.SeedSequence([int(seed), 0x44454C49]))
    order = rng.permutation(COFFEE_IDS)
    return str(order[0]), str(order[1])


def final_slot_assignment_for_seed(seed: int) -> dict[str, str]:
    """根据 seed 将两杯咖啡一一分配到左右放置区域。"""

    rng = np.random.default_rng(np.random.SeedSequence([int(seed), 0x46494E41]))
    slots = rng.permutation(np.asarray(["left", "right"]))
    return {coffee_id: str(slots[index]) for index, coffee_id in enumerate(COFFEE_IDS)}


def resolve_task_definition(**values: Any) -> TaskDefinition:
    """把任务配置解析为稳定且经过 ID 校验的任务定义。"""

    task_id = str(values.get("id", TASK_ID))
    if task_id != TASK_ID:
        raise ValueError(f"WA-01 config must use task.id={TASK_ID!r}")
    return TaskDefinition(
        task_id=task_id,
        display_name=str(values.get("display_name", "Sort sugared and unsugared coffee")),
        description=str(
            values.get(
                "description",
                "Deliver both coffees and select the one that received sugar.",
            )
        ),
        success_conditions=tuple(
            str(item)
            for item in values.get(
                "success_conditions",
                (
                    "sugar_consumed",
                    "cups_together_before_selection",
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
    """一次性解析运送顺序和最终槽位。"""

    seeded_order = delivery_order_for_seed(seed)
    configured_first = values.get("first_delivery_coffee")
    first = seeded_order[0] if configured_first is None else validate_coffee_id(configured_first)
    second = "coffee_B" if first == "coffee_A" else "coffee_A"
    return EpisodePlan(
        seed=int(seed),
        delivery_order=(first, second),
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
    "COFFEE_INITIAL_YAW_BOUNDS",
    "MAX_INITIAL_CUP_REACH_DISTANCE",
    "MIN_INITIAL_CUP_SEPARATION",
    "coffee_initial_poses_for_seed",
    "delivery_order_for_seed",
    "final_slot_assignment_for_seed",
    "resolve_episode_plan",
    "resolve_scene_config",
    "resolve_task_definition",
    "sugared_coffee_for_seed",
    "validate_coffee_id",
]
