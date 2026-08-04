"""连接 WA-01 流程操作、通用控制器和语义事件的薄适配器。"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from bench.execution.manipulation.object_interactions import (
    get_object_hold_status,
    is_object_released,
    lift_held_object_for_transport,
)
from bench.tasks.wa01.execution.geometry import (
    build_reachability_target_resolver,
    resolve_dynamic_place_target,
)


def build_stage_handlers(
    controller: Any,
    *,
    accept_sugar_delivery: bool = True,
) -> dict[str, Callable[..., dict[str, Any]]]:
    """建立 OperateStage 名称到具体控制器或语义操作的映射。"""

    controller.reachability_target_resolver = build_reachability_target_resolver(
        controller
    )

    def place_dynamic(target: str, **kwargs: Any) -> dict[str, Any]:
        """解析餐桌动态目标并调用通用世界坐标放置接口。"""

        target_kind = str(kwargs.pop("target_kind", ""))
        resolved = resolve_dynamic_place_target(
            controller,
            target,
            target_kind=target_kind,
            options=kwargs,
        )
        if resolved is None:
            return {
                "primitive": "wa01_place_at_task_target",
                "object_name": target,
                "target_kind": target_kind,
                "pass": False,
                "reason": f"unsupported_target_kind:{target_kind}",
            }
        result = controller.place_at_world_pose(
            target,
            target_name=str(resolved["target_name"]),
            target_pos_world=resolved["target_pos_world"],
            collision_options=resolved.get("collision_options"),
            **dict(resolved.get("place_kwargs") or {}),
        )
        result.update(
            {
                "primitive": "wa01_place_at_task_target",
                "target_kind": target_kind,
                "target_info": resolved["target_info"],
            }
        )
        return result

    def record_sugar(target: str, **kwargs: Any) -> dict[str, Any]:
        """在糖块物理投递完成后触发加糖语义事件。"""

        del kwargs
        return apply_sugar_delivery_event(
            controller,
            target,
            coffee_id=controller.env.target_sugared_coffee_id,
            accept=accept_sugar_delivery,
        )

    def lift_held(target: str, **kwargs: Any) -> dict[str, Any]:
        """将当前持有杯子按该杯配置的高度显式抬升。"""

        return lift_held_object_for_transport(
            controller,
            target,
            stage=str(kwargs.get("stage", f"lift_{target}_for_transport")),
            lift_height=float(kwargs.get("lift_height", 0.18)),
            settle_steps=int(kwargs.get("settle_steps", 5)),
        )

    def verify_pick(target: str, **kwargs: Any) -> dict[str, Any]:
        """在实际抓取前检查物品当前位置是否可达。"""

        return verify_current_pick_reachability(controller, target, **kwargs)

    def verify_hold(target: str, **kwargs: Any) -> dict[str, Any]:
        """在 Base 运输前检查杯子仍被可靠持有。"""

        return verify_coffee_cup_held(controller, target, **kwargs)

    def verify_placement(target: str, **kwargs: Any) -> dict[str, Any]:
        """在放置后检查杯子物理状态、区域和释放状态。"""

        return verify_dining_placement(controller, target, **kwargs)

    def verify_ready(target: str, **kwargs: Any) -> dict[str, Any]:
        """在最终选杯前统一检查并锁存双杯就位条件。"""

        del target
        return verify_cups_ready_for_selection(controller, **kwargs)

    def select(target: str, **kwargs: Any) -> dict[str, Any]:
        """记录最终选择并立即验证是否选择正确。"""

        del kwargs
        return apply_selection_event(controller, target)

    return {
        "place_dynamic": place_dynamic,
        "record_sugar_delivery": record_sugar,
        "lift_held": lift_held,
        "verify_pick_reachability": verify_pick,
        "verify_transport_hold": verify_hold,
        "verify_dining_placement": verify_placement,
        "verify_cups_ready": verify_ready,
        "select": select,
    }


def verify_current_pick_reachability(
    controller: Any,
    object_name: str,
    **kwargs: Any,
) -> dict[str, Any]:
    """使用物品当前位姿进行抓取入口规划预检查。"""

    if not controller.runtime_config.execute:
        return _dry_result("verify_current_pick_reachability", object_name)
    _refresh(controller)
    summary = controller.current_base_entry_reachability_summary(
        object_name,
        purpose="pick",
        holding=None,
        **kwargs,
    )
    passed = bool(summary.get("pass", False))
    return {
        "primitive": "verify_current_pick_reachability",
        "object_name": object_name,
        "reachability": summary,
        "checks": {"current_object_pose_reachable": passed},
        "pass": passed,
        "reason": None if passed else summary.get("reason", "pick_target_unreachable"),
    }


def verify_coffee_cup_held(
    controller: Any,
    coffee_id: str,
    *,
    max_end_effector_distance: float = 0.18,
    **kwargs: Any,
) -> dict[str, Any]:
    """通过接触、抓取阈值和末端距离判断杯子是否仍被持有。"""

    del kwargs
    if not controller.runtime_config.execute:
        return _dry_result("verify_coffee_cup_held_for_transport", coffee_id)
    env = _refresh(controller)
    hold = get_object_hold_status(
        env,
        coffee_id,
        max_end_effector_distance=float(max_end_effector_distance),
    )
    passed = bool(hold["held"])
    return {
        "primitive": "verify_coffee_cup_held_for_transport",
        **hold,
        "pass": passed,
        "reason": None if passed else "coffee_cup_not_held_for_transport",
    }


def verify_dining_placement(
    controller: Any,
    coffee_id: str,
    *,
    release_distance: float = 0.12,
    dining_target_slot: str | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """检查单杯接触餐桌、保持竖直、进入计划区域且已释放。"""

    del kwargs
    if not controller.runtime_config.execute:
        result = _dry_result("verify_coffee_cup_dining_placement", coffee_id)
        result["dining_target_slot"] = dining_target_slot
        return result
    env = _refresh(controller)
    slot = str(
        dining_target_slot
        or env.planned_dining_slot_by_coffee.get(coffee_id)
        or ""
    )
    checks = {
        "correctly_placed": bool(
            env.is_coffee_cup_correctly_placed_on_dining_counter(coffee_id)
        ),
        "inside_planned_region": bool(
            slot and env.is_coffee_cup_center_inside_dining_region(coffee_id, slot)
        ),
        "released": is_object_released(
            env,
            coffee_id,
            distance=float(release_distance),
        ),
    }
    passed = all(checks.values())
    return {
        "primitive": "verify_coffee_cup_dining_placement",
        "object_name": coffee_id,
        "dining_target_slot": slot,
        "placement_state": {
            "on_dining_counter": bool(
                getattr(env, f"{coffee_id}_on_dining_counter", False)
            ),
            "upright": bool(getattr(env, f"{coffee_id}_upright", False)),
        },
        "release_distance": float(release_distance),
        "checks": checks,
        "pass": passed,
        "reason": None if passed else "invalid_dining_placement",
    }


def verify_cups_ready_for_selection(controller: Any, **kwargs: Any) -> dict[str, Any]:
    """检查加糖与双杯条件，并在成立时锁存选杯前状态。"""

    del kwargs
    if not controller.runtime_config.execute:
        return _dry_result("verify_coffee_cups_ready_for_selection")
    env = _refresh(controller)
    checks = {
        "dining_regions_occupied": bool(env.dining_regions_occupied),
        "cups_together": bool(env.cups_together),
        "sugar_consumed": bool(env.sugar_consumed),
        "sugar_assignment_correct": bool(env.sugar_assignment_correct),
    }
    if all(checks.values()):
        env.record_cups_together_before_selection()
    checks["cups_together_before_selection_recorded"] = bool(
        env.cups_together_before_selection
    )
    passed = all(checks.values())
    return {
        "primitive": "verify_coffee_cups_ready_for_selection",
        "checks": checks,
        "pass": passed,
        "reason": None if passed else "coffee_cups_not_ready_for_selection",
    }


def apply_sugar_delivery_event(
    controller: Any,
    sugar_name: str,
    *,
    coffee_id: str,
    accept: bool,
) -> dict[str, Any]:
    """把成功的糖块投递转换为一次且仅一次的任务语义事件。"""

    env = controller.env
    already_consumed = bool(env.sugar_consumed)
    applied = bool(accept) and not already_consumed
    if applied:
        env.record_sugar_delivery(coffee_id)
    env = _refresh(controller)
    checks = {
        "sugar_consumed": bool(env.sugar_consumed),
        "sugar_assignment_correct": bool(env.sugar_assignment_correct),
    }
    passed = all(checks.values())
    return {
        "primitive": "record_sugar_delivery",
        "sugar_name": sugar_name,
        "coffee_id": coffee_id,
        "accept": bool(accept),
        "already_consumed": already_consumed,
        "applied": applied,
        "checks": checks,
        "pass": passed,
        "reason": None if passed else "sugar_delivery_not_recorded",
    }


def apply_selection_event(controller: Any, coffee_id: str) -> dict[str, Any]:
    """记录最终选杯并返回选择记录与正确性检查。"""

    env = controller.env
    env.record_sugared_coffee_selection(coffee_id)
    env = _refresh(controller)
    checks = {
        "selection_recorded": env.selected_sugared_coffee_id == coffee_id,
        "selection_correct": bool(env.correct_sugared_selection),
    }
    passed = all(checks.values())
    return {
        "primitive": "record_sugared_coffee_selection",
        "coffee_id": coffee_id,
        "selected_sugared_coffee_id": env.selected_sugared_coffee_id,
        "checks": checks,
        "pass": passed,
        "reason": None if passed else "incorrect_sugared_coffee_selection",
    }


def _refresh(controller: Any) -> Any:
    """同步 controller observation，并刷新环境派生物理状态。"""

    controller.refresh_obs()
    controller.env.update_state()
    return controller.env


def _dry_result(primitive: str, object_name: str | None = None) -> dict[str, Any]:
    """为 execute=false 模式生成延迟到真实执行的通过结果。"""

    result: dict[str, Any] = {
        "primitive": primitive,
        "dry_run": True,
        "checks": {"deferred_to_execute": True},
        "pass": True,
    }
    if object_name is not None:
        result["object_name"] = object_name
    return result


__all__ = [
    "apply_selection_event",
    "apply_sugar_delivery_event",
    "build_stage_handlers",
    "verify_coffee_cup_held",
    "verify_cups_ready_for_selection",
    "verify_current_pick_reachability",
    "verify_dining_placement",
]
