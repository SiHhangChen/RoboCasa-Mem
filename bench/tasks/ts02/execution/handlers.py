"""Thin adapters from TS02 workflow operations to physical and semantic events."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from bench.tasks.ts02.domain.semantics import (
    close_precondition_checks,
    completion_button_precondition_checks,
    open_precondition_checks,
)
from bench.tasks.ts02.execution.completion_button import (
    press_completion_button_physically,
)
from bench.tasks.ts02.execution.manipulation import (
    DrawerPullConfig,
    DrawerPushCloseConfig,
    close_drawer_physically,
    locked_base_post_step_callback,
    open_drawer_physically,
)


def build_stage_handlers(
    controller: Any,
) -> dict[str, Callable[..., dict[str, Any]]]:
    return {
        "open_fixture": lambda target, **kwargs: open_fixture(
            controller, str(target), **kwargs
        ),
        "inspect_fixture": lambda target, **kwargs: inspect_fixture(
            controller, str(target), **kwargs
        ),
        "close_fixture": lambda target, **kwargs: close_fixture(
            controller, str(target), **kwargs
        ),
        "press_completion_button": lambda target, **kwargs: press_completion_button(
            controller, str(target), **kwargs
        ),
    }


def open_fixture(
    controller: Any,
    fixture_name: str,
    *,
    actuation_mode: str = "semantic",
    open_fraction: float = 1.0,
    enforce_proximity: bool = True,
    max_base_distance: float = 1.0,
    hook_orientation_blend: float = 0.90,
    handle_side_offset: float = 0.0,
    handle_vertical_offset: float = 0.0,
    prehook_distance: float = 0.125,
    hook_offset: float = 0.005,
    pull_distance: float = 0.125,
    open_fraction_threshold: float = 0.70,
    gripper_steps: int = 25,
    grasp_pose_file: str | None = None,
    close_gripper_on_hook: bool = False,
    close_gripper_action: float = 1.0,
    close_gripper_steps: int = 25,
    **kwargs: Any,
) -> dict[str, Any]:
    del kwargs
    env = controller.env
    preconditions = open_precondition_checks(env.semantic_state, fixture_name)
    if not all(preconditions.values()):
        return env.record_open_event(fixture_name, drawer_open=False)
    if bool(enforce_proximity):
        distance = float(env.robot_base_distance_to_slot(fixture_name))
        if distance > float(max_base_distance):
            result = env.record_open_event(
                fixture_name,
                drawer_open=False,
                failure_reason="fixture_not_reachable_from_base",
            )
            result["proximity"] = {
                "base_distance": distance,
                "max_base_distance": float(max_base_distance),
            }
            return result

    mode = str(actuation_mode).strip().lower()
    if mode == "semantic":
        env.set_slot_open_fraction(fixture_name, float(open_fraction))
        actuation = {
            "primitive": "semantic_drawer_open",
            "fixture": fixture_name,
            "target_fraction": float(open_fraction),
            "pass": True,
        }
    elif mode == "contact":
        actuation = open_drawer_physically(
            controller,
            fixture_name=fixture_name,
            config=DrawerPullConfig(
                hook_orientation_blend=float(hook_orientation_blend),
                handle_side_offset=float(handle_side_offset),
                handle_vertical_offset=float(handle_vertical_offset),
                prehook_distance=float(prehook_distance),
                hook_offset=float(hook_offset),
                pull_distance=float(pull_distance),
                open_fraction_threshold=float(open_fraction_threshold),
                gripper_steps=int(gripper_steps),
                grasp_pose_file=grasp_pose_file,
                close_gripper_on_hook=bool(close_gripper_on_hook),
                close_gripper_action=float(close_gripper_action),
                close_gripper_steps=int(close_gripper_steps),
            ),
        )
    else:
        raise ValueError(f"unsupported TS02 drawer-open mode: {actuation_mode!r}")
    fraction = float(env.slot_open_fraction(fixture_name))
    physically_open = bool(
        actuation.get("pass", False) and fraction >= float(env.open_threshold)
    )
    result = env.record_open_event(
        fixture_name,
        drawer_open=physically_open,
        failure_reason=(
            None
            if physically_open
            else str(actuation.get("reason") or "drawer_not_open")
        ),
    )
    result.update(
        {
            "actuation_mode": mode,
            "actuation": actuation,
            "drawer_open_fraction": fraction,
        }
    )
    return result


def inspect_fixture(
    controller: Any,
    fixture_name: str,
    *,
    dwell_steps: int = 0,
    **kwargs: Any,
) -> dict[str, Any]:
    del kwargs
    env = controller.env
    known = fixture_name in env.fixture_names
    fraction = float(env.slot_open_fraction(fixture_name)) if known else 0.0
    result = env.record_inspection_event(
        fixture_name,
        drawer_open=known and fraction >= float(env.open_threshold),
    )
    result.update(
        {
            "dwell_steps": int(dwell_steps),
            "drawer_open_fraction": fraction,
            "inspection_event_only": True,
        }
    )
    return result


def close_fixture(
    controller: Any,
    fixture_name: str,
    *,
    actuation_mode: str = "semantic",
    closed_fraction: float = 0.0,
    close_overshoot: float = 0.005,
    closed_fraction_threshold: float = 0.08,
    return_arm_after_close: bool = False,
    **kwargs: Any,
) -> dict[str, Any]:
    del kwargs
    env = controller.env
    preconditions = close_precondition_checks(env.semantic_state, fixture_name)
    if not all(preconditions.values()):
        known = fixture_name in env.fixture_names
        fraction = float(env.slot_open_fraction(fixture_name)) if known else 1.0
        return env.record_close_event(
            fixture_name,
            drawer_closed=known and fraction <= float(env.closed_threshold),
        )

    mode = str(actuation_mode).strip().lower()
    if mode == "semantic":
        env.set_slot_open_fraction(fixture_name, float(closed_fraction))
        actuation = {
            "primitive": "semantic_drawer_close",
            "fixture": fixture_name,
            "target_fraction": float(closed_fraction),
            "pass": True,
        }
    elif mode == "contact":
        actuation = close_drawer_physically(
            controller,
            fixture_name=fixture_name,
            config=DrawerPushCloseConfig(
                close_overshoot=float(close_overshoot),
                closed_fraction_threshold=float(closed_fraction_threshold),
            ),
        )
    else:
        raise ValueError(f"unsupported TS02 drawer-close mode: {actuation_mode!r}")
    fraction = float(env.slot_open_fraction(fixture_name))
    physically_closed = bool(
        actuation.get("pass", False) and fraction <= float(env.closed_threshold)
    )
    result = env.record_close_event(
        fixture_name,
        drawer_closed=physically_closed,
        failure_reason=(
            None
            if physically_closed
            else str(actuation.get("reason") or "drawer_not_closed")
        ),
    )
    result.update(
        {
            "actuation_mode": mode,
            "actuation": actuation,
            "drawer_open_fraction": fraction,
        }
    )
    if (
        result["pass"]
        and bool(return_arm_after_close)
        and bool(getattr(controller.runtime_config, "execute", False))
    ):
        arm = controller.return_arm_to_regular_pose(
            stage="ts02_post_close_return_to_regular_pose",
            gripper_action=-1.0,
            post_step_callback=locked_base_post_step_callback(env),
        )
        result["arm_reset"] = arm
        result["checks"]["arm_reset"] = bool(arm.get("pass", False))
        result["pass"] = bool(all(result["checks"].values()))
        if not result["pass"]:
            result["reason"] = str(
                arm.get("reason") or arm.get("failure_reason") or "arm_reset_failed"
            )
    return result


def press_completion_button(
    controller: Any,
    target: str,
    *,
    prepress_z_offset: float = 0.08,
    press_z_offset: float = -0.010,
    hold_steps: int = 12,
    gripper_action: float = 1.0,
    near_miss_position_tolerance: float = 0.09,
    press_depth_threshold: float = 0.004,
    require_physical_button_motion: bool = True,
    **kwargs: Any,
) -> dict[str, Any]:
    """Execute the final physical press, then commit the completion event."""

    del kwargs
    if str(target) != "completion_button":
        return {
            "primitive": "press_completion_button",
            "target": str(target),
            "pass": False,
            "reason": "unknown_completion_button_target",
        }
    env = controller.env
    all_closed = _all_drawers_closed(env)
    preconditions = completion_button_precondition_checks(
        env.semantic_state,
        all_drawers_closed=all_closed,
    )
    if not all(preconditions.values()):
        return env.record_completion_button_event(
            button_physically_pressed=False,
            all_drawers_closed=all_closed,
        )
    actuation = press_completion_button_physically(
        controller,
        prepress_z_offset=float(prepress_z_offset),
        press_z_offset=float(press_z_offset),
        hold_steps=int(hold_steps),
        gripper_action=float(gripper_action),
        near_miss_position_tolerance=float(near_miss_position_tolerance),
        press_depth_threshold=float(press_depth_threshold),
        require_physical_button_motion=bool(require_physical_button_motion),
    )
    if bool(actuation.get("dry_run", False)):
        return {
            **actuation,
            "target": "completion_button",
            "semantic_completion": {"deferred_to_execute": True},
        }
    physical_motion = bool(
        actuation.get("physical_button_motion_observed", False)
        and actuation.get("pass", False)
    )
    semantic = env.record_completion_button_event(
        button_physically_pressed=physical_motion,
        all_drawers_closed=_all_drawers_closed(env),
        failure_reason=str(
            actuation.get("reason") or "completion_button_not_physically_pressed"
        ),
    )
    checks = {
        "physical_press_operation": bool(actuation.get("pass", False)),
        "physical_button_motion": physical_motion,
        "semantic_completion_committed": bool(semantic.get("pass", False)),
    }
    return {
        "primitive": "press_completion_button",
        "target": "completion_button",
        "actuation": actuation,
        "semantic_completion": semantic,
        "checks": checks,
        "pass": bool(all(checks.values())),
        "reason": (
            None
            if all(checks.values())
            else str(
                semantic.get("reason")
                or actuation.get("reason")
                or "completion_button_press_failed"
            )
        ),
    }


def _all_drawers_closed(env: Any) -> bool:
    if hasattr(env, "all_drawers_closed"):
        return bool(env.all_drawers_closed())
    return all(
        float(env.slot_open_fraction(name)) <= float(env.closed_threshold)
        for name in env.fixture_names
    )


__all__ = [
    "build_stage_handlers",
    "close_fixture",
    "inspect_fixture",
    "open_fixture",
    "press_completion_button",
]
