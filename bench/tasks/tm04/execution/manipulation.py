"""TM04 hybrid manipulation assembly for water, cabinet, can, and sugar."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from bench.execution.manipulation import ManipulationController
from bench.tasks.tm02.execution.completion_button import press_completion_button_physically
from bench.tasks.tm02.execution.manipulation import (
    ManipulationAssembly,
    build_manipulation_assembly as build_tm02_manipulation_assembly,
)
from bench.tasks.ts03.execution.manipulation import _install_held_sugar_follow_assist

from ..domain.contracts import COMPLETION_BUTTON


LOCAL_OPERATIONS = {
    "open_tm04_cabinet",
    "close_tm04_cabinet",
    "pick_sugar",
    "place_sugar_in_hot_chocolate",
    "press_completion_button",
}


def _build_saucepan_release_candidate_groups(
    anchor_world: Any,
    *,
    release_offsets_xy: Any,
    yaw_offsets_deg: Any,
    sugar_step: int,
    max_radius_m: float,
) -> list[dict[str, Any]]:
    """Build deterministic offset fallbacks with absolute top-down yaws."""

    anchor = np.asarray(anchor_world, dtype=float).reshape(-1)
    offsets = np.asarray(release_offsets_xy, dtype=float)
    yaws = tuple(float(value) for value in yaw_offsets_deg)
    if anchor.size != 3 or not np.all(np.isfinite(anchor)):
        raise ValueError("TM04 saucepan anchor must be a finite xyz position")
    if (
        offsets.ndim != 2
        or offsets.shape[0] == 0
        or offsets.shape[1] != 2
        or not np.all(np.isfinite(offsets))
    ):
        raise ValueError("TM04 sugar release offsets must be finite XY pairs")
    if not yaws or not all(np.isfinite(value) for value in yaws):
        raise ValueError("TM04 placement yaw offsets must be finite and non-empty")
    radius = float(max_radius_m)
    if not np.isfinite(radius) or radius <= 0.0:
        raise ValueError("TM04 release target radius must be positive")

    start = (max(1, int(sugar_step)) - 1) % len(offsets)
    ordered_offsets = np.concatenate((offsets[start:], offsets[:start]), axis=0)
    groups: list[dict[str, Any]] = []
    fixed_order = 0
    for fallback_rank, offset in enumerate(ordered_offsets):
        candidates = []
        inside = bool(float(np.linalg.norm(offset)) <= radius + 1e-9)
        for yaw_index, yaw in enumerate(yaws):
            candidates.append(
                {
                    "target_pos_world": anchor.astype(float).tolist(),
                    "release_offset_x": float(offset[0]),
                    "release_offset_y": float(offset[1]),
                    "placement_base_yaw_deg": float(yaw),
                    "release_target_inside_saucepan": inside,
                    "metadata": {
                        "fallback_rank": int(fallback_rank),
                        "yaw_index": int(yaw_index),
                        "yaw_mode": "base_cardinal_topdown",
                        "release_offset_xy": offset.astype(float).tolist(),
                        "fixed_candidate_order": int(fixed_order),
                    },
                }
            )
            fixed_order += 1
        groups.append(
            {
                "group_name": f"release_offset_{fallback_rank}",
                "fallback_rank": int(fallback_rank),
                "candidates": candidates,
            }
        )
    return groups


def build_manipulation_assembly(
    env: Any,
    obs: dict[str, Any],
    *,
    settings: Mapping[str, Any],
    planner_script: Path,
    execute: bool,
    missing_resources: tuple[str, ...],
    resources: Mapping[str, Path],
) -> ManipulationAssembly:
    assembly = build_tm02_manipulation_assembly(
        env,
        obs,
        settings=settings,
        planner_script=planner_script,
        execute=execute,
        missing_resources=missing_resources,
        resources=resources,
    )
    controller = assembly.controller
    handlers = controller.operation_handlers
    handlers["open_tm04_cabinet"] = handlers["open_tm02_cabinet"]
    handlers["close_tm04_cabinet"] = handlers["close_tm02_cabinet"]

    def pick_sugar(target: str, **kwargs: Any) -> dict[str, Any]:
        values = dict(kwargs)
        values.pop("require_physical_grasp", None)
        return controller.pick(str(target), **values)

    def place_sugar(target: str, **kwargs: Any) -> dict[str, Any]:
        values = dict(kwargs)
        values.pop("require_release", None)
        values.pop("target_receptacle", None)
        sugar_step = int(values.pop("sugar_step", 1))
        yaw_offsets = values.pop("placement_yaw_offsets_deg")
        release_offsets = values.pop("release_offsets_xy")
        max_radius = float(values.pop("release_target_max_radius_m"))
        target_world = np.asarray(
            env.object_world_position(env.hot_chocolate_name), dtype=float
        )
        candidate_groups = _build_saucepan_release_candidate_groups(
            target_world,
            release_offsets_xy=release_offsets,
            yaw_offsets_deg=yaw_offsets,
            sugar_step=sugar_step,
            max_radius_m=max_radius,
        )

        def check_saucepan_before_release() -> dict[str, Any]:
            stable = bool(env.hot_chocolate_stable())
            return {
                "pass": stable,
                "reason": None if stable else "hot_chocolate_saucepan_moved_before_release",
            }

        result = controller.place_at_world_pose_candidates(
            str(target),
            candidate_groups=candidate_groups,
            target_name=env.hot_chocolate_name,
            pre_release_check=check_saucepan_before_release,
            **values,
        )
        result["live_receptacle_anchor_world"] = target_world.astype(float).tolist()
        result["sugar_step"] = sugar_step
        if not controller.runtime_config.execute:
            result["physical_checks"] = {
                "skipped": True,
                "reason": "dry_run_has_no_object_side_effects",
            }
            return result
        controller.refresh_obs()
        physical_checks = {
            "sugar_in_hot_chocolate": bool(
                env.sugars_in_hot_chocolate().get(str(target), False)
            ),
            "sugar_released": bool(env.sugar_released(str(target))),
            "hot_chocolate_stable": bool(env.hot_chocolate_stable()),
        }
        result["physical_checks"] = physical_checks
        result["pass"] = bool(result.get("pass", False) and all(physical_checks.values()))
        if not result["pass"] and result.get("reason") is None:
            result["reason"] = "tm04_sugar_delivery_physical_validation_failed"
        return result

    def press_finish(target: str, **kwargs: Any) -> dict[str, Any]:
        if str(target) != COMPLETION_BUTTON:
            return {
                "primitive": "tm04_completion_button_press",
                "pass": False,
                "reason": "unknown_tm04_completion_button",
            }
        if not controller.runtime_config.execute:
            values = dict(kwargs)
            press_depth = float(
                values.pop(
                    "press_depth_threshold_m",
                    values.pop("minimum_press_displacement_m", 0.004),
                )
            )
            return press_completion_button_physically(
                controller,
                button_name=COMPLETION_BUTTON,
                prepress_z_offset=float(values.pop("prepress_z_offset", 0.08)),
                press_z_offset=float(values.pop("press_z_offset", -0.010)),
                hold_steps=int(values.pop("hold_steps", 12)),
                gripper_action=float(values.pop("gripper_action", 1.0)),
                near_miss_position_tolerance=float(
                    values.pop("near_miss_position_tolerance", 0.09)
                ),
                press_depth_threshold=press_depth,
                require_physical_button_motion=bool(
                    values.pop("require_physical_button_motion", True)
                ),
            )
        state = env.semantic_state
        preconditions = {
            "route_core_complete": len(state.event_history) == 6,
            "exactly_two_sugar_events": len(state.completed_sugar_names) == 2,
            "faucet_off": not env.faucet_is_on(),
            "pot_filled": env.pot_is_filled(),
            "can_released_in_place_region": env.can_in_place_region() and env.can_released(),
            "cabinet_closed": env.cabinet_open_fraction() <= env.final_closed_threshold,
            "exactly_two_sugars_in_hot_chocolate": sum(env.sugars_in_hot_chocolate().values()) == 2,
            "no_absorbing_failure": not state.absorbing_failure,
        }
        if not all(preconditions.values()):
            return {
                "primitive": "tm04_completion_button_press",
                "preconditions": preconditions,
                "pass": False,
                "reason": "tm04_completion_preconditions_not_met",
            }
        values = dict(kwargs)
        press_depth = float(
            values.pop(
                "press_depth_threshold_m",
                values.pop("minimum_press_displacement_m", 0.004),
            )
        )
        actuation = press_completion_button_physically(
            controller,
            button_name=COMPLETION_BUTTON,
            prepress_z_offset=float(values.pop("prepress_z_offset", 0.08)),
            press_z_offset=float(values.pop("press_z_offset", -0.010)),
            hold_steps=int(values.pop("hold_steps", 12)),
            gripper_action=float(values.pop("gripper_action", 1.0)),
            near_miss_position_tolerance=float(
                values.pop("near_miss_position_tolerance", 0.09)
            ),
            press_depth_threshold=press_depth,
            require_physical_button_motion=bool(
                values.pop("require_physical_button_motion", True)
            ),
        )
        result = env.task_diagnostics()
        return {
            "primitive": "tm04_completion_button_press",
            "preconditions": preconditions,
            "actuation": actuation,
            "task_diagnostics": result,
            "pass": bool(actuation.get("pass", False) and result.get("success", False)),
            "reason": None if result.get("success", False) else "tm04_completion_press_failed",
        }

    handlers["pick_sugar"] = pick_sugar
    handlers["place_sugar_in_hot_chocolate"] = place_sugar
    handlers["press_completion_button"] = press_finish
    original_execute = controller.execute

    def execute(operation: str, target: str, **kwargs: Any):
        op = str(getattr(operation, "value", operation))
        if op in LOCAL_OPERATIONS:
            return ManipulationController.execute(controller, op, target, **kwargs)
        return original_execute(op, target, **kwargs)

    controller.execute = execute  # type: ignore[method-assign]
    _install_held_sugar_follow_assist(env, controller)
    return assembly


__all__ = ["LOCAL_OPERATIONS", "build_manipulation_assembly"]
