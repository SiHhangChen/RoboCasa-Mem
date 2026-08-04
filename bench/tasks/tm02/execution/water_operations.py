"""Physical TM02 water-button, fill, and service-placement primitives.

No function in this module writes task history.  Callers receive concrete
motion and physical-query evidence; ``handlers.py`` is responsible for making
the corresponding semantic event only after that evidence passes.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from bench.execution.manipulation.object_interactions import get_object_hold_status
TEMPERATURES = ("hot", "cold")
CUP_IDS = ("cup_A", "cup_B")
SERVICE_SLOTS = ("slot_1", "slot_2")
from bench.tasks.tm02.execution.water_geometry import (
    button_press_pose_world,
    current_held_or_object_position_world,
    load_anchor_pose,
    pose_world_to_base,
    resolve_held_fill_pose,
    service_slot_position_world,
)


def operate_water_button(
    controller: Any,
    *,
    mode: str,
    temperature: str,
    prepress_z_offset: float,
    press_z_offset: float,
    hold_steps: int,
    gripper_action: float,
    near_miss_position_tolerance: float,
    require_physical_button_motion: bool = True,
    control_mode: str = "buttons",
    faucet_switch_pose_file: str | None = None,
    **_unused: Any,
) -> dict[str, Any]:
    """Physically press a bench-owned temperature button, then set the fixture.

    The added MuJoCo button is a physical interaction target but is not wired
    to RoboCasa's faucet internals.  The bridge to ``set_verified_faucet_state``
    is therefore permitted only after an executed press reaches the calibrated
    target.  It is deliberately not a semantic fill or selection proxy.
    """

    normalized_mode = str(mode).strip().lower()
    normalized_temp = str(temperature).strip().lower()
    if normalized_mode not in {"on", "off"}:
        return _failure("operate_water_button", "invalid_faucet_mode", mode=mode)
    if normalized_temp not in TEMPERATURES:
        return _failure(
            "operate_water_button", "invalid_water_temperature", temperature=temperature
        )
    if str(control_mode).strip().lower() not in {
        "button",
        "buttons",
        "water_button",
        "water_buttons",
    }:
        return _failure(
            "operate_water_button",
            "unsupported_water_control_mode",
            control_mode=control_mode,
        )
    if int(hold_steps) < 1:
        return _failure("operate_water_button", "button_hold_steps_must_be_positive")
    if not bool(controller.runtime_config.execute):
        return _dry_result(
            "operate_water_button",
            mode=normalized_mode,
            temperature=normalized_temp,
        )

    env = controller.env
    try:
        controller.refresh_obs()
        button = np.asarray(env.water_button_position(normalized_temp), dtype=float)
        prepress_pose = button_press_pose_world(
            controller.obs,
            button_position_world=button,
            z_offset=float(prepress_z_offset),
        )
        press_pose = button_press_pose_world(
            controller.obs,
            button_position_world=button,
            z_offset=float(press_z_offset),
        )
        faucet_calibration = None
        if faucet_switch_pose_file not in (None, "", "none"):
            _, payload = load_anchor_pose(str(faucet_switch_pose_file))
            faucet_calibration = {
                "file": str(faucet_switch_pose_file),
                "anchor": payload.get("anchor"),
            }
    except Exception as exc:
        return _failure(
            "operate_water_button",
            "water_button_target_unavailable",
            mode=normalized_mode,
            temperature=normalized_temp,
            error=f"{type(exc).__name__}: {exc}",
        )

    prepress = _run_world_stage(
        controller,
        prepress_pose,
        stage="water_button_prepress",
        gripper_action=float(gripper_action),
    )
    press = (
        _run_world_stage(
            controller,
            press_pose,
            stage="water_button_press",
            gripper_action=float(gripper_action),
            record_trajectory=True,
        )
        if bool(prepress.get("pass", False))
        else _skipped("water_button_press", "prepress_failed")
    )
    press_eef = np.asarray(
        controller.stage_executor.current_eef_site_world(), dtype=float
    )
    endpoint_error = float(np.linalg.norm(press_eef[:3, 3] - press_pose[:3, 3]))
    press_endpoint = endpoint_error <= float(near_miss_position_tolerance)
    hold = (
        controller.run_gripper_command(float(gripper_action), int(hold_steps))
        if bool(press.get("pass", False))
        else _skipped("water_button_hold", "press_failed")
    )
    retreat = (
        _run_world_stage(
            controller,
            prepress_pose,
            stage="water_button_retreat",
            gripper_action=float(gripper_action),
        )
        if _gripper_command_pass(hold, required_steps=int(hold_steps))
        else _skipped("water_button_retreat", "button_hold_failed")
    )
    motion_verified = bool(
        prepress.get("pass", False)
        and press.get("pass", False)
        and press_endpoint
        and _gripper_command_pass(hold, required_steps=int(hold_steps))
    )
    state_applied = False
    state_matches = False
    if motion_verified or not bool(require_physical_button_motion):
        try:
            env.set_verified_faucet_state(
                mode=normalized_mode,
                temperature=normalized_temp,
            )
            state_applied = True
            state_matches = bool(
                env.faucet_matches(
                    mode=normalized_mode,
                    temperature=normalized_temp if normalized_mode == "on" else None,
                )
            )
        except Exception:
            state_matches = False
    checks = {
        "prepress": bool(prepress.get("pass", False)),
        "press": bool(press.get("pass", False)),
        "press_endpoint": bool(press_endpoint),
        "button_hold": _gripper_command_pass(hold, required_steps=int(hold_steps)),
        "physical_button_motion": bool(motion_verified),
        "faucet_fixture_state": bool(state_matches),
        "retreat": bool(retreat.get("pass", False)),
    }
    if not bool(require_physical_button_motion):
        # The operation remains physical even in permissive calibration mode;
        # only this diagnostic requirement becomes non-blocking.
        checks["physical_button_motion"] = True
    return {
        "primitive": "operate_water_button",
        "mode": normalized_mode,
        "temperature": normalized_temp,
        "button_position_world": button.astype(float).tolist(),
        "prepress_target_world": prepress_pose.astype(float).tolist(),
        "press_target_world": press_pose.astype(float).tolist(),
        "press_endpoint_world": press_eef[:3, 3].astype(float).tolist(),
        "press_endpoint_error_m": endpoint_error,
        "near_miss_position_tolerance": float(near_miss_position_tolerance),
        "fixture_state_applied_after_verified_press": bool(state_applied),
        "faucet_switch_calibration": faucet_calibration,
        "stages": {
            "prepress": _without_obs(prepress),
            "press": _without_obs(press),
            "hold": _without_obs(hold),
            "retreat": _without_obs(retreat),
        },
        "checks": checks,
        "pass": bool(all(checks.values())),
        "reason": None if all(checks.values()) else "water_button_operation_failed",
    }


def fill_held_cup_under_faucet(
    controller: Any,
    *,
    cup_id: str,
    temperature: str,
    hold_steps: int,
    gripper_action: float,
    cup_z_offset_from_water: float,
    entry_z_offset: float,
    xy_near_water_tolerance: float,
    sink_fill_pose_file: str | None = None,
    water_site_offset_x: float = 0.0,
    water_site_offset_y: float = 0.0,
    require_physical_fill: bool = True,
    **_unused: Any,
) -> dict[str, Any]:
    """Move an actually held cup through the live water stream and wait.

    The result contains no fill event.  It verifies faucet state, actual grasp,
    cup-under-water geometry, and dwell time so the handler can safely commit
    ``record_verified_fill`` afterward.
    """

    if cup_id not in CUP_IDS:
        return _failure("fill_held_cup_under_faucet", "unknown_cup", cup_id=cup_id)
    normalized_temp = str(temperature).strip().lower()
    if normalized_temp not in TEMPERATURES:
        return _failure(
            "fill_held_cup_under_faucet",
            "invalid_water_temperature",
            temperature=temperature,
        )
    if int(hold_steps) < 1:
        return _failure("fill_held_cup_under_faucet", "fill_hold_steps_must_be_positive")
    if not bool(controller.runtime_config.execute):
        return _dry_result(
            "fill_held_cup_under_faucet",
            cup_id=cup_id,
            temperature=normalized_temp,
        )

    env = controller.env
    try:
        controller.refresh_obs()
        faucet_ok = bool(env.faucet_matches(mode="on", temperature=normalized_temp))
        hold_before = get_object_hold_status(env, cup_id)
        fill = resolve_held_fill_pose(
            controller,
            cup_id,
            cup_z_offset_from_water=float(cup_z_offset_from_water),
            water_site_offset_x=float(water_site_offset_x),
            water_site_offset_y=float(water_site_offset_y),
            sink_fill_pose_file=sink_fill_pose_file,
        )
    except Exception as exc:
        return _failure(
            "fill_held_cup_under_faucet",
            "fill_target_unavailable",
            cup_id=cup_id,
            temperature=normalized_temp,
            error=f"{type(exc).__name__}: {exc}",
        )

    fill_pose = np.asarray(fill["target_eef_world"], dtype=float)
    entry_pose = np.array(fill_pose, dtype=float, copy=True)
    entry_pose[:3, 3] += np.asarray([0.0, 0.0, float(entry_z_offset)], dtype=float)
    callback_factory = getattr(env, "water_stream_intercept_callback", None)
    stream_callback = (
        callback_factory(cup_id) if callable(callback_factory) else None
    )

    entry = _run_world_stage(
        controller,
        entry_pose,
        stage="held_fill_entry",
        gripper_action=float(gripper_action),
        holding_object=cup_id,
        post_step_callback=stream_callback,
    )
    descend = (
        _run_world_stage(
            controller,
            fill_pose,
            stage="held_fill_under_faucet",
            gripper_action=float(gripper_action),
            holding_object=cup_id,
            record_trajectory=True,
            post_step_callback=stream_callback,
        )
        if bool(entry.get("pass", False))
        else _skipped("held_fill_under_faucet", "entry_failed")
    )
    cup_under_water = _cup_is_under_water(env, cup_id)
    cup_position_at_fill = _actual_cup_position(env, controller, cup_id)
    water_site = np.asarray(fill["water_site_world"], dtype=float)
    xy_error = float(np.linalg.norm(cup_position_at_fill[:2] - water_site[:2]))
    cup_xy_near_water = xy_error <= float(xy_near_water_tolerance)
    hold = (
        _run_fill_dwell(
            controller,
            env,
            cup_id=cup_id,
            gripper_action=float(gripper_action),
            hold_steps=int(hold_steps),
        )
        if bool(descend.get("pass", False))
        and bool(cup_under_water)
        and bool(faucet_ok)
        else _skipped("held_fill_hold", "cup_not_verified_under_water")
    )
    checks = {
        "faucet_on_at_requested_temperature": bool(faucet_ok),
        "cup_held_before_fill": bool(hold_before.get("held", False)),
        "entry": bool(entry.get("pass", False)),
        "under_water_motion": bool(descend.get("pass", False)),
        "cup_under_water": bool(cup_under_water),
        "fill_dwell": _gripper_command_pass(hold, required_steps=int(hold_steps)),
        "liquid_level_reached_full": bool(
            (hold.get("visual_checks") or {}).get("liquid_level_reached_full", False)
        ),
        "water_stream_intercepted_at_cup": bool(
            (hold.get("visual_checks") or {}).get(
                "water_stream_intercepted_at_cup", False
            )
        ),
        "post_fill_motion_deferred_to_holding_return": True,
    }
    if not bool(require_physical_fill):
        checks["cup_under_water"] = True
    passed = bool(all(checks.values()))
    if not passed:
        restore_stream = getattr(env, "set_water_stream_intercept", None)
        if callable(restore_stream):
            restore_stream(None)
    return {
        "primitive": "fill_held_cup_under_faucet",
        "cup_id": cup_id,
        "temperature": normalized_temp,
        "water_site_world": water_site.astype(float).tolist(),
        "desired_cup_world": np.asarray(fill["desired_cup_world"], dtype=float).tolist(),
        "cup_position_at_fill_world": cup_position_at_fill.astype(float).tolist(),
        "cup_xy_error_to_water_m": xy_error,
        "cup_xy_near_water": bool(cup_xy_near_water),
        "cup_xy_near_water_tolerance": float(xy_near_water_tolerance),
        "cup_under_water": bool(cup_under_water),
        "hold_steps": int(hold_steps),
        "calibration": fill.get("calibration"),
        "stages": {
            "entry": _without_obs(entry),
            "under_water": _without_obs(descend),
            "hold": _without_obs(hold),
        },
        "post_fill_arm_motion": "return_arm_holding_cup_next_workflow_stage",
        "water_stream_restore_deferred_to_holding_return": bool(passed),
        "checks": checks,
        "pass": passed,
        "reason": None if passed else "physical_fill_not_verified",
    }


def place_cup_on_service_slot(
    controller: Any,
    *,
    cup_id: str,
    slot: str,
    **kwargs: Any,
) -> dict[str, Any]:
    """Use the generic controller to place a held cup at one neutral slot.

    This operation intentionally does not record staging.  The next workflow
    stage performs an independent physical placement/release check before the
    semantic event is accepted.
    """

    if cup_id not in CUP_IDS:
        return _failure("place_cup_on_service_slot", "unknown_cup", cup_id=cup_id)
    if slot not in SERVICE_SLOTS:
        return _failure("place_cup_on_service_slot", "invalid_service_slot", slot=slot)
    if not bool(controller.runtime_config.execute):
        return _dry_result("place_cup_on_service_slot", cup_id=cup_id, slot=slot)
    try:
        target = service_slot_position_world(controller.env, slot)
    except Exception as exc:
        return _failure(
            "place_cup_on_service_slot",
            "service_slot_target_unavailable",
            cup_id=cup_id,
            slot=slot,
            error=f"{type(exc).__name__}: {exc}",
        )
    values = dict(kwargs)
    target_tolerance = float(values.pop("service_target_xy_tolerance", 0.08))
    release_distance = float(values.pop("release_distance", 0.12))
    require_pre_release_check = bool(values.pop("require_pre_release_check", True))
    pre_release_min_height = float(
        values.pop("pre_release_min_height_from_slot", 0.0)
    )
    pre_release_max_height = float(
        values.pop("pre_release_max_height_from_slot", 0.10)
    )
    separation_distance = float(
        values.pop("release_separation_assist_distance", 0.0)
    )
    if separation_distance < 0.0:
        return _failure(
            "place_cup_on_service_slot",
            "invalid_release_separation_assist_distance",
            cup_id=cup_id,
            slot=slot,
            release_separation_assist_distance=separation_distance,
        )
    values.pop("service_slot", None)
    # Enter vertically, then keep the ring exit outside the same horizontal
    # clearance radius.  A zero exit standoff would send the retreat back over
    # the cup after the dedicated clearance stage.
    values.setdefault("placement_approach_angles_deg", [0.0])
    values.setdefault("placement_entry_standoff", 0.0)
    values.setdefault(
        "placement_exit_standoff",
        float(values.get("post_release_clearance_distance", 0.0)),
    )
    values.setdefault(
        "retreat_height", float(values.get("placement_exit_z_offset", 0.22))
    )
    env = controller.env
    missing = object()
    previous_distance = getattr(env, "_wa02_release_separation_distance", missing)
    trace = {
        "enabled": bool(separation_distance > 0.0),
        "cup_id": cup_id,
        "distance_threshold_m": float(separation_distance),
        "captured": False,
        "pinned_steps": 0,
        "separated": False,
    }
    env._wa02_release_separation_distance = float(separation_distance)
    env._wa02_release_separation_trace = trace

    def pre_release_check() -> dict[str, Any]:
        controller.refresh_obs()
        cup_position = np.asarray(env.object_position(cup_id), dtype=float).reshape(3)
        xy_error = float(np.linalg.norm(cup_position[:2] - target[:2]))
        height_from_slot = float(cup_position[2] - target[2])
        cup_rotation = np.asarray(
            env.sim.data.body_xmat[env.obj_body_id[cup_id]], dtype=float
        ).reshape(3, 3)
        upright_z_axis = float(cup_rotation[2, 2])
        upright_threshold = float(
            np.cos(np.deg2rad(float(env.cup_upright_threshold_deg)))
        )
        try:
            from robocasa.utils import object_utils as OU

            counter_contact = bool(
                OU.check_obj_fixture_contact(env, cup_id, env.counter)
            )
        except Exception:
            counter_contact = False
        checks = {
            "cup_xy_at_service_slot": bool(xy_error <= target_tolerance),
            "cup_center_inside_service_region": bool(
                env.cup_is_in_service_region(cup_id, slot)
            ),
            "cup_near_counter_height": bool(
                pre_release_min_height <= height_from_slot <= pre_release_max_height
            ),
            "cup_upright": bool(upright_z_axis >= upright_threshold),
        }
        return {
            "pass": bool(all(checks.values())),
            "reason": (
                None if all(checks.values()) else "physical_pre_release_pose_not_verified"
            ),
            "checks": checks,
            "cup_position_world": cup_position.astype(float).tolist(),
            "slot_position_world": target.astype(float).tolist(),
            "slot_xy_error_m": xy_error,
            "height_from_slot_m": height_from_slot,
            "min_height_from_slot_m": pre_release_min_height,
            "max_height_from_slot_m": pre_release_max_height,
            "upright_z_axis": upright_z_axis,
            "upright_threshold": upright_threshold,
            "counter_contact": counter_contact,
        }

    try:
        result = controller.place_at_world_pose(
            cup_id,
            target_name=f"{cup_id}_{slot}_service_slot",
            target_pos_world=target.tolist(),
            pre_release_check=(pre_release_check if require_pre_release_check else None),
            **values,
        )
    finally:
        if previous_distance is missing:
            delattr(env, "_wa02_release_separation_distance")
        else:
            env._wa02_release_separation_distance = previous_distance
    result.update(
        {
            "primitive": "place_cup_on_service_slot",
            "cup_id": cup_id,
            "slot": slot,
            "service_slot_position_world": target.astype(float).tolist(),
            "service_target_xy_tolerance": target_tolerance,
            "release_distance": release_distance,
            "release_separation_assist": dict(trace),
            "semantic_event_deferred": "record_verified_staging",
        }
    )
    return result


def _run_world_stage(
    controller: Any,
    pose_world: np.ndarray,
    *,
    stage: str,
    gripper_action: float,
    holding_object: str | None = None,
    record_trajectory: bool = False,
    post_step_callback: Any | None = None,
) -> dict[str, Any]:
    return controller.run_planned_stage(
        pose_world_to_base(controller.obs, pose_world),
        stage=stage,
        gripper_action=float(gripper_action),
        holding_object=holding_object,
        record_trajectory=bool(record_trajectory),
        post_step_callback=post_step_callback,
    )


def _run_fill_dwell(
    controller: Any,
    env: Any,
    *,
    cup_id: str,
    gripper_action: float,
    hold_steps: int,
) -> dict[str, Any]:
    """Hold the cup under water while animating level and stream interception."""

    steps = int(hold_steps)
    if steps < 1:
        return _skipped("held_fill_hold", "fill_hold_steps_must_be_positive")
    progress_by_cup = getattr(env, "cup_fill_progress", {})
    start_progress = float(
        np.clip(float(dict(progress_by_cup).get(str(cup_id), 0.0)), 0.0, 1.0)
    )
    set_progress = getattr(env, "set_cup_fill_progress", None)
    set_intercept = getattr(env, "set_water_stream_intercept", None)
    intercept = (
        dict(set_intercept(str(cup_id)))
        if callable(set_intercept)
        else {"intercepted": False, "reason": "visual_api_unavailable"}
    )
    completed_steps = 0
    progress_trace: list[float] = []
    latest: dict[str, Any] = {}
    try:
        for index in range(steps):
            progress = start_progress + (1.0 - start_progress) * (
                float(index + 1) / float(steps)
            )
            if callable(set_progress):
                set_progress(str(cup_id), progress)
            latest = dict(
                controller.run_gripper_command(float(gripper_action), 1)
            )
            step_count = int(latest.get("steps", 0))
            completed_steps += step_count
            if step_count < 1:
                break
            progress_trace.append(float(progress))
    except Exception:
        if callable(set_intercept):
            set_intercept(None)
        raise

    final_progress = (
        float(progress_trace[-1]) if progress_trace else start_progress
    )
    visual_checks = {
        "liquid_level_reached_full": bool(
            callable(set_progress)
            and completed_steps >= steps
            and final_progress >= 1.0 - 1.0e-6
        ),
        "water_stream_intercepted_at_cup": bool(
            intercept.get("intercepted", False)
        ),
    }
    return {
        **_without_obs(latest),
        "stage": "held_fill_hold",
        "steps": int(completed_steps),
        "requested_steps": steps,
        "gripper_action": float(gripper_action),
        "start_progress": start_progress,
        "final_progress": final_progress,
        "progress_trace": progress_trace,
        "water_stream_intercept": intercept,
        "visual_checks": visual_checks,
        "pass": bool(
            completed_steps >= steps and all(visual_checks.values())
        ),
    }


def _cup_is_under_water(env: Any, cup_id: str) -> bool:
    try:
        return bool(env.sink.check_obj_under_water(env, cup_id))
    except Exception:
        return False


def _actual_cup_position(env: Any, controller: Any, cup_id: str) -> np.ndarray:
    try:
        return np.asarray(env.object_position(cup_id), dtype=float).reshape(3)
    except Exception:
        return current_held_or_object_position_world(controller, cup_id)


def _gripper_command_pass(result: dict[str, Any], *, required_steps: int) -> bool:
    if bool(result.get("dry_run", False)):
        return True
    return int(result.get("steps", 0)) >= int(required_steps)


def _skipped(stage: str, reason: str) -> dict[str, Any]:
    return {"stage": stage, "skipped": True, "pass": False, "reason": reason}


def _without_obs(value: dict[str, Any]) -> dict[str, Any]:
    return {key: item for key, item in value.items() if key != "obs"}


def _failure(primitive: str, reason: str, **details: Any) -> dict[str, Any]:
    return {"primitive": primitive, "pass": False, "reason": reason, **details}


def _dry_result(primitive: str, **details: Any) -> dict[str, Any]:
    return {
        "primitive": primitive,
        "dry_run": True,
        "semantic_event_deferred": True,
        "checks": {"deferred_to_execute": True},
        "pass": True,
        **details,
    }


__all__ = [
    "fill_held_cup_under_faucet",
    "operate_water_button",
    "place_cup_on_service_slot",
]
