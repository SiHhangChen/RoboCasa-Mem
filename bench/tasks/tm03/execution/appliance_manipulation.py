"""Physical native-stove-knob and drawer manipulation for TM03.

The knob is moved solely through gripper contact plus planned arm motion. The
two identical SingleTabletopDrawer fixtures reuse TM03's proven handle-relative
hook calibration and contact pull / push primitives.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from bench.execution.manipulation import ManipulationController, ManipulationRuntimeConfig
from bench.execution.contracts import StageKind, StageResult
from bench.execution.planning import (
    MplibPlannerConfig,
    matrix_from_base_obs,
    matrix_from_json_pose,
    matrix_from_site_pose,
)
from bench.execution.planning.model_alignment import pose_from_matrix
from bench.execution.planning.mplib_planner import run_mplib_planner_subprocess
from bench.tasks.tm03.execution.drawer_physics import (
    DrawerPullConfig,
    DrawerPushCloseConfig,
    close_drawer_physically as close_tm03_drawer_physically,
    open_drawer_physically as open_tm03_drawer_physically,
)
from bench.tasks.tm03.execution.completion_button import (
    press_completion_button_physically,
)
from bench.tasks.tm03.execution.drawer_geometry import resolve_drawer_motion_frame


TURN_ON_OPERATION = "turn_native_stove_knob_on"
TURN_OFF_OPERATION = "turn_native_stove_knob_off"
SUPPORTED_GENERIC_OPERATIONS = {"move_eef", "return_arm"}


@dataclass(frozen=True)
class ManipulationAssembly:
    controller: "TM03ManipulationController"
    planner_settings: Mapping[str, Any]


@dataclass(frozen=True)
class KnobInteractionCalibration:
    source: Path
    target_knob: str
    target_burner: str
    pose_files: Mapping[str, Path]
    turn_on_waypoints_deg: tuple[float, ...]
    turn_off_waypoints_deg: tuple[float, ...]
    open_steps: int
    close_steps: int
    release_steps: int
    off_qpos_max_rad: float
    on_qpos_min_rad: float


class TM03ManipulationController(ManipulationController):
    """Run calibrated physical knob stages and gate all uncalibrated actions."""

    def __init__(
        self,
        *args: Any,
        knob_calibration: KnobInteractionCalibration,
        missing_resources: tuple[str, ...],
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.knob_calibration = knob_calibration
        self.missing_resources = tuple(str(item) for item in missing_resources)

    def execute(self, operation: str, target: str, **kwargs: Any) -> StageResult:
        op = str(getattr(operation, "value", operation))
        if op == TURN_ON_OPERATION:
            raw = self._turn_native_knob(target, turn_on=True, **kwargs)
            return _stage_result(raw)
        if op == TURN_OFF_OPERATION:
            raw = self._turn_native_knob(target, turn_on=False, **kwargs)
            return _stage_result(raw)
        if op in SUPPORTED_GENERIC_OPERATIONS or op in self.operation_handlers:
            return super().execute(op, target, **kwargs)
        reason = "tm03_physical_manipulation_calibration_required"
        return StageResult(
            index=-1,
            kind=StageKind.OPERATE,
            success=False,
            details={
                "primitive": "tm03_calibration_gate",
                "operation": op,
                "target": str(target),
                "missing_resources": list(self.missing_resources),
                "semantic_shortcut_used": False,
                "pass": False,
                "reason": reason,
            },
            failure_code=reason,
        )

    def _turn_native_knob(
        self,
        target: str,
        *,
        turn_on: bool,
        **kwargs: Any,
    ) -> dict[str, Any]:
        del kwargs
        calibration = self.knob_calibration
        operation = TURN_ON_OPERATION if turn_on else TURN_OFF_OPERATION
        if str(target) != calibration.target_knob:
            return _knob_failure(
                operation,
                target,
                reason="tm03_knob_target_mismatch",
                expected_target=calibration.target_knob,
            )

        self.refresh_obs()
        initial_state = _live_knob_state(self.env)
        start_ok = (
            initial_state["knob_qpos_rad"] <= calibration.off_qpos_max_rad
            if turn_on
            else initial_state["knob_qpos_rad"] >= calibration.on_qpos_min_rad
        )
        if self.runtime_config.execute and not start_ok:
            return _knob_failure(
                operation,
                target,
                reason=(
                    "tm03_knob_not_off_at_turn_on_start"
                    if turn_on
                    else "tm03_knob_not_on_at_turn_off_start"
                ),
                initial_state=initial_state,
            )

        pose_role = "pinch_off" if turn_on else "pinch_on_for_off"
        approach_world, approach_info = _resolve_anchor_pose_world(
            self.env,
            calibration.pose_files[pose_role],
        )
        pinch_off_world, pinch_off_info = _resolve_anchor_pose_world(
            self.env,
            calibration.pose_files["pinch_off"],
        )
        waypoint_angles = (
            calibration.turn_on_waypoints_deg
            if turn_on
            else calibration.turn_off_waypoints_deg
        )
        if turn_on:
            endpoint_role = "pinch_on"
            endpoint_world, endpoint_info = _resolve_anchor_pose_world(
                self.env,
                calibration.pose_files[endpoint_role],
            )
        else:
            endpoint_role = f"pinch_off_overtravel_{_angle_label(waypoint_angles[-1])}deg"
            endpoint_world = _knob_rotation_pose(
                pinch_off_world,
                angle_deg=waypoint_angles[-1],
            )
            endpoint_info = {
                **pinch_off_info,
                "pose_name": endpoint_role,
                "derived_from": "pinch_off",
                "rotation_angle_deg": waypoint_angles[-1],
                "resolved_pose_world": endpoint_world.tolist(),
            }

        command_summaries: list[dict[str, Any]] = []
        move_summaries: list[dict[str, Any]] = []
        open_result = self.run_gripper_command(-1.0, calibration.open_steps)
        command_summaries.append({"command": "open", **_without_obs(open_result)})

        approach = self.run_planned_stage(
            _world_target_in_current_base(self, approach_world),
            stage=f"{operation}_approach",
            gripper_action=-1.0,
        )
        move_summaries.append(approach)
        if not bool(approach.get("pass", False)):
            return self._knob_motion_failure(
                operation=operation,
                target=target,
                initial_state=initial_state,
                approach_info=approach_info,
                endpoint_info=endpoint_info,
                command_summaries=command_summaries,
                move_summaries=move_summaries,
                reason="tm03_knob_approach_failed",
            )

        close_result = self.run_gripper_command(1.0, calibration.close_steps)
        command_summaries.append({"command": "close", **_without_obs(close_result)})
        executed_waypoint_angles: list[float] = []
        for index, angle_deg in enumerate(waypoint_angles):
            executed_waypoint_angles.append(float(angle_deg))
            target_world = _knob_rotation_pose(
                pinch_off_world,
                angle_deg=float(angle_deg),
            )
            if index == len(waypoint_angles) - 1:
                target_world = endpoint_world
            stage = self.run_planned_stage(
                _world_target_in_current_base(self, target_world),
                stage=f"{operation}_rotate_{_angle_label(angle_deg)}deg",
                gripper_action=1.0,
            )
            move_summaries.append(stage)
            if not bool(stage.get("pass", False)):
                contact_limit_state = _live_knob_state(self.env)
                stage_checks = dict(stage.get("checks") or {})
                accepted_contact_limit = bool(
                    not turn_on
                    and float(angle_deg) < 0.0
                    and stage_checks.get("plan_success", False)
                    and stage_checks.get("runtime_collision_free", False)
                    and contact_limit_state["knob_qpos_rad"]
                    <= calibration.off_qpos_max_rad
                    and not contact_limit_state["target_burner_on"]
                )
                if accepted_contact_limit:
                    stage["accepted_contact_limited_overtravel"] = {
                        "reason": "native_knob_reached_physical_off_stop",
                        "commanded_marker_angle_deg": float(angle_deg),
                        "knob_qpos_rad": contact_limit_state["knob_qpos_rad"],
                        "off_qpos_max_rad": calibration.off_qpos_max_rad,
                        "target_burner_on": contact_limit_state["target_burner_on"],
                    }
                    stage["pass"] = True
                    stage["failure_phase"] = None
                    stage["failure_reason"] = None
                    stage_checks["physical_knob_off_stop"] = True
                    stage_checks["contact_limited_overtravel_accepted"] = True
                    stage["checks"] = stage_checks
                    endpoint_role = (
                        "contact_limited_overtravel_"
                        f"{_angle_label(angle_deg)}deg"
                    )
                    endpoint_info = {
                        **endpoint_info,
                        "pose_name": endpoint_role,
                        "accepted_physical_state": contact_limit_state,
                    }
                    break
                release = self.run_gripper_command(-1.0, calibration.release_steps)
                command_summaries.append(
                    {"command": "release_after_failure", **_without_obs(release)}
                )
                return self._knob_motion_failure(
                    operation=operation,
                    target=target,
                    initial_state=initial_state,
                    approach_info=approach_info,
                    endpoint_info=endpoint_info,
                    command_summaries=command_summaries,
                    move_summaries=move_summaries,
                    reason="tm03_knob_rotation_waypoint_failed",
                )

        state_before_release = _live_knob_state(self.env)
        release_result = self.run_gripper_command(-1.0, calibration.release_steps)
        command_summaries.append({"command": "release", **_without_obs(release_result)})
        final_state = _live_knob_state(self.env)
        if self.runtime_config.execute:
            physical_target = (
                final_state["knob_qpos_rad"] >= calibration.on_qpos_min_rad
                and final_state["target_burner_on"]
                if turn_on
                else final_state["knob_qpos_rad"] <= calibration.off_qpos_max_rad
                and not final_state["target_burner_on"]
            )
        else:
            physical_target = True
        checks = {
            "valid_start_state": bool(start_ok) or not self.runtime_config.execute,
            "approach_planned_and_executed": bool(approach.get("pass", False)),
            "rotation_sequence_completed_or_physical_stop": all(
                bool(stage.get("pass", False)) for stage in move_summaries[1:]
            ),
            "physical_knob_target": bool(physical_target),
            "no_direct_knob_state_write": True,
            "no_knob_actuator_command": True,
        }
        diagnostics = final_state.get("task_diagnostics") or {}
        invalid_environment = list(diagnostics.get("invalid_environment_labels", ()))
        checks["environment_integrity"] = not invalid_environment
        passed = bool(all(checks.values()))
        return {
            "primitive": "pinch_and_rotate_native_knob",
            "operation": operation,
            "target": str(target),
            "target_burner": calibration.target_burner,
            "calibration_file": str(calibration.source),
            "pose_role": pose_role,
            "endpoint_role": endpoint_role,
            "pinch_on_reused_as_turn_off_start": not turn_on,
            "rotation_waypoints_deg": list(waypoint_angles),
            "executed_rotation_waypoints_deg": executed_waypoint_angles,
            "initial_state": initial_state,
            "state_before_release": state_before_release,
            "final_state": final_state,
            "approach_pose": approach_info,
            "pinch_off_pose": pinch_off_info,
            "endpoint_pose": endpoint_info,
            "gripper_commands": command_summaries,
            "stages": move_summaries,
            "checks": checks,
            "semantic_shortcut_used": False,
            "dry_run": not self.runtime_config.execute,
            "pass": passed,
            "reason": None if passed else "tm03_knob_physical_target_not_reached",
        }

    def _knob_motion_failure(
        self,
        *,
        operation: str,
        target: str,
        initial_state: Mapping[str, Any],
        approach_info: Mapping[str, Any],
        endpoint_info: Mapping[str, Any],
        command_summaries: list[dict[str, Any]],
        move_summaries: list[dict[str, Any]],
        reason: str,
    ) -> dict[str, Any]:
        return {
            "primitive": "pinch_and_rotate_native_knob",
            "operation": operation,
            "target": str(target),
            "initial_state": dict(initial_state),
            "final_state": _live_knob_state(self.env),
            "approach_pose": dict(approach_info),
            "endpoint_pose": dict(endpoint_info),
            "gripper_commands": command_summaries,
            "stages": move_summaries,
            "semantic_shortcut_used": False,
            "pass": False,
            "reason": reason,
        }


def build_manipulation_assembly(
    env: Any,
    obs: dict[str, Any],
    *,
    settings: Mapping[str, Any],
    planner_script: Path,
    interaction_poses_file: Path,
    execute: bool,
    missing_resources: tuple[str, ...],
) -> ManipulationAssembly:
    calibration = load_knob_interaction_calibration(interaction_poses_file)
    planner = _section(settings, "planner")
    controller = TM03ManipulationController(
        env,
        obs,
        planner=_make_planner(
            script_path=planner_script,
            conda_env=str(planner.get("conda_env", "membench")),
            timeout_s=float(planner.get("subprocess_timeout_s", 60.0)),
        ),
        runtime_config=_runtime_config(
            execute=execute,
            runtime=_section(settings, "runtime"),
            planner=planner,
            static_scene=_section(settings, "static_scene"),
        ),
        knob_calibration=calibration,
        missing_resources=missing_resources,
    )
    controller.operation_handlers = {
        "open_drawer_physically": lambda target, **kwargs: (
            _open_drawer_with_tm03_relative_pose(controller, str(target), **kwargs)
        ),
        "inspect_drawer": lambda target, **kwargs: _inspect_drawer(
            controller, str(target), **kwargs
        ),
        "close_drawer_physically": lambda target, **kwargs: (
            _close_drawer_with_tm03_contact(controller, str(target), **kwargs)
        ),
        "retreat_and_return_arm_after_drawer": lambda target, **kwargs: (
            _retreat_and_return_arm_after_drawer(
                controller, str(target), **kwargs
            )
        ),
        "press_completion_button": lambda target, **kwargs: (
            _press_completion_button(controller, str(target), **kwargs)
        ),
    }
    return ManipulationAssembly(controller=controller, planner_settings=planner)


def _open_drawer_with_tm03_relative_pose(
    controller: TM03ManipulationController,
    drawer: str,
    *,
    actuation_mode: str = "contact",
    hook_orientation_blend: float = 0.90,
    handle_side_offset: float = 0.0,
    handle_vertical_offset: float = 0.0,
    prehook_distance: float = 0.125,
    hook_offset: float = 0.005,
    pull_distance: float = 0.18,
    open_fraction_threshold: float = 0.70,
    gripper_steps: int = 25,
    grasp_pose_file: str | Path | None = None,
    close_gripper_on_hook: bool = True,
    close_gripper_action: float = 1.0,
    close_gripper_steps: int = 25,
    **kwargs: Any,
) -> dict[str, Any]:
    del kwargs
    if drawer not in tuple(getattr(controller.env, "drawer_names", ())):
        return _drawer_failure("physical_drawer_pull", drawer, "unknown_tm03_drawer")
    if str(actuation_mode).strip().lower() != "contact":
        return _drawer_failure(
            "physical_drawer_pull",
            drawer,
            "tm03_drawer_requires_contact_actuation",
        )
    result = open_tm03_drawer_physically(
        controller,
        fixture_name=drawer,
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
    result.update(
        {
            "operation": "open_drawer_physically",
            "relative_calibration_reused_from": "TM03",
            "relative_calibration_mode": "translate_template_to_live_handle",
            "semantic_shortcut_used": False,
            "task_diagnostics": controller.env.task_diagnostics(),
        }
    )
    return result


def _inspect_drawer(
    controller: TM03ManipulationController,
    drawer: str,
    *,
    dwell_steps: int = 10,
    **kwargs: Any,
) -> dict[str, Any]:
    del kwargs
    if drawer not in tuple(getattr(controller.env, "drawer_names", ())):
        return _drawer_failure("physical_drawer_inspection", drawer, "unknown_tm03_drawer")
    steps = max(0, int(dwell_steps))
    if not controller.runtime_config.execute:
        return _drawer_failure(
            "physical_drawer_inspection",
            drawer,
            "physical_drawer_inspection_requires_execute",
        )
    hold = controller.run_gripper_command(1.0, steps)
    fraction = float(controller.env.drawer_open_fraction(drawer))
    state = controller.env.semantic_state
    checks = {
        "inspection_dwell_executed": int(hold.get("steps", 0)) == steps,
        "drawer_remains_open": fraction
        >= float(controller.env.physical_observer.config.drawer_open_threshold),
        "active_drawer_matches": str(state.active_drawer) == drawer,
        "physical_dwell_marked_inspected": bool(state.drawer_inspected),
    }
    passed = bool(all(checks.values()))
    return {
        "primitive": "physical_drawer_inspection_dwell",
        "operation": "inspect_drawer",
        "fixture": drawer,
        "dwell_steps": steps,
        "hold_gripper": _without_obs(hold),
        "drawer_open_fraction": fraction,
        "checks": checks,
        "task_diagnostics": controller.env.task_diagnostics(),
        "semantic_shortcut_used": False,
        "pass": passed,
        "reason": None if passed else "tm03_drawer_inspection_dwell_not_observed",
    }


def _close_drawer_with_tm03_contact(
    controller: TM03ManipulationController,
    drawer: str,
    *,
    actuation_mode: str = "contact",
    close_overshoot: float = 0.005,
    closed_fraction_threshold: float = 0.08,
    **kwargs: Any,
) -> dict[str, Any]:
    del kwargs
    if drawer not in tuple(getattr(controller.env, "drawer_names", ())):
        return _drawer_failure("physical_drawer_push_close", drawer, "unknown_tm03_drawer")
    if str(actuation_mode).strip().lower() != "contact":
        return _drawer_failure(
            "physical_drawer_push_close",
            drawer,
            "tm03_drawer_requires_contact_actuation",
        )
    result = close_tm03_drawer_physically(
        controller,
        fixture_name=drawer,
        config=DrawerPushCloseConfig(
            close_overshoot=float(close_overshoot),
            closed_fraction_threshold=float(closed_fraction_threshold),
        ),
    )
    result.update(
        {
            "operation": "close_drawer_physically",
            "contact_primitive_reused_from": "TM03",
            "semantic_shortcut_used": False,
            "task_diagnostics": controller.env.task_diagnostics(),
        }
    )
    return result


def _retreat_and_return_arm_after_drawer(
    controller: TM03ManipulationController,
    drawer: str,
    *,
    retreat_distance: float = 0.125,
    release_steps: int = 10,
    **kwargs: Any,
) -> dict[str, Any]:
    """Disengage from a closed handle before returning to regular arm qpos."""

    del kwargs
    if drawer not in tuple(getattr(controller.env, "drawer_names", ())):
        return _drawer_failure(
            "drawer_post_close_retreat_and_return",
            drawer,
            "unknown_tm03_drawer",
        )
    if not controller.runtime_config.execute:
        return _drawer_failure(
            "drawer_post_close_retreat_and_return",
            drawer,
            "physical_drawer_retreat_requires_execute",
        )

    controller.refresh_obs()
    env = controller.env
    fixture = env.cabinet_for_drawer(drawer)
    initial_fraction = float(env.drawer_open_fraction(drawer))
    closed_threshold = float(env.physical_observer.config.drawer_closed_threshold)
    if initial_fraction > closed_threshold:
        return _drawer_failure(
            "drawer_post_close_retreat_and_return",
            drawer,
            "tm03_drawer_not_closed_before_retreat",
        )

    failures_before = tuple(env.semantic_state.failure_labels)
    release = controller.run_gripper_command(-1.0, max(0, int(release_steps)))
    controller.refresh_obs()
    current_world = np.asarray(
        matrix_from_site_pose(env, env.robots[0]),
        dtype=float,
    )
    frame = resolve_drawer_motion_frame(
        fixture_yaw=float(fixture.rot),
        current_eef_rotation_world=current_world[:3, :3],
        hook_orientation_blend=0.0,
    )
    retreat_world = current_world.copy()
    retreat_world[:3, 3] += (
        frame.open_direction_world * max(0.0, float(retreat_distance))
    )
    retreat = controller.run_planned_stage(
        _world_target_in_current_base(controller, retreat_world),
        stage=f"tm03_{drawer}_post_close_retreat",
        gripper_action=-1.0,
    )
    if not bool(retreat.get("pass", False)):
        return {
            "primitive": "drawer_post_close_retreat_and_return",
            "fixture": drawer,
            "release_gripper": _without_obs(release),
            "retreat": retreat,
            "pass": False,
            "reason": "tm03_drawer_post_close_retreat_failed",
        }

    arm_return = controller.return_arm_to_regular_pose(gripper_action=-1.0)
    final_fraction = float(env.drawer_open_fraction(drawer))
    failures_after = tuple(env.semantic_state.failure_labels)
    new_failures = [
        label for label in failures_after if label not in failures_before
    ]
    checks = {
        "release_executed": int(release.get("steps", 0)) == max(
            0, int(release_steps)
        ),
        "retreat_executed": bool(retreat.get("pass", False)),
        "arm_returned": bool(arm_return.get("pass", False)),
        "drawer_remains_closed": final_fraction <= closed_threshold,
        "no_new_semantic_failure": not new_failures,
    }
    passed = bool(all(checks.values()))
    return {
        "primitive": "drawer_post_close_retreat_and_return",
        "fixture": drawer,
        "retreat_distance": float(retreat_distance),
        "release_gripper": _without_obs(release),
        "retreat": retreat,
        "arm_return": arm_return,
        "drawer_open_fraction_initial": initial_fraction,
        "drawer_open_fraction_final": final_fraction,
        "new_failure_labels": new_failures,
        "checks": checks,
        "semantic_shortcut_used": False,
        "pass": passed,
        "reason": None if passed else "tm03_drawer_post_close_return_failed",
    }


def _press_completion_button(
    controller: TM03ManipulationController,
    button: str,
    *,
    prepress_z_offset: float = 0.08,
    press_z_offset: float = -0.010,
    hold_steps: int = 12,
    gripper_action: float = 1.0,
    near_miss_position_tolerance: float = 0.09,
    press_depth_threshold_m: float = 0.004,
    require_physical_button_motion: bool = True,
    **kwargs: Any,
) -> dict[str, Any]:
    """Physically press either TM03 completion button and verify its event."""

    del kwargs
    names = tuple(getattr(controller.env, "completion_button_world_xyzs", {}))
    if button not in names:
        return _drawer_failure(
            "physical_tm03_completion_button_press",
            button,
            "unknown_tm03_completion_button",
        )
    state = controller.env.semantic_state
    closed_threshold = float(
        controller.env.physical_observer.config.drawer_closed_threshold
    )
    all_drawers_closed = all(
        float(controller.env.drawer_open_fraction(drawer)) <= closed_threshold
        for drawer in controller.env.drawer_names
    )
    preconditions = {
        "a1_complete": bool(state.a1_complete),
        "a2_complete": bool(state.a2_complete),
        "b1_complete": bool(state.b1_complete),
        "b2_complete": bool(state.b2_complete),
        "burner_off": not bool(controller.env.bound_burner_on()),
        "all_drawers_closed": all_drawers_closed,
        "no_active_drawer": state.active_drawer is None,
        "no_absorbing_failure": not bool(state.absorbing_failure),
        "not_previously_pressed": not bool(state.completion_button_pressed),
    }
    if not all(preconditions.values()):
        return {
            "primitive": "physical_tm03_completion_button_press",
            "button": button,
            "preconditions": preconditions,
            "semantic_shortcut_used": False,
            "pass": False,
            "reason": "tm03_completion_button_preconditions_not_met",
        }

    actuation = press_completion_button_physically(
        controller,
        button_name=button,
        prepress_z_offset=float(prepress_z_offset),
        press_z_offset=float(press_z_offset),
        hold_steps=int(hold_steps),
        gripper_action=float(gripper_action),
        near_miss_position_tolerance=float(near_miss_position_tolerance),
        press_depth_threshold=float(press_depth_threshold_m),
        require_physical_button_motion=bool(require_physical_button_motion),
    )
    diagnostics = controller.env.task_diagnostics()
    checks = {
        "physical_press_operation": bool(actuation.get("pass", False)),
        "physical_button_motion": bool(
            actuation.get("physical_button_motion_observed", False)
        ),
        "completion_event_committed": bool(
            diagnostics.get("completion_button_pressed", False)
        ),
        "correct_button_committed": str(
            diagnostics.get("completion_button_id")
        )
        == button,
        "task_success": bool(diagnostics.get("success", False)),
        "no_absorbing_failure": not bool(
            diagnostics.get("absorbing_failure", False)
        ),
    }
    passed = bool(all(checks.values()))
    return {
        "primitive": "physical_tm03_completion_button_press",
        "button": button,
        "preconditions": preconditions,
        "actuation": actuation,
        "task_diagnostics": diagnostics,
        "checks": checks,
        "semantic_shortcut_used": False,
        "pass": passed,
        "reason": None if passed else "tm03_completion_button_press_failed",
    }


def _drawer_failure(primitive: str, drawer: str, reason: str) -> dict[str, Any]:
    return {
        "primitive": primitive,
        "fixture": str(drawer),
        "semantic_shortcut_used": False,
        "pass": False,
        "reason": reason,
    }


def load_knob_interaction_calibration(path: str | Path) -> KnobInteractionCalibration:
    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if payload.get("config_kind") != "tm03_native_stove_knob_interaction":
        raise ValueError(f"invalid TM03 knob interaction resource: {source}")
    pose_values = _section(payload, "pose_files")
    required_roles = {"pinch_off", "pinch_on", "pinch_on_for_off"}
    if set(pose_values) != required_roles:
        raise ValueError(
            "TM03 knob pose_files must contain exactly " + ", ".join(sorted(required_roles))
        )
    pose_files = {
        role: (source.parent / str(value)).resolve()
        for role, value in pose_values.items()
    }
    for role, pose_path in pose_files.items():
        if not pose_path.is_file():
            raise FileNotFoundError(f"missing TM03 {role} pose: {pose_path}")
        pose_payload = json.loads(pose_path.read_text(encoding="utf-8"))
        if pose_payload.get("config_kind") != "anchor_pose":
            raise ValueError(f"TM03 {role} is not an anchor pose: {pose_path}")

    waypoints = _section(payload, "rotation_waypoints_deg")
    turn_on = _finite_float_tuple(waypoints.get("turn_on"), "turn_on waypoints")
    turn_off = _finite_float_tuple(waypoints.get("turn_off"), "turn_off waypoints")
    if not turn_on or turn_on[-1] != 90.0 or any(
        right <= left for left, right in zip(turn_on, turn_on[1:])
    ):
        raise ValueError("TM03 turn_on waypoints must increase and end at 90 degrees")
    if not turn_off or 0.0 not in turn_off or turn_off[-1] >= 0.0 or any(
        right >= left for left, right in zip(turn_off, turn_off[1:])
    ):
        raise ValueError(
            "TM03 turn_off waypoints must decrease, pass through 0, and end "
            "with negative overtravel"
        )
    gripper = _section(payload, "gripper")
    validation = _section(payload, "validation")
    step_values = {
        name: int(gripper[name])
        for name in ("open_steps", "close_steps", "release_steps")
    }
    if any(value <= 0 for value in step_values.values()):
        raise ValueError("TM03 knob gripper step counts must be positive")
    off_max = float(validation["off_qpos_max_rad"])
    on_min = float(validation["on_qpos_min_rad"])
    if not np.isfinite(off_max) or not np.isfinite(on_min) or not 0.0 <= off_max < on_min:
        raise ValueError("TM03 knob validation thresholds are invalid")
    return KnobInteractionCalibration(
        source=source.resolve(),
        target_knob=str(payload["target_knob"]),
        target_burner=str(payload["target_burner"]),
        pose_files=pose_files,
        turn_on_waypoints_deg=turn_on,
        turn_off_waypoints_deg=turn_off,
        open_steps=step_values["open_steps"],
        close_steps=step_values["close_steps"],
        release_steps=step_values["release_steps"],
        off_qpos_max_rad=off_max,
        on_qpos_min_rad=on_min,
    )


def _runtime_config(
    *,
    execute: bool,
    runtime: Mapping[str, Any],
    planner: Mapping[str, Any],
    static_scene: Mapping[str, Any],
) -> ManipulationRuntimeConfig:
    runtime_values = dict(runtime)
    planner_values = dict(planner)
    planner_values.pop("conda_env", None)
    planner_values.pop("subprocess_timeout_s", None)
    planner_config = MplibPlannerConfig(**planner_values)
    return ManipulationRuntimeConfig(
        execute=bool(execute),
        max_joint_step=float(runtime_values.pop("max_joint_step", 0.015)),
        settle_steps=int(runtime_values.pop("settle_steps", 20)),
        position_threshold=float(runtime_values.pop("position_threshold", 0.04)),
        orientation_threshold_deg=float(
            runtime_values.pop("orientation_threshold_deg", 8.0)
        ),
        qpos_threshold=float(runtime_values.pop("qpos_threshold", 0.12)),
        open_steps=30,
        close_steps=30,
        release_steps=30,
        planner_args=planner_config.planner_args(),
        static_scene_options=dict(static_scene),
    )


def _make_planner(*, script_path: Path, conda_env: str, timeout_s: float):
    def plan(payload: dict[str, Any]) -> dict[str, Any]:
        code, result = run_mplib_planner_subprocess(
            script_path=script_path,
            plan_request=payload,
            conda_env=conda_env,
            timeout_s=timeout_s,
        )
        if result is None:
            return {
                "subprocess_returncode": int(code),
                "plan": {"status": "planner_subprocess_no_json", "attempts": []},
            }
        result["subprocess_returncode"] = int(code)
        return result

    return plan


def _resolve_anchor_pose_world(
    env: Any,
    path: Path,
) -> tuple[np.ndarray, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    pose = _section(payload, "pose")
    anchor = _section(payload, "anchor")
    world = pose.get("world")
    if not isinstance(world, Mapping):
        raise ValueError(f"TM03 anchor pose needs pose.world: {path}")
    matrix = matrix_from_json_pose(dict(world))
    selector = str(anchor.get("selector", ""))
    if not selector.startswith("geom:"):
        raise ValueError(f"TM03 knob pose must use a geom anchor: {path}")
    anchor_name = selector.removeprefix("geom:")
    geom_id = int(env.sim.model.geom_name2id(anchor_name))
    current_anchor = np.asarray(env.sim.data.geom_xpos[geom_id], dtype=float)
    offset = np.asarray(pose.get("offset_from_anchor_world"), dtype=float).reshape(3)
    matrix[:3, 3] = current_anchor + offset
    return matrix, {
        "pose_file": str(path),
        "pose_name": payload.get("pose_name"),
        "selector": selector,
        "current_anchor_world": current_anchor.tolist(),
        "offset_from_anchor_world": offset.tolist(),
        "resolved_pose_world": matrix.tolist(),
    }


def _world_target_in_current_base(
    controller: ManipulationController,
    target_world: np.ndarray,
) -> dict[str, Any]:
    controller.refresh_obs()
    base_world = matrix_from_base_obs(controller.obs)
    return pose_from_matrix(np.linalg.inv(base_world) @ target_world)


def _knob_rotation_pose(pinch_off_world: np.ndarray, *, angle_deg: float) -> np.ndarray:
    angle = np.deg2rad(-float(angle_deg))
    delta = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    result = np.array(pinch_off_world, dtype=float, copy=True)
    result[:3, :3] = result[:3, :3] @ delta
    return result


def _live_knob_state(env: Any) -> dict[str, Any]:
    diagnostics = getattr(env, "task_diagnostics", None)
    return {
        "knob_qpos_rad": float(env.native_knob_qpos()),
        "knob_qvel_rad_s": float(env.native_knob_qvel()),
        "target_burner_on": bool(env.bound_burner_on()),
        "task_diagnostics": diagnostics() if callable(diagnostics) else None,
    }


def _stage_result(raw: dict[str, Any]) -> StageResult:
    passed = bool(raw.get("pass", False))
    return StageResult(
        index=-1,
        kind=StageKind.OPERATE,
        success=passed,
        details=raw,
        failure_code=None if passed else str(raw.get("reason") or "tm03_knob_stage_failed"),
    )


def _knob_failure(
    operation: str,
    target: str,
    *,
    reason: str,
    **details: Any,
) -> dict[str, Any]:
    return {
        "primitive": "pinch_and_rotate_native_knob",
        "operation": operation,
        "target": str(target),
        **details,
        "semantic_shortcut_used": False,
        "pass": False,
        "reason": reason,
    }


def _section(payload: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = payload.get(name, {})
    if not isinstance(value, Mapping):
        raise ValueError(f"TM03 {name} must be a mapping")
    return value


def _finite_float_tuple(value: Any, label: str) -> tuple[float, ...]:
    try:
        result = tuple(float(item) for item in value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"TM03 {label} must be a sequence") from exc
    if not all(np.isfinite(item) for item in result):
        raise ValueError(f"TM03 {label} must be finite")
    return result


def _angle_label(value: float) -> str:
    number = float(value)
    return str(int(number)) if number.is_integer() else str(number).replace(".", "p")


def _without_obs(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _without_obs(item) for key, item in value.items() if key != "obs"}
    if isinstance(value, list):
        return [_without_obs(item) for item in value]
    if isinstance(value, tuple):
        return [_without_obs(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


__all__ = [
    "KnobInteractionCalibration",
    "ManipulationAssembly",
    "TM03ManipulationController",
    "build_manipulation_assembly",
    "load_knob_interaction_calibration",
]
