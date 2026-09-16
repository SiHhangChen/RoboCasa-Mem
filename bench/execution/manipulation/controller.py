"""Manipulation controllers for bench-owned operation execution."""

from __future__ import annotations

import copy
from collections.abc import Callable, Mapping
from typing import Any

import numpy as np

from bench.execution.manipulation.drivers import (
    ArmDriver,
    RoboCasaArmDriver,
    TrajectoryStopCheck,
)
from bench.execution.manipulation.pick_payloads import PickPayloadSelector, resolve_pick_payload
from bench.execution.manipulation.finger_alignment import measure_finger_pad_center
from bench.execution.manipulation.planning_bridge import (
    ManipulationRuntimeConfig,
    PlannedStageExecutor,
    PlannerFn,
    default_planner,
    skipped_stage,
)
from bench.execution.contracts import StageKind, StageResult
from bench.execution.planning import (
    ManualGraspConfig,
    PickPlaceConfig,
    matrix_from_base_obs,
    matrix_from_pos_quat_xyzw,
    make_receptacle_place_target_poses,
    make_world_place_target_pose_candidates,
    matrix_from_json_pose,
    pose_from_matrix,
)
from bench.execution.planning.pick_place_targets import object_pose_matrix, object_pos


ReachabilityTargetResolver = Callable[[dict[str, Any]], dict[str, Any] | None]
PreReleaseCheck = Callable[[], dict[str, Any]]


def _dict_to_stage_result(raw: dict[str, Any]) -> StageResult:
    """Convert a manipulation result dict to a StageResult."""
    return StageResult(
        index=-1,
        kind=StageKind.OPERATE,
        success=bool(raw.get("pass", False)),
        details=raw,
        failure_code=raw.get("reason") if not raw.get("pass") else None,
    )


def _operation_name(operation: str) -> str:
    return str(getattr(operation, "value", operation))


def _truthy(value: Any, default: bool = False) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return value
    raw = str(value).strip().lower()
    if raw in {"1", "true", "yes", "y", "on"}:
        return True
    if raw in {"0", "false", "no", "n", "off"}:
        return False
    return bool(value)


def _stage_settle_steps(
    kwargs: Mapping[str, Any],
    stage: str,
    *,
    fallback_key: str = "pick_settle_steps",
) -> int | None:
    value = kwargs.get(f"{stage}_settle_steps", kwargs.get(fallback_key))
    if value is None:
        return None
    return max(0, int(value))


def topdown_eef_rotation_world_for_base_yaw(
    obs: Mapping[str, Any],
    yaw_deg: float,
) -> np.ndarray:
    """Return a world-frame top-down EEF rotation at an absolute Base yaw."""

    angle = np.deg2rad(float(yaw_deg))
    yaw = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    topdown = np.diag([1.0, -1.0, -1.0])
    return matrix_from_base_obs(dict(obs))[:3, :3] @ yaw @ topdown


def prepared_stage_plan_trajectory_metrics(
    prepared_stage_plans: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Measure a prepared stage sequence without introducing ranking criteria."""

    waypoints: list[np.ndarray] = []
    joint_count: int | None = None
    for stage_name, prepared in prepared_stage_plans.items():
        result = prepared.get("planner_result") if isinstance(prepared, Mapping) else None
        plan = result.get("plan") if isinstance(result, Mapping) else None
        trajectory = plan.get("trajectory") if isinstance(plan, Mapping) else None
        try:
            values = np.asarray(trajectory, dtype=float)
        except (TypeError, ValueError):
            values = np.empty((0, 0), dtype=float)
        if values.ndim != 2 or values.shape[0] < 1 or values.shape[1] < 1:
            return {
                "valid": False,
                "reason": f"invalid_trajectory_shape:{stage_name}",
                "total_joint_path_length_rad": None,
                "waypoint_count": 0,
                "max_single_joint_step_rad": None,
            }
        if not np.all(np.isfinite(values)):
            return {
                "valid": False,
                "reason": f"non_finite_trajectory:{stage_name}",
                "total_joint_path_length_rad": None,
                "waypoint_count": int(values.shape[0]),
                "max_single_joint_step_rad": None,
            }
        if joint_count is None:
            joint_count = int(values.shape[1])
        elif int(values.shape[1]) != joint_count:
            return {
                "valid": False,
                "reason": f"trajectory_joint_count_mismatch:{stage_name}",
                "total_joint_path_length_rad": None,
                "waypoint_count": int(len(waypoints)),
                "max_single_joint_step_rad": None,
            }
        waypoints.extend(np.array(row, copy=True) for row in values)
    if not waypoints:
        return {
            "valid": False,
            "reason": "no_prepared_stage_trajectories",
            "total_joint_path_length_rad": None,
            "waypoint_count": 0,
            "max_single_joint_step_rad": None,
        }
    stacked = np.stack(waypoints, axis=0)
    deltas = np.diff(stacked, axis=0)
    total = float(np.sum(np.linalg.norm(deltas, axis=1))) if len(deltas) else 0.0
    max_step = float(np.max(np.abs(deltas))) if deltas.size else 0.0
    return {
        "valid": True,
        "reason": None,
        "total_joint_path_length_rad": total,
        "waypoint_count": int(len(stacked)),
        "max_single_joint_step_rad": max_step,
    }


class ManipulationController:
    """Independent pick/place/return controller for RoboCasa-like envs."""

    def __init__(
        self,
        env: Any,
        obs: dict[str, Any],
        *,
        arm_driver: ArmDriver | None = None,
        planner: PlannerFn | None = None,
        pick_place_config: PickPlaceConfig | None = None,
        manual_grasp_config: ManualGraspConfig | None = None,
        runtime_config: ManipulationRuntimeConfig | None = None,
        plan_template: dict[str, Any] | None = None,
        model_candidates: list[dict[str, Any]] | None = None,
        operation_handlers: Mapping[str, Callable[..., dict[str, Any]]] | None = None,
        pick_payload_selector: PickPayloadSelector | None = None,
        reachability_target_resolver: ReachabilityTargetResolver | None = None,
    ) -> None:
        self.env = env
        self.obs = obs
        self.driver = arm_driver or RoboCasaArmDriver(env, obs)
        self.pick_place_config = pick_place_config or PickPlaceConfig()
        self.manual_grasp_config = manual_grasp_config or ManualGraspConfig()
        self.runtime_config = runtime_config or ManipulationRuntimeConfig()
        self.operation_handlers = dict(operation_handlers or {})
        self.pick_payload_selector = pick_payload_selector
        self.reachability_target_resolver = reachability_target_resolver
        self._virtual_held_object: str | None = None
        self._virtual_object_from_eef: np.ndarray | None = None
        self.stage_executor = PlannedStageExecutor(
            env,
            obs,
            driver=self.driver,
            planner=planner or default_planner,
            runtime_config=self.runtime_config,
            plan_template=plan_template,
            model_candidates=model_candidates,
        )

    def execute(
        self,
        operation: str,
        target: str,
        **kwargs: Any,
    ) -> StageResult:
        self.refresh_obs()
        op = _operation_name(operation)
        if op == "move_eef":
            raw = self.move_eef_to_pose(target_name=target, **kwargs)
        elif op == "pick":
            raw = self.pick(target, **kwargs)
        elif op == "place":
            receptacle = kwargs.pop("receptacle", target)
            raw = self.place(target, receptacle, **kwargs)
        elif op == "place_at":
            raw = self.place_at_world_pose(target, **kwargs)
        elif op == "return_arm":
            raw = self.return_arm_to_regular_pose(**kwargs)
        elif op in self.operation_handlers:
            handler = self.operation_handlers.get(op)
            raw = handler(target, **kwargs)
        else:
            return StageResult(
                index=-1,
                kind=StageKind.OPERATE,
                success=False,
                failure_code=f"unknown operation: {operation!r}",
            )
        return _dict_to_stage_result(raw)

    def move_eef_to_pose(
        self,
        *,
        target_name: str,
        target_pose: Mapping[str, Any],
        frame: str = "base",
        planner_stage: str | None = None,
        gripper_action: float = -1.0,
        holding_object: str | None = None,
        collision_options: dict[str, Any] | None = None,
        record_trajectory: bool = False,
        settle_steps: int | None = None,
        post_step_callback: Callable[[dict[str, Any]], dict[str, Any] | None]
        | None = None,
    ) -> dict[str, Any]:
        """Plan or execute one explicit end-effector pose.

        The task probe CLI resolves live anchor poses before invoking this
        operation, so the controller only needs the two planner-facing frames:
        ``base`` and ``world``.
        """

        self.refresh_obs()
        pose_matrix = matrix_from_json_pose(dict(target_pose))
        if pose_matrix.shape != (4, 4) or not np.all(np.isfinite(pose_matrix)):
            raise ValueError("target_pose must describe a finite 4x4 transform")
        resolved_frame = str(frame).strip().lower()
        if resolved_frame == "world":
            base_world = matrix_from_base_obs(self.obs)
            pose_matrix = np.linalg.inv(base_world) @ pose_matrix
        elif resolved_frame != "base":
            raise ValueError("move_eef frame must be 'base' or 'world'")
        target_base = pose_from_matrix(pose_matrix)
        stage_name = str(planner_stage or target_name or "move_eef")
        result = self.run_planned_stage(
            target_base,
            stage=stage_name,
            gripper_action=float(gripper_action),
            holding_object=holding_object,
            collision_options=collision_options,
            record_trajectory=bool(record_trajectory),
            settle_steps=settle_steps,
            post_step_callback=post_step_callback,
        )
        return {
            **result,
            "primitive": "move_eef",
            "target_name": str(target_name),
            "input_frame": resolved_frame,
            "input_target_pose": copy.deepcopy(dict(target_pose)),
        }

    def refresh_obs(self) -> dict[str, Any]:
        """Refresh cached observations after external base or env motion."""
        pending_navigation_obs = getattr(self, "_pending_navigation_obs", None)
        if pending_navigation_obs is not None:
            self._pending_navigation_obs = None
            self._sync_obs(pending_navigation_obs)
            return pending_navigation_obs
        obs = self.env._get_observations(force_update=True)
        self._sync_obs(obs)
        return obs

    def pick(self, object_name: str, **kwargs: Any) -> dict[str, Any]:
        initial_object_pos = self._object_pos_world(object_name)
        post_step_callback = kwargs.get("post_step_callback")
        pick_kwargs = dict(kwargs)
        pick_kwargs.pop("post_step_callback", None)
        if "object_pos_world" not in pick_kwargs:
            pick_kwargs["object_pos_world"] = self._virtual_object_static_pos_world(object_name)
        payload, selection = resolve_pick_payload(
            env=self.env,
            obs=self.obs,
            object_name=object_name,
            kwargs=pick_kwargs,
            pick_place_config=self.pick_place_config,
            manual_grasp_config=self.manual_grasp_config,
            pick_payload_selector=self.pick_payload_selector,
            dry_run_stage=self.dry_run_stage,
            dry_run_pick_sequence=self.dry_run_pick_sequence,
            dry_run_pick_then_world_place=self.dry_run_pick_then_world_place,
        )
        if payload is None:
            return self._failed_pick_summary(
                object_name,
                reason=str(
                    (selection or {}).get("reason", "pick_payload_selection_failed")
                ),
                manual_grasp_selection=selection,
            )
        open_result = self._run_gripper(
            -1.0,
            self.runtime_config.open_steps,
            post_step_callback=post_step_callback,
        )
        retreat_to_pregrasp = _truthy(
            kwargs.get(
                "retreat_to_pregrasp_after_grasp",
                kwargs.get("post_grasp_retreat_to_pregrasp"),
            ),
            default=False,
        )
        close_gripper_action = float(kwargs.get("close_gripper_action", 1.0))
        if not -1.0 <= close_gripper_action <= 1.0:
            raise ValueError("close_gripper_action must be in [-1, 1]")
        held_gripper_action = float(
            kwargs.get("held_gripper_action", close_gripper_action)
        )
        if not -1.0 <= held_gripper_action <= 1.0:
            raise ValueError("held_gripper_action must be in [-1, 1]")
        preshape_gripper_steps = max(
            0,
            int(kwargs.get("preshape_gripper_steps", 0)),
        )
        preshape_gripper_action = float(
            kwargs.get("preshape_gripper_action", 1.0)
        )
        descent_gripper_action = float(
            kwargs.get(
                "descent_gripper_action",
                0.0 if preshape_gripper_steps > 0 else -1.0,
            )
        )
        for name, action in (
            ("preshape_gripper_action", preshape_gripper_action),
            ("descent_gripper_action", descent_gripper_action),
        ):
            if not -1.0 <= action <= 1.0:
                raise ValueError(f"{name} must be in [-1, 1]")

        stages = []
        pregrasp = self.stage_executor.run(
            payload["poses"]["pregrasp"]["base"],
            stage="pregrasp",
            gripper_action=-1.0,
            post_step_callback=post_step_callback,
            settle_steps=_stage_settle_steps(kwargs, "pregrasp"),
        )
        self._accept_pick_pregrasp_near_miss(pregrasp, kwargs=kwargs)
        stages.append(pregrasp)
        preshape_result = (
            self._run_gripper(
                preshape_gripper_action,
                preshape_gripper_steps,
                post_step_callback=post_step_callback,
            )
            if pregrasp["pass"] and preshape_gripper_steps > 0
            else {
                "dry_run": not self.runtime_config.execute,
                "skipped": True,
                "steps": 0,
            }
        )
        finger_alignment = {"enabled": False, "pass": True}
        if pregrasp["pass"] and _truthy(kwargs.get("align_preshape_finger_center"), default=False):
            if self.runtime_config.execute:
                finger_alignment = self._align_preshape_fingers(
                    payload, object_name=object_name, initial_object_pos=initial_object_pos,
                    post_step_callback=post_step_callback,
                )
            else:
                finger_alignment = {"enabled": True, "pass": True, "skipped": "dry_run"}
        grasp = (
            self.stage_executor.run(
                payload["poses"]["grasp"]["base"],
                stage="grasp",
                gripper_action=descent_gripper_action,
                record_trajectory=retreat_to_pregrasp,
                post_step_callback=post_step_callback,
                settle_steps=_stage_settle_steps(kwargs, "grasp"),
            )
            if pregrasp["pass"] and finger_alignment["pass"]
            else skipped_stage("grasp", "pregrasp_or_finger_alignment_failed")
        )
        grasp_for_close = bool(grasp.get("pass", False)) or self._accept_pick_grasp_near_miss(
            grasp,
            kwargs=kwargs,
        )
        stages.append(grasp)
        close_result = (
            self._run_gripper(
                close_gripper_action,
                self.runtime_config.close_steps,
                post_step_callback=post_step_callback,
            )
            if grasp_for_close
            else {"dry_run": not self.runtime_config.execute, "skipped": True, "steps": 0}
        )
        post_close_hold_steps = max(0, int(kwargs.get("post_close_hold_steps", 0)))
        post_close_hold = (
            self._run_gripper(
                held_gripper_action,
                post_close_hold_steps,
                post_step_callback=post_step_callback,
            )
            if grasp_for_close and post_close_hold_steps > 0
            else {
                "dry_run": not self.runtime_config.execute,
                "skipped": True,
                "steps": 0,
            }
        )
        grasp_validation_after_close = self._pick_grasp_validation(
            object_name,
            initial_object_pos=initial_object_pos,
        )
        weak_grasp_after_close = not (
            bool(grasp_validation_after_close.get("contact", False))
            and bool(grasp_validation_after_close.get("grasped_threshold_0.6", False))
        )
        reclose_gripper_steps = max(0, int(kwargs.get("reclose_gripper_steps", 0)))
        reclose_on_weak_grasp = _truthy(
            kwargs.get("reclose_on_weak_grasp"),
            default=False,
        )
        reclose_result = (
            self._run_gripper(
                float(kwargs.get("reclose_gripper_action", close_gripper_action)),
                reclose_gripper_steps,
                post_step_callback=post_step_callback,
            )
            if (
                grasp_for_close
                and reclose_on_weak_grasp
                and weak_grasp_after_close
                and reclose_gripper_steps > 0
            )
            else {
                "dry_run": not self.runtime_config.execute,
                "skipped": True,
                "steps": 0,
            }
        )
        post_grasp_stage_name = "retreat" if retreat_to_pregrasp else "lift"
        if retreat_to_pregrasp:
            post_grasp_stages = [
                self._reverse_pick_retreat(
                    grasp,
                    gripper_action=held_gripper_action,
                )
                if grasp_for_close
                else skipped_stage("retreat", "grasp_failed")
            ]
            carry_lift_height = float(
                kwargs.get(
                    "post_retreat_lift_height",
                    kwargs.get("carry_lift_height", 0.0),
                )
            )
            if carry_lift_height > 1e-9:
                post_grasp_stages.append(
                    self._lift_current_eef_pose(
                        object_name,
                        stage="carry_lift",
                        lift_height=carry_lift_height,
                        gripper_action=held_gripper_action,
                    )
                    if bool(post_grasp_stages[-1].get("pass", False))
                    else skipped_stage("carry_lift", "retreat_failed")
                )
        else:
            post_grasp_stages = [
                self.stage_executor.run(
                    payload["poses"]["lift"]["base"],
                    stage="lift",
                    gripper_action=held_gripper_action,
                    holding_object=object_name,
                    post_step_callback=post_step_callback,
                    settle_steps=_stage_settle_steps(kwargs, "lift"),
                )
                if grasp_for_close
                else skipped_stage(post_grasp_stage_name, "grasp_failed")
            ]
        stages.extend(post_grasp_stages)
        checks = self._stage_checks(stages, command_name="close", command=close_result)
        if finger_alignment["enabled"]:
            checks["finger_center_alignment"] = bool(finger_alignment["pass"])
        grasp_validation = self._pick_grasp_validation(
            object_name,
            initial_object_pos=initial_object_pos,
        )
        grasp_confirmed = bool(grasp_validation.get("contact", False)) and bool(
            grasp_validation.get("grasped_threshold_0.6", False)
        )
        accepted_without_lift = False
        if self.runtime_config.execute:
            checks["object_grasp_or_lift"] = bool(
                grasp_confirmed
                or grasp_validation.get("object_lifted", False)
            )
            if (
                not checks.get(post_grasp_stage_name, False)
                and grasp_confirmed
            ):
                post_grasp_stages[-1]["accepted_without_lift_reason"] = (
                    "object_grasped_after_close_but_lift_plan_failed"
                )
                checks[post_grasp_stage_name] = True
                accepted_without_lift = True
        if (
            (not self.runtime_config.execute)
            and bool(all(checks.values()))
            and payload.get("poses", {}).get("grasp", {}).get("world") is not None
        ):
            self._set_virtual_held_object(object_name, payload["poses"]["grasp"]["world"])
        elif (
            self.runtime_config.execute
            and grasp_confirmed
            and payload.get("manual_eef_site_in_object") is not None
        ):
            self._virtual_held_object = str(object_name)
            self._virtual_object_from_eef = np.linalg.inv(
                matrix_from_json_pose(payload["manual_eef_site_in_object"])
            )
        return {
            "primitive": "pick",
            "object_name": object_name,
            "grasp_strategy": payload.get("grasp_strategy"),
            "object_bbox": payload.get("object_bbox"),
            "manual_grasp_pose_file": payload.get("manual_grasp_pose_file"),
            "manual_grasp_selection": selection,
            "post_grasp_retreat_strategy": (
                "reverse_grasp_trajectory" if retreat_to_pregrasp else "lift"
            ),
            "held_gripper_action": held_gripper_action,
            "close_gripper_action": close_gripper_action,
            "preshape_gripper_action": preshape_gripper_action,
            "preshape_gripper_steps": preshape_gripper_steps,
            "descent_gripper_action": descent_gripper_action,
            "target_poses": payload["poses"],
            "open_gripper": self._without_obs(open_result),
            "preshape_gripper": self._without_obs(preshape_result),
            "finger_center_alignment": finger_alignment,
            "close_gripper": self._without_obs(close_result),
            "post_close_hold": self._without_obs(post_close_hold),
            "grasp_validation_after_close": grasp_validation_after_close,
            "weak_grasp_after_close": weak_grasp_after_close,
            "reclose_gripper": self._without_obs(reclose_result),
            "stages": stages,
            "grasp_validation": grasp_validation,
            "checks": checks,
            "dry_run": not self.runtime_config.execute,
            "pass": bool(all(checks.values())),
        }

    def _align_preshape_fingers(
        self, payload: dict[str, Any], *, object_name: str,
        initial_object_pos: np.ndarray, post_step_callback: Any,
    ) -> dict[str, Any]:
        report: dict[str, Any] = {"enabled": True, "pass": False}
        try:
            measurement = measure_finger_pad_center(self.env)
            report["before"] = measurement
            target = matrix_from_json_pose(payload["poses"]["grasp"]["world"])
            site_id = int(self.env.robots[0].eef_site_id["right"])
            live_rotation = np.asarray(self.env.sim.data.site_xmat[site_id]).reshape(3, 3)
            local_axis = live_rotation.T @ np.asarray(measurement["closing_axis_world"])
            target_axis = target[:3, :3] @ local_axis
            points = np.asarray(payload["object_bbox"]["points"], dtype=float)
            required_opening = float(np.ptp(points @ target_axis)) + 0.004
            report["required_opening_m"] = required_opening
            reopen_steps = 0
            while measurement["opening_m"] < required_opening and reopen_steps < 20:
                self._run_gripper(-1.0, 1, post_step_callback=post_step_callback)
                reopen_steps += 1
                measurement = measure_finger_pad_center(self.env)
            report.update({"after": measurement, "reopen_steps": reopen_steps})
            if measurement["opening_m"] < required_opening:
                report["reason"] = "insufficient_finger_opening"
                return report
            offset = np.asarray(measurement["pad_midpoint_in_tcp"], dtype=float)
            # Keep the calibrated grasp depth; only compensate lateral pad offset.
            offset[2] = 0.0
            correction = -(target[:3, :3] @ offset)
            object_motion = self._object_pos_world(object_name) - initial_object_pos
            object_motion[2] = 0.0
            correction += object_motion
            report["object_center_world"] = (
                np.asarray(payload["object_bbox"]["center"]) + object_motion
            ).tolist()
            report["correction_world_m"] = correction.tolist()
            if float(np.linalg.norm(correction)) > 0.03:
                report["reason"] = "finger_center_correction_exceeds_3cm"
                return report
            self.refresh_obs()
            base_world = matrix_from_base_obs(self.obs)
            corrected = copy.deepcopy(payload)
            for name in ("grasp", "lift"):
                pose = matrix_from_json_pose(corrected["poses"][name]["world"])
                pose[:3, 3] += correction
                corrected["poses"][name] = {
                    "world": pose_from_matrix(pose),
                    "base": pose_from_matrix(np.linalg.inv(base_world) @ pose),
                }
            snapshot = self._virtual_state_snapshot()
            try:
                validation = []
                for name, action in (("grasp", 0.0), ("lift", 1.0)):
                    stage = self.stage_executor.run(
                        corrected["poses"][name]["base"], stage=name,
                        gripper_action=action, execute=False, commit_virtual_state=True,
                        holding_object=object_name if name == "lift" else None,
                    )
                    validation.append(stage)
                    if not stage.get("pass", False):
                        break
                report["preflight"] = validation
            finally:
                self._restore_virtual_state_snapshot(snapshot)
            if len(validation) != 2 or not all(stage.get("pass", False) for stage in validation):
                report["reason"] = "corrected_finger_center_preflight_failed"
                return report
            payload["poses"] = corrected["poses"]
            report["pass"] = True
            return report
        except (ValueError, KeyError, AttributeError) as error:
            report["reason"] = str(error)
            return report

    def _accept_pick_grasp_near_miss(
        self,
        grasp_stage: dict[str, Any],
        *,
        kwargs: dict[str, Any],
    ) -> bool:
        """Allow a close attempt when EEF pose reached but joint-settle was marginal."""
        if not _truthy(kwargs.get("allow_grasp_near_miss"), default=False):
            return False
        if bool(grasp_stage.get("pass", False)):
            return True
        if grasp_stage.get("stage") != "grasp":
            return False
        checks = grasp_stage.get("checks") or {}
        if not (
            bool(checks.get("plan_success", False))
            and bool(checks.get("runtime_collision_free", False))
            and bool(checks.get("final_orientation", False))
        ):
            return False
        execution = grasp_stage.get("execution") or {}
        position_error = float(
            execution.get("final_position_error_m", float("inf"))
        )
        max_position_error_value = kwargs.get(
            "grasp_near_miss_max_position_error"
        )
        max_position_error = (
            None
            if max_position_error_value is None
            else float(max_position_error_value)
        )
        position_ok = bool(checks.get("final_position", False)) or bool(
            max_position_error is not None
            and np.isfinite(position_error)
            and position_error <= max_position_error
        )
        if not position_ok:
            return False
        qpos_error = float(execution.get("final_qpos_error", float("inf")))
        max_qpos_error = float(kwargs.get("grasp_near_miss_max_qpos_error", 0.25))
        if not np.isfinite(qpos_error) or qpos_error > max_qpos_error:
            return False
        grasp_stage["accepted_near_miss_for_close"] = {
            "reason": "eef_pose_reached_but_final_qpos_over_threshold",
            "final_qpos_error": qpos_error,
            "max_qpos_error": max_qpos_error,
        }
        grasp_stage["pass"] = True
        checks = dict(checks)
        if not bool(checks.get("final_position", False)):
            grasp_stage["accepted_near_miss_for_close"].update(
                {
                    "final_position_error_m": position_error,
                    "max_position_error_m": max_position_error,
                }
            )
            checks["final_position"] = True
            checks["near_miss_final_position_accepted"] = True
        checks["final_qpos"] = True
        checks["near_miss_final_qpos_accepted"] = True
        grasp_stage["checks"] = checks
        grasp_stage["failure_phase"] = None
        grasp_stage["failure_reason"] = None
        return True

    def _accept_pick_pregrasp_near_miss(
        self,
        pregrasp_stage: dict[str, Any],
        *,
        kwargs: dict[str, Any],
    ) -> bool:
        """Allow a pick to continue when pregrasp motion reached near the target."""
        if not _truthy(kwargs.get("allow_pregrasp_near_miss"), default=False):
            return False
        if bool(pregrasp_stage.get("pass", False)):
            return True
        if pregrasp_stage.get("stage") != "pregrasp":
            return False
        checks = pregrasp_stage.get("checks") or {}
        if not (
            bool(checks.get("plan_success", False))
            and bool(checks.get("runtime_collision_free", False))
        ):
            return False
        execution = pregrasp_stage.get("execution") or {}
        position_error = float(execution.get("final_position_error_m", float("inf")))
        max_position_error = float(
            kwargs.get("pregrasp_near_miss_max_position_error", 0.22)
        )
        if not np.isfinite(position_error) or position_error > max_position_error:
            return False
        pregrasp_stage["accepted_near_miss_for_grasp"] = {
            "reason": "pregrasp_motion_near_target_with_valid_plan",
            "final_position_error_m": position_error,
            "max_position_error_m": max_position_error,
        }
        pregrasp_stage["pass"] = True
        checks = dict(checks)
        checks["final_position"] = True
        checks["final_orientation"] = True
        checks["final_qpos"] = True
        checks["near_miss_accepted"] = True
        pregrasp_stage["checks"] = checks
        pregrasp_stage["failure_phase"] = None
        pregrasp_stage["failure_reason"] = None
        return True

    def _accept_place_near_miss(
        self,
        place_stage: dict[str, Any],
        *,
        kwargs: dict[str, Any],
    ) -> bool:
        """Allow release when a place motion stops close enough to the target."""
        checks = place_stage.get("checks") or {}
        execution = place_stage.get("execution") or {}
        early_stop = execution.get("early_stop") or {}
        if (
            _truthy(kwargs.get("allow_place_trajectory_early_stop"), default=False)
            and bool(early_stop.get("stop", False))
            and bool(checks.get("plan_success", False))
            and bool(checks.get("runtime_collision_free", False))
        ):
            place_stage["accepted_trajectory_early_stop_for_release"] = {
                "reason": str(
                    early_stop.get("reason") or "place_trajectory_stop_check"
                ),
                "early_stop": dict(early_stop),
            }
            place_stage["pass"] = True
            checks = dict(checks)
            checks["final_position"] = True
            checks["final_orientation"] = True
            checks["final_qpos"] = True
            checks["trajectory_early_stop_accepted"] = True
            place_stage["checks"] = checks
            place_stage["failure_phase"] = None
            place_stage["failure_reason"] = None
            return True
        if not _truthy(kwargs.get("allow_place_near_miss"), default=False):
            return False
        if bool(place_stage.get("pass", False)):
            return True
        if place_stage.get("stage") != "place":
            return False
        checks = place_stage.get("checks") or {}
        qpos_ok = bool(checks.get("final_qpos", False))
        if (
            not qpos_ok
            and _truthy(kwargs.get("allow_place_qpos_near_miss"), default=False)
        ):
            execution = place_stage.get("execution") or {}
            qpos_error = float(execution.get("final_qpos_error", float("inf")))
            max_qpos_error = float(kwargs.get("place_near_miss_max_qpos_error", 0.22))
            qpos_ok = bool(np.isfinite(qpos_error) and qpos_error <= max_qpos_error)
        else:
            qpos_error = None
            max_qpos_error = None
        if not (
            bool(checks.get("plan_success", False))
            and bool(checks.get("runtime_collision_free", False))
            and bool(checks.get("final_orientation", False))
            and qpos_ok
        ):
            return False
        execution = place_stage.get("execution") or {}
        position_error = float(execution.get("final_position_error_m", float("inf")))
        max_position_error = float(kwargs.get("place_near_miss_max_position_error", 0.08))
        if not np.isfinite(position_error) or position_error > max_position_error:
            return False
        place_stage["accepted_near_miss_for_release"] = {
            "reason": "place_motion_near_target_with_valid_plan",
            "final_position_error_m": position_error,
            "max_position_error_m": max_position_error,
            "final_qpos_error": qpos_error,
            "max_qpos_error": max_qpos_error,
        }
        place_stage["pass"] = True
        checks = dict(checks)
        checks["final_position"] = True
        checks["final_qpos"] = True
        checks["near_miss_accepted"] = True
        place_stage["checks"] = checks
        place_stage["failure_phase"] = None
        place_stage["failure_reason"] = None
        return True

    def dry_run_stage(
        self,
        target_pose_base: dict[str, Any],
        *,
        stage: str,
        gripper_action: float,
        holding_object: str | None = None,
        collision_options: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return self.stage_executor.run(
            target_pose_base,
            stage=stage,
            gripper_action=float(gripper_action),
            holding_object=holding_object,
            collision_options=collision_options,
            execute=False,
            commit_virtual_state=False,
        )

    def dry_run_pick_sequence(
        self,
        pick_payload: dict[str, Any],
        *,
        object_name: str,
    ) -> dict[str, Any]:
        """Plan pregrasp, grasp, and lift without committing virtual state."""
        snapshot = self._virtual_state_snapshot()
        try:
            stages: list[dict[str, Any]] = []
            for stage_name, gripper_action in (
                ("pregrasp", -1.0),
                ("grasp", -1.0),
                ("lift", 1.0),
            ):
                stage = self.stage_executor.run(
                    pick_payload["poses"][stage_name]["base"],
                    stage=stage_name,
                    gripper_action=gripper_action,
                    holding_object=object_name if stage_name == "lift" else None,
                    execute=False,
                    commit_virtual_state=True,
                )
                stages.append(stage)
                if not bool(stage.get("pass", False)):
                    break
            summary = self._dry_run_pick_place_summary(
                stages,
                object_name=object_name,
            )
            summary["primitive"] = "dry_run_pick_sequence"
            return summary
        finally:
            self._restore_virtual_state_snapshot(snapshot)

    def current_base_entry_reachability_summary(
        self,
        anchor: str,
        *,
        purpose: str,
        holding: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Dry-run the next pick/place entry stage from the current base pose."""
        self.refresh_obs()
        snapshot = self._virtual_state_snapshot()
        try:
            dry_entry = self._dry_run_entry_stage(
                anchor,
                purpose=purpose,
                holding=holding,
                **kwargs,
            )
        finally:
            self._restore_virtual_state_snapshot(snapshot)
        reachable = bool(dry_entry.get("pass", False))
        summary = {
            "primitive": "navigate_to",
            "anchor": anchor,
            "purpose": purpose,
            "holding": holding,
            "skipped": reachable,
            "reason": (
                "current_base_entry_stage_reachable"
                if reachable
                else "current_base_entry_stage_not_reachable"
            ),
            "candidate_prevalidation": {
                "enabled": True,
                "pass": reachable,
                "checks": {"entry_stage_plan": reachable},
                "dry_entry_stage": dry_entry,
            },
            "committed": False,
            "dry_run_only": True,
            "checks": {"entry_stage_plan": reachable},
            "pass": reachable,
        }
        for key in ("target_kind", "target_name", "target_info"):
            if key in kwargs:
                summary[key] = kwargs[key]
        return summary

    def current_base_world_place_reachability_summary(
        self,
        object_name: str,
        *,
        target_name: str,
        target_pos_world: Any,
        place_height: float | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Dry-run a world-frame place entry stage from the current base pose."""
        self.refresh_obs()
        snapshot = self._virtual_state_snapshot()
        try:
            dry_place = self._dry_run_world_place_stage(
                object_name,
                target_name=target_name,
                target_pos_world=target_pos_world,
                place_height=place_height,
                **kwargs,
            )
        finally:
            self._restore_virtual_state_snapshot(snapshot)
        reachable = bool(dry_place.get("pass", False))
        target_pos = np.array(target_pos_world, dtype=float)
        return {
            "primitive": "navigate_to",
            "anchor": target_name,
            "purpose": "place",
            "holding": object_name,
            "target_pos_world": target_pos.astype(float).tolist(),
            "skipped": reachable,
            "reason": (
                "current_base_world_place_stage_reachable"
                if reachable
                else "current_base_world_place_stage_not_reachable"
            ),
            "candidate_prevalidation": {
                "enabled": True,
                "pass": reachable,
                "checks": {"entry_stage_plan": reachable},
                "dry_entry_stage": dry_place,
            },
            "committed": False,
            "dry_run_only": True,
            "checks": {"entry_stage_plan": reachable},
            "pass": reachable,
        }

    def _dry_run_entry_stage(
        self,
        anchor: str,
        *,
        purpose: str,
        holding: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        entry_stage = self._entry_stage_for_purpose(purpose)
        if entry_stage is None:
            return self._dry_run_unavailable_entry(
                "entry",
                reason=f"unsupported_navigation_purpose:{purpose}",
                anchor=anchor,
                purpose=purpose,
                holding=holding,
            )

        if purpose == "pick":
            payload, selection = resolve_pick_payload(
                env=self.env,
                obs=self.obs,
                object_name=anchor,
                kwargs={
                    **kwargs,
                    "object_pos_world": self._virtual_object_static_pos_world(anchor),
                },
                pick_place_config=self.pick_place_config,
            manual_grasp_config=self.manual_grasp_config,
            pick_payload_selector=self.pick_payload_selector,
            dry_run_stage=self.dry_run_stage,
            dry_run_pick_sequence=self.dry_run_pick_sequence,
            dry_run_pick_then_world_place=self.dry_run_pick_then_world_place,
        )
            if payload is None:
                summary = self._dry_run_unavailable_entry(
                    entry_stage,
                    reason=str(
                        (selection or {}).get("reason", "pick_payload_selection_failed")
                    ),
                    anchor=anchor,
                    purpose=purpose,
                    holding=holding,
                )
                summary["manual_grasp_selection"] = selection
                return summary
            summary = self.dry_run_stage(
                payload["poses"]["pregrasp"]["base"],
                stage=entry_stage,
                gripper_action=-1.0,
            )
            summary["manual_grasp_selection"] = selection
            summary["target_poses"] = payload.get("poses")
            return self._tag_dry_run_entry_summary(
                summary,
                anchor=anchor,
                purpose=purpose,
                holding=holding,
            )

        target = self._resolve_place_entry_target(anchor, holding=holding, kwargs=kwargs)
        if target is None:
            return self._dry_run_unavailable_entry(
                entry_stage,
                reason="entry_target_unavailable",
                anchor=anchor,
                purpose=purpose,
                holding=holding,
            )
        if target.get("mode") == "world":
            summary = self._dry_run_world_place_stage(
                str(target["object_name"]),
                target_name=str(target["target_name"]),
                target_pos_world=target["target_pos_world"],
                place_height=target.get("place_height"),
                retreat_height=target.get("retreat_height"),
                retreat_x=target.get("retreat_x"),
                retreat_y=target.get("retreat_y"),
                release_offset_x=target.get("release_offset_x"),
                release_offset_y=target.get("release_offset_y"),
                post_release_clearance_distance=target.get(
                    "post_release_clearance_distance"
                ),
                post_release_clearance_z_offset=target.get(
                    "post_release_clearance_z_offset"
                ),
                placement_reference=target.get("placement_reference"),
                placement_object_upright=bool(
                    target.get("placement_object_upright", False)
                ),
                placement_yaw_offset_deg=target.get("placement_yaw_offset_deg"),
                placement_yaw_offsets_deg=target.get("placement_yaw_offsets_deg"),
                placement_approach_angles_deg=target.get("placement_approach_angles_deg"),
                placement_entry_standoff=target.get("placement_entry_standoff"),
                placement_exit_standoff=target.get("placement_exit_standoff"),
                placement_entry_z_offset=target.get("placement_entry_z_offset"),
                placement_exit_z_offset=target.get("placement_exit_z_offset"),
                collision_options=target.get("collision_options"),
            )
        else:
            payload = target["payload"]
            summary = self.dry_run_stage(
                payload["poses"]["place"]["base"],
                stage=entry_stage,
                gripper_action=1.0,
                holding_object=str(target["object_name"]),
                collision_options=target.get("collision_options"),
            )
            summary["target_poses"] = payload.get("poses")
            summary = self._tag_dry_run_entry_summary(
                summary,
                anchor=anchor,
                purpose=purpose,
                holding=holding,
            )
        for key in ("target_kind", "target_name", "target_info"):
            if key in target:
                summary[key] = target[key]
        return summary

    def _dry_run_world_place_stage(
        self,
        object_name: str,
        *,
        target_name: str,
        target_pos_world: Any,
        place_height: float | None = None,
        retreat_height: float | None = None,
        retreat_x: float | None = None,
        retreat_y: float | None = None,
        release_offset_x: float | None = None,
        release_offset_y: float | None = None,
        post_release_clearance_distance: float | None = None,
        post_release_clearance_z_offset: float | None = None,
        placement_reference: str | None = None,
        placement_object_upright: bool = False,
        placement_yaw_offset_deg: float | None = None,
        placement_yaw_offsets_deg: Any | None = None,
        placement_approach_angles_deg: Any | None = None,
        placement_entry_standoff: Any | None = None,
        placement_exit_standoff: Any | None = None,
        placement_entry_z_offset: Any | None = None,
        placement_exit_z_offset: Any | None = None,
        collision_options: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        try:
            target_pos = np.array(target_pos_world, dtype=float)
            payload, selection = self._select_world_place_payload(
                object_name=object_name,
                target_pos_world=target_pos,
                collision_options=collision_options,
                placement_reference=placement_reference,
                placement_object_upright=placement_object_upright,
                placement_yaw_offset_deg=placement_yaw_offset_deg,
                placement_yaw_offsets_deg=placement_yaw_offsets_deg,
                placement_approach_angles_deg=placement_approach_angles_deg,
                placement_entry_standoff=placement_entry_standoff,
                placement_exit_standoff=placement_exit_standoff,
                placement_entry_z_offset=placement_entry_z_offset,
                placement_exit_z_offset=placement_exit_z_offset,
                place_kwargs=self.pick_place_config.place_target_kwargs(
                    place_height=place_height,
                    retreat_height=retreat_height,
                    retreat_x=retreat_x,
                    retreat_y=retreat_y,
                    release_offset_x=release_offset_x,
                    release_offset_y=release_offset_y,
                    post_release_clearance_distance=post_release_clearance_distance,
                    post_release_clearance_z_offset=post_release_clearance_z_offset,
                ),
            )
        except Exception as exc:
            summary = self._dry_run_unavailable_entry(
                "place",
                reason="world_place_target_unavailable",
                anchor=target_name,
                purpose="place",
                holding=object_name,
            )
            summary["error"] = f"{type(exc).__name__}: {exc}"
            return summary

        summary = self._dry_run_place_payload_entry_sequence(
            payload,
            object_name=object_name,
            collision_options=collision_options,
        )
        summary["placement_orientation_selection"] = selection
        summary["target_poses"] = payload.get("poses")
        summary["target_pos_world"] = target_pos.astype(float).tolist()
        summary["target_name"] = target_name
        return self._tag_dry_run_entry_summary(
            summary,
            anchor=target_name,
            purpose="place",
            holding=object_name,
        )

    def dry_run_pick_then_world_place(
        self,
        pick_payload: dict[str, Any],
        *,
        object_name: str,
        target_pos_world: Any,
        base_pos: Any | None = None,
        base_quat_xyzw: Any | None = None,
        place_height: float | None = None,
        retreat_height: float | None = None,
        retreat_x: float | None = None,
        retreat_y: float | None = None,
        release_offset_x: float | None = None,
        release_offset_y: float | None = None,
        post_release_clearance_distance: float | None = None,
        post_release_clearance_z_offset: float | None = None,
        placement_reference: str | None = None,
        placement_object_upright: bool = False,
        placement_yaw_offset_deg: Any | None = None,
        placement_yaw_offsets_deg: Any | None = None,
        placement_approach_angles_deg: Any | None = None,
        placement_entry_standoff: Any | None = None,
        placement_exit_standoff: Any | None = None,
        placement_entry_z_offset: Any | None = None,
        placement_exit_z_offset: Any | None = None,
        collision_options: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Plan pick + world-place without committing dry-run virtual state."""
        snapshot = self._virtual_state_snapshot()
        try:
            stages: list[dict[str, Any]] = []
            for stage_name, gripper_action in (
                ("pregrasp", -1.0),
                ("grasp", -1.0),
                ("lift", 1.0),
            ):
                stage = self.stage_executor.run(
                    pick_payload["poses"][stage_name]["base"],
                    stage=stage_name,
                    gripper_action=gripper_action,
                    holding_object=object_name if stage_name == "lift" else None,
                    execute=False,
                    commit_virtual_state=True,
                )
                stages.append(stage)
                if not bool(stage.get("pass", False)):
                    return self._dry_run_pick_place_summary(stages, object_name=object_name)

            self._set_virtual_held_object(
                object_name,
                pick_payload["poses"]["grasp"]["world"],
            )
            obs_for_place = (
                self._obs_with_base_pose(base_pos=base_pos, base_quat_xyzw=base_quat_xyzw)
                if base_pos is not None or base_quat_xyzw is not None
                else self.obs
            )
            current_eef_world = self.stage_executor.current_eef_site_world()
            if base_pos is not None or base_quat_xyzw is not None:
                current_eef_world = (
                    matrix_from_pos_quat_xyzw(
                        np.array(obs_for_place["robot0_base_pos"], dtype=float),
                        np.array(obs_for_place["robot0_base_quat"], dtype=float),
                    )
                    @ matrix_from_json_pose(
                        self.stage_executor._virtual_eef_site_in_base
                        or pick_payload["poses"]["lift"]["base"]
                    )
                )
            current_object_pos = self._virtual_object_pos_world_for_eef(
                object_name,
                current_eef_world,
            )
            current_object_pose = None
            if (
                self._virtual_held_object == str(object_name)
                and self._virtual_object_from_eef is not None
            ):
                current_object_pose = current_eef_world @ self._virtual_object_from_eef
            if current_object_pos is None:
                current_object_pos = self._virtual_object_pos_world(object_name)
                current_object_pose = self._virtual_object_pose_world(object_name)
            place_payload, selection = self._select_world_place_payload(
                object_name=object_name,
                target_pos_world=np.array(target_pos_world, dtype=float),
                obs=obs_for_place,
                current_eef_world=current_eef_world,
                current_object_pos_world=current_object_pos,
                current_object_pose_world=current_object_pose,
                placement_reference=placement_reference,
                placement_object_upright=placement_object_upright,
                placement_yaw_offset_deg=placement_yaw_offset_deg,
                placement_yaw_offsets_deg=placement_yaw_offsets_deg,
                placement_approach_angles_deg=placement_approach_angles_deg,
                placement_entry_standoff=placement_entry_standoff,
                placement_exit_standoff=placement_exit_standoff,
                placement_entry_z_offset=placement_entry_z_offset,
                placement_exit_z_offset=placement_exit_z_offset,
                collision_options=collision_options,
                place_kwargs=self.pick_place_config.place_target_kwargs(
                    place_height=place_height,
                    retreat_height=retreat_height,
                    retreat_x=retreat_x,
                    retreat_y=retreat_y,
                    release_offset_x=release_offset_x,
                    release_offset_y=release_offset_y,
                    post_release_clearance_distance=post_release_clearance_distance,
                    post_release_clearance_z_offset=post_release_clearance_z_offset,
                ),
            )
            place_stage_sequence = []
            if "entry" in place_payload.get("poses", {}):
                place_stage_sequence.append(("entry", 1.0))
            place_stage_sequence.append(("place", 1.0))
            if "clearance" in place_payload.get("poses", {}):
                place_stage_sequence.append(("clearance", -1.0))
            place_stage_sequence.append(("retreat", -1.0))
            for stage_name, gripper_action in place_stage_sequence:
                stage = self.stage_executor.run(
                    place_payload["poses"][stage_name]["base"],
                    stage=stage_name,
                    gripper_action=gripper_action,
                    holding_object=object_name if stage_name in {"entry", "place"} else None,
                    collision_options=(
                        collision_options if stage_name in {"entry", "place"} else None
                    ),
                    execute=False,
                    commit_virtual_state=True,
                )
                stages.append(stage)
                if not bool(stage.get("pass", False)):
                    summary = self._dry_run_pick_place_summary(
                        stages,
                        object_name=object_name,
                    )
                    summary["place_payload"] = {
                        key: value
                        for key, value in place_payload.items()
                        if key != "poses"
                    }
                    summary["placement_orientation_selection"] = selection
                    return summary
            summary = self._dry_run_pick_place_summary(stages, object_name=object_name)
            summary["place_payload"] = {
                key: value for key, value in place_payload.items() if key != "poses"
            }
            summary["placement_orientation_selection"] = selection
            return summary
        finally:
            self._restore_virtual_state_snapshot(snapshot)

    def run_planned_stage(
        self,
        target_pose_base: dict[str, Any],
        *,
        stage: str,
        gripper_action: float,
        holding_object: str | None = None,
        collision_options: dict[str, Any] | None = None,
        execute: bool | None = None,
        record_trajectory: bool = False,
        settle_steps: int | None = None,
        post_step_callback: Callable[[dict[str, Any]], dict[str, Any] | None]
        | None = None,
    ) -> dict[str, Any]:
        """Run one generic arm-planning stage for task-level primitives."""
        summary = self.stage_executor.run(
            target_pose_base,
            stage=stage,
            gripper_action=float(gripper_action),
            holding_object=holding_object,
            collision_options=collision_options,
            execute=execute,
            record_trajectory=record_trajectory,
            settle_steps=settle_steps,
            post_step_callback=post_step_callback,
        )
        self._sync_obs(self.stage_executor.obs)
        return summary

    def run_gripper_command(self, action: float, steps: int) -> dict[str, Any]:
        """Run a gripper command while holding the current arm position."""
        return self._run_gripper(float(action), int(steps))

    def execute_arm_trajectory(
        self,
        trajectory: list[list[float]],
        *,
        stage: str,
        gripper_action: float,
        settle_steps: int | None = None,
    ) -> dict[str, Any]:
        """Execute a preplanned arm trajectory through the configured driver."""
        if not self.runtime_config.execute:
            return {
                "stage": stage,
                "dry_run": True,
                "skipped": True,
                "trajectory_len": len(trajectory),
                "pass": len(trajectory) > 0,
            }
        execution = self.driver.execute_trajectory(
            trajectory,
            max_joint_step=self.runtime_config.max_joint_step,
            settle_steps=(
                self.runtime_config.settle_steps
                if settle_steps is None
                else int(settle_steps)
            ),
            gripper_action=float(gripper_action),
        )
        self._sync_obs(execution["obs"])
        qpos_ok = float(execution["final_qpos_error"]) < self.runtime_config.qpos_threshold
        runtime_collision_free = not bool(execution.get("collision", {}).get("collision", False))
        return {
            "stage": stage,
            "side_effect_boundary": "arm_moved",
            "execution": self._without_obs(execution),
            "checks": {
                "trajectory_available": len(trajectory) > 0,
                "final_qpos": qpos_ok,
                "runtime_collision_free": runtime_collision_free,
            },
            "trajectory_len": len(trajectory),
            "pass": bool(qpos_ok and runtime_collision_free),
            "failure_phase": None if qpos_ok and runtime_collision_free else "execution",
            "failure_reason": (
                None
                if qpos_ok and runtime_collision_free
                else "final_qpos"
                if not qpos_ok
                else "runtime_collision"
            ),
        }

    def _failed_pick_summary(
        self,
        object_name: str,
        *,
        reason: str,
        manual_grasp_selection: dict[str, Any] | None,
    ) -> dict[str, Any]:
        stages = [
            skipped_stage("pregrasp", reason),
            skipped_stage("grasp", "pregrasp_failed"),
            skipped_stage("lift", "grasp_failed"),
        ]
        checks = self._stage_checks(
            stages,
            command_name="close",
            command={"skipped": True},
        )
        return {
            "primitive": "pick",
            "object_name": object_name,
            "reason": reason,
            "manual_grasp_selection": manual_grasp_selection,
            "target_poses": None,
            "open_gripper": {"skipped": True, "reason": reason},
            "close_gripper": {"skipped": True, "reason": reason},
            "stages": stages,
            "checks": checks,
            "dry_run": not self.runtime_config.execute,
            "pass": False,
        }

    def place(self, object_name: str, receptacle: str, **kwargs: Any) -> dict[str, Any]:
        payload = make_receptacle_place_target_poses(
            self.env,
            self.obs,
            object_name=object_name,
            receptacle_name=receptacle,
            current_eef_world=self.stage_executor.current_eef_site_world(),
            receptacle_pos_world=self._virtual_object_static_pos_world(receptacle),
            **self.pick_place_config.place_target_kwargs(
                place_height=kwargs.get("place_height"),
                retreat_height=kwargs.get("retreat_height"),
                retreat_x=kwargs.get("retreat_x"),
                retreat_y=kwargs.get("retreat_y"),
                release_offset_x=kwargs.get("release_offset_x"),
                release_offset_y=kwargs.get("release_offset_y"),
                post_release_clearance_distance=kwargs.get(
                    "post_release_clearance_distance"
                ),
                post_release_clearance_z_offset=kwargs.get(
                    "post_release_clearance_z_offset"
                ),
            ),
        )
        return self._execute_place_payload(
            object_name,
            target_label=receptacle,
            target_payload=payload,
            primitive="place",
            collision_options=kwargs.get("collision_options"),
            place_kwargs=kwargs,
        )

    def place_at_world_pose(self, object_name: str, **kwargs: Any) -> dict[str, Any]:
        target_pos_world = kwargs.get("target_pos_world", kwargs.get("target_pos"))
        if target_pos_world is None:
            return {
                "primitive": "place_at",
                "object_name": object_name,
                "pass": False,
                "reason": "place_at requires target_pos_world or target_pos",
            }
        target_name = str(kwargs.get("target_name", object_name))
        # A real grasp can settle or slip slightly after the planned grasp
        # pose.  Placement must use the live EEF-to-object transform; reusing
        # the planned attachment would command the gripper upright while
        # leaving the physical object tilted.  This is a read-only planning
        # input -- object motion still comes exclusively from env.step(action).
        live_eef_world = None
        live_object_pose = None
        live_object_pos = None
        if self.runtime_config.execute:
            live_eef_world = self.stage_executor.current_eef_site_world()
            live_object_pose = self._object_pose_world(object_name)
            live_object_pos = live_object_pose[:3, 3]
        payload, selection = self._select_world_place_payload(
            object_name=object_name,
            target_pos_world=np.array(target_pos_world, dtype=float),
            current_eef_world=live_eef_world,
            current_object_pos_world=live_object_pos,
            current_object_pose_world=live_object_pose,
            collision_options=kwargs.get("collision_options"),
            placement_reference=kwargs.get(
                "placement_reference",
                kwargs.get("place_reference"),
            ),
            placement_object_upright=bool(
                kwargs.get("placement_object_upright", False)
            ),
            placement_yaw_offset_deg=kwargs.get(
                "placement_yaw_offset_deg",
                kwargs.get("place_yaw_offset_deg"),
            ),
            placement_yaw_offsets_deg=kwargs.get(
                "placement_yaw_offsets_deg",
                kwargs.get("place_yaw_offsets_deg"),
            ),
            placement_approach_angles_deg=kwargs.get(
                "placement_approach_angles_deg",
                kwargs.get("place_approach_angles_deg"),
            ),
            placement_entry_standoff=kwargs.get(
                "placement_entry_standoff",
                kwargs.get("place_entry_standoff"),
            ),
            placement_exit_standoff=kwargs.get(
                "placement_exit_standoff",
                kwargs.get("place_exit_standoff"),
            ),
            placement_entry_z_offset=kwargs.get(
                "placement_entry_z_offset",
                kwargs.get("place_entry_z_offset"),
            ),
            placement_exit_z_offset=kwargs.get(
                "placement_exit_z_offset",
                kwargs.get("place_exit_z_offset"),
            ),
            post_release_clearance_direction_world=kwargs.get(
                "post_release_clearance_direction_world"
            ),
            place_kwargs=self.pick_place_config.place_target_kwargs(
                place_height=kwargs.get("place_height"),
                retreat_height=kwargs.get("retreat_height"),
                retreat_x=kwargs.get("retreat_x"),
                retreat_y=kwargs.get("retreat_y"),
                release_offset_x=kwargs.get("release_offset_x"),
                release_offset_y=kwargs.get("release_offset_y"),
                post_release_clearance_distance=kwargs.get(
                    "post_release_clearance_distance"
                ),
                post_release_clearance_z_offset=kwargs.get(
                    "post_release_clearance_z_offset"
                ),
            ),
        )
        result = self._execute_place_payload(
            object_name,
            target_label=target_name,
            target_payload=payload,
            primitive="place_at",
            collision_options=kwargs.get("collision_options"),
            place_kwargs=kwargs,
            pre_release_check=kwargs.get("pre_release_check"),
            place_trajectory_stop_check=kwargs.get("place_trajectory_stop_check"),
        )
        result["target_pos_world"] = np.array(target_pos_world, dtype=float).tolist()
        result["placement_orientation_selection"] = selection
        return result

    def place_at_world_pose_candidates(
        self,
        object_name: str,
        *,
        candidate_groups: list[dict[str, Any]],
        pre_release_check: PreReleaseCheck | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Plan grouped place candidates and execute the shortest valid plan once.

        Every candidate in a group is evaluated. A later group is considered
        only when the preceding group has no valid plan. Candidate plans are
        ranked solely by total joint-space path length, with their fixed input
        order as the deterministic tie-break.
        """

        target_name = str(kwargs.get("target_name", object_name))
        collision_options = kwargs.get("collision_options")
        common_place_kwargs = self.pick_place_config.place_target_kwargs(
            place_height=kwargs.get("place_height"),
            retreat_height=kwargs.get("retreat_height"),
            retreat_x=kwargs.get("retreat_x"),
            retreat_y=kwargs.get("retreat_y"),
            release_offset_x=None,
            release_offset_y=None,
            post_release_clearance_distance=kwargs.get(
                "post_release_clearance_distance"
            ),
            post_release_clearance_z_offset=kwargs.get(
                "post_release_clearance_z_offset"
            ),
        )
        current_eef_world = np.array(
            self.stage_executor.current_eef_site_world(),
            copy=True,
        )
        current_object_pos_world = self._virtual_object_pos_world(object_name)
        current_object_pose_world = self._virtual_object_pose_world(object_name)
        if current_object_pos_world is None:
            current_object_pos_world = self._object_pos_world(object_name)
        # Candidate placement must use the live object orientation after a
        # physical grasp; otherwise upright placement can be evaluated against
        # a stale or missing object pose.
        if current_object_pose_world is None:
            current_object_pose_world = self._object_pose_world(object_name)
            if current_object_pos_world is None:
                current_object_pos_world = current_object_pose_world[:3, 3]

        candidate_records: list[dict[str, Any]] = []
        group_records: list[dict[str, Any]] = []
        selected: dict[str, Any] | None = None
        evaluation_order = 0
        for group_index, raw_group in enumerate(candidate_groups):
            group = dict(raw_group)
            group_name = str(group.get("group_name", group_index))
            group_candidates = list(group.get("candidates") or ())
            valid_in_group: list[dict[str, Any]] = []
            group_start = len(candidate_records)
            for candidate_index, raw_candidate in enumerate(group_candidates):
                candidate = dict(raw_candidate)
                metadata = dict(candidate.get("metadata") or {})
                fixed_order = int(
                    metadata.get("fixed_candidate_order", evaluation_order)
                )
                inside_target = bool(
                    candidate.get("release_target_inside_saucepan", True)
                )
                record: dict[str, Any] = {
                    "group_name": group_name,
                    "group_index": int(group_index),
                    "candidate_index": int(candidate_index),
                    "evaluation_order": int(evaluation_order),
                    "fixed_candidate_order": fixed_order,
                    "release_target_inside_saucepan": inside_target,
                    **metadata,
                }
                evaluation_order += 1
                if not inside_target:
                    record.update(
                        {
                            "pass": False,
                            "failure_phase": "geometry",
                            "failure_reason": "release_target_outside_saucepan",
                            "trajectory_metrics": None,
                            "stages": [],
                        }
                    )
                    candidate_records.append(record)
                    continue
                target_pos_world = candidate.get("target_pos_world")
                if target_pos_world is None:
                    record.update(
                        {
                            "pass": False,
                            "failure_phase": "geometry",
                            "failure_reason": "candidate_target_pos_world_missing",
                            "trajectory_metrics": None,
                            "stages": [],
                        }
                    )
                    candidate_records.append(record)
                    continue
                candidate_place_kwargs = dict(common_place_kwargs)
                candidate_place_kwargs["release_offset_x"] = float(
                    candidate.get("release_offset_x", 0.0)
                )
                candidate_place_kwargs["release_offset_y"] = float(
                    candidate.get("release_offset_y", 0.0)
                )
                try:
                    payloads = self._make_world_place_payload_candidates(
                        object_name=object_name,
                        target_pos_world=np.asarray(target_pos_world, dtype=float),
                        obs=self.obs,
                        current_eef_world=current_eef_world,
                        current_object_pos_world=current_object_pos_world,
                        current_object_pose_world=current_object_pose_world,
                        placement_reference=candidate.get(
                            "placement_reference",
                            kwargs.get("placement_reference", kwargs.get("place_reference")),
                        ),
                        placement_object_upright=bool(
                            candidate.get(
                                "placement_object_upright",
                                kwargs.get("placement_object_upright", False),
                            )
                        ),
                        target_eef_rot_world=(
                            topdown_eef_rotation_world_for_base_yaw(
                                self.obs,
                                float(candidate["placement_base_yaw_deg"]),
                            )
                            if candidate.get("placement_base_yaw_deg") is not None
                            else None
                        ),
                        placement_yaw_offset_deg=(
                            None
                            if candidate.get("placement_base_yaw_deg") is not None
                            else candidate.get("placement_yaw_offset_deg")
                        ),
                        placement_yaw_offsets_deg=None,
                        placement_approach_angles_deg=candidate.get(
                            "placement_approach_angle_deg",
                            kwargs.get(
                                "placement_approach_angles_deg",
                                kwargs.get("place_approach_angles_deg"),
                            ),
                        ),
                        placement_entry_standoff=kwargs.get(
                            "placement_entry_standoff",
                            kwargs.get("place_entry_standoff"),
                        ),
                        placement_exit_standoff=kwargs.get(
                            "placement_exit_standoff",
                            kwargs.get("place_exit_standoff"),
                        ),
                        placement_entry_z_offset=kwargs.get(
                            "placement_entry_z_offset",
                            kwargs.get("place_entry_z_offset"),
                        ),
                        placement_exit_z_offset=kwargs.get(
                            "placement_exit_z_offset",
                            kwargs.get("place_exit_z_offset"),
                        ),
                        post_release_clearance_direction_world=kwargs.get(
                            "post_release_clearance_direction_world"
                        ),
                        place_kwargs=candidate_place_kwargs,
                    )
                    if len(payloads) != 1:
                        raise ValueError(
                            "each explicit world-place candidate must resolve to one payload"
                        )
                    payload = payloads[0]
                    dry_summary = self._dry_run_place_payload_entry_sequence(
                        payload,
                        object_name=object_name,
                        collision_options=collision_options,
                        capture_prepared_plans=True,
                    )
                except TypeError:
                    raise
                except Exception as exc:
                    record.update(
                        {
                            "pass": False,
                            "failure_phase": "candidate_setup",
                            "failure_reason": f"{type(exc).__name__}: {exc}",
                            "trajectory_metrics": None,
                            "stages": [],
                        }
                    )
                    candidate_records.append(record)
                    continue
                prepared = dry_summary.get("prepared_stage_plans") or {}
                metrics = prepared_stage_plan_trajectory_metrics(prepared)
                candidate_pass = bool(dry_summary.get("pass", False)) and bool(
                    metrics.get("valid", False)
                )
                record.update(
                    {
                        "pass": candidate_pass,
                        "failure_phase": (
                            None
                            if candidate_pass
                            else dry_summary.get("failure_phase") or "trajectory_validation"
                        ),
                        "failure_reason": (
                            None
                            if candidate_pass
                            else dry_summary.get("failure_reason") or metrics.get("reason")
                        ),
                        "trajectory_metrics": metrics,
                        "stages": dry_summary.get("stages", []),
                    }
                )
                candidate_records.append(record)
                if candidate_pass:
                    valid_in_group.append(
                        {
                            "score": float(
                                metrics["total_joint_path_length_rad"]
                            ),
                            "fixed_order": fixed_order,
                            "payload": payload,
                            "prepared_stage_plans": prepared,
                            "record": record,
                            "target_pos_world": np.asarray(
                                target_pos_world,
                                dtype=float,
                            ).tolist(),
                        }
                    )
            group_records.append(
                {
                    "group_name": group_name,
                    "group_index": int(group_index),
                    "candidate_count": int(len(group_candidates)),
                    "valid_candidate_count": int(len(valid_in_group)),
                    "candidate_record_start": int(group_start),
                    "candidate_record_end": int(len(candidate_records)),
                }
            )
            if valid_in_group:
                selected = min(
                    valid_in_group,
                    key=lambda item: (item["score"], item["fixed_order"]),
                )
                selected["record"]["selected"] = True
                break

        selection = {
            "enabled": True,
            "mode": "grouped_world_place_candidates",
            "group_order": [
                str(dict(group).get("group_name", index))
                for index, group in enumerate(candidate_groups)
            ],
            "fallback_policy": "next_group_only_when_current_group_has_no_valid_plan",
            "scoring_metric": "total_joint_path_length_rad",
            "tie_break": "fixed_candidate_order",
            "attempted_group_count": int(len(group_records)),
            "attempted_candidate_count": int(len(candidate_records)),
            "groups": group_records,
            "candidates": candidate_records,
        }
        if selected is None:
            selection.update(
                {
                    "pass": False,
                    "reason": "no_valid_place_candidate",
                    "selected_candidate": None,
                }
            )
            return {
                "primitive": "place_at_candidates",
                "object_name": object_name,
                "target": target_name,
                "placement_candidate_selection": selection,
                "release_gripper": {
                    "skipped": True,
                    "reason": "no_valid_place_candidate",
                    "steps": 0,
                },
                "stages": [],
                "checks": {
                    "candidate_selection": False,
                    "release_command": False,
                },
                "dry_run": not self.runtime_config.execute,
                "pass": False,
                "reason": "no_valid_place_candidate",
            }

        selection.update(
            {
                "pass": True,
                "reason": "shortest_valid_joint_path_in_first_valid_group",
                "selected_candidate": {
                    key: value
                    for key, value in selected["record"].items()
                    if key != "stages"
                },
            }
        )
        result = self._execute_place_payload(
            object_name,
            target_label=target_name,
            target_payload=selected["payload"],
            primitive="place_at_candidates",
            collision_options=collision_options,
            place_kwargs=kwargs,
            prepared_stage_plans=selected["prepared_stage_plans"],
            pre_release_check=pre_release_check,
            place_trajectory_stop_check=kwargs.get("place_trajectory_stop_check"),
        )
        result["target_pos_world"] = selected["target_pos_world"]
        result["placement_candidate_selection"] = selection
        return result

    def _select_world_place_payload(
        self,
        *,
        object_name: str,
        target_pos_world: np.ndarray,
        collision_options: dict[str, Any] | None,
        place_kwargs: dict[str, float],
        obs: dict[str, Any] | None = None,
        current_eef_world: np.ndarray | None = None,
        current_object_pos_world: np.ndarray | None = None,
        current_object_pose_world: np.ndarray | None = None,
        placement_reference: str | None = None,
        placement_object_upright: bool = False,
        target_eef_rot_world: np.ndarray | None = None,
        placement_yaw_offset_deg: Any | None = None,
        placement_yaw_offsets_deg: Any | None = None,
        placement_approach_angles_deg: Any | None = None,
        placement_entry_standoff: Any | None = None,
        placement_exit_standoff: Any | None = None,
        placement_entry_z_offset: Any | None = None,
        placement_exit_z_offset: Any | None = None,
        post_release_clearance_direction_world: Any | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        yaw_requested = (
            placement_yaw_offset_deg is not None
            or placement_yaw_offsets_deg is not None
        )
        approach_requested = placement_approach_angles_deg is not None
        payloads = self._make_world_place_payload_candidates(
            object_name=object_name,
            target_pos_world=target_pos_world,
            obs=obs,
            current_eef_world=current_eef_world,
            current_object_pos_world=current_object_pos_world,
            current_object_pose_world=current_object_pose_world,
            placement_reference=placement_reference,
            target_eef_rot_world=target_eef_rot_world,
            placement_object_upright=placement_object_upright,
            placement_yaw_offset_deg=placement_yaw_offset_deg,
            placement_yaw_offsets_deg=placement_yaw_offsets_deg,
            placement_approach_angles_deg=placement_approach_angles_deg,
            placement_entry_standoff=placement_entry_standoff,
            placement_exit_standoff=placement_exit_standoff,
            placement_entry_z_offset=placement_entry_z_offset,
            placement_exit_z_offset=placement_exit_z_offset,
            post_release_clearance_direction_world=(
                post_release_clearance_direction_world
            ),
            place_kwargs=place_kwargs,
        )
        if not payloads:
            raise ValueError("no valid world-place target pose candidates")
        if not yaw_requested and not approach_requested:
            return (
                payloads[0],
                {
                    "enabled": False,
                    "reason": "no_placement_sampling",
                    "selected_yaw_offset_deg": None,
                    "selected_approach_angle_deg": None,
                    "candidate_count": 0,
                },
            )

        candidates: list[dict[str, Any]] = []
        first_payload: dict[str, Any] | None = None
        first_selection: dict[str, Any] | None = None
        for payload in payloads:
            yaw_offset = payload.get("placement_yaw_offset_deg")
            approach_angle = payload.get("placement_approach_angle_deg")
            dry_summary = self._dry_run_place_payload_entry_sequence(
                payload,
                object_name=object_name,
                collision_options=collision_options,
            )
            candidate = {
                "yaw_offset_deg": None if yaw_offset is None else float(yaw_offset),
                "approach_angle_deg": (
                    None if approach_angle is None else float(approach_angle)
                ),
                "pass": bool(dry_summary.get("pass", False)),
                "failure_phase": dry_summary.get("failure_phase"),
                "failure_reason": dry_summary.get("failure_reason"),
                "stages": dry_summary.get("stages"),
            }
            candidates.append(candidate)
            selection = {
                "enabled": True,
                "mode": "placement_pose_candidates",
                "selected_yaw_offset_deg": (
                    None if yaw_offset is None else float(yaw_offset)
                ),
                "selected_approach_angle_deg": (
                    None if approach_angle is None else float(approach_angle)
                ),
                "candidate_count": len(payloads),
                "candidates": list(candidates),
            }
            if first_payload is None:
                first_payload = payload
                first_selection = dict(selection)
            if candidate["pass"]:
                selection["reason"] = "first_passing_candidate"
                return payload, selection

        assert first_payload is not None and first_selection is not None
        first_selection["reason"] = "no_passing_candidate_using_first"
        first_selection["candidates"] = candidates
        return first_payload, first_selection

    def _dry_run_place_payload_entry_sequence(
        self,
        payload: dict[str, Any],
        *,
        object_name: str,
        collision_options: dict[str, Any] | None,
        capture_prepared_plans: bool = False,
    ) -> dict[str, Any]:
        snapshot = self._virtual_state_snapshot()
        stages: list[dict[str, Any]] = []
        prepared_stage_plans: dict[str, dict[str, Any]] = {}
        try:
            stage_sequence = []
            if "entry" in payload.get("poses", {}):
                stage_sequence.append("entry")
            stage_sequence.append("place")
            for stage_name in stage_sequence:
                stage = self.stage_executor.run(
                    payload["poses"][stage_name]["base"],
                    stage=stage_name,
                    gripper_action=1.0,
                    holding_object=object_name,
                    collision_options=collision_options,
                    execute=False,
                    commit_virtual_state=True,
                    capture_prepared_plan=bool(capture_prepared_plans),
                )
                stages.append(stage)
                prepared = stage.get("prepared_plan")
                if isinstance(prepared, dict):
                    prepared_stage_plans[stage_name] = prepared
                if not bool(stage.get("pass", False)):
                    break
        finally:
            self._restore_virtual_state_snapshot(snapshot)
        checks = {
            str(stage.get("stage")): bool(stage.get("pass", False))
            for stage in stages
        }
        first_failed = next(
            (stage for stage in stages if not bool(stage.get("pass", False))),
            None,
        )
        return {
            "primitive": "dry_run_place_payload_entry_sequence",
            "stages": [
                {
                    "stage": stage.get("stage"),
                    "pass": bool(stage.get("pass", False)),
                    "failure_reason": (
                        None
                        if bool(stage.get("pass", False))
                        else stage.get("failure_reason")
                        or stage.get("reason")
                        or (stage.get("plan") or {}).get("status")
                    ),
                    "plan_status": (stage.get("plan") or {}).get("status"),
                    "trajectory_len": (stage.get("plan") or {}).get("trajectory_len"),
                    "plan_attempts": (stage.get("plan") or {}).get("attempts", []),
                    "collision_preflight": stage.get("collision_preflight"),
                    "target_eef_site_in_base": stage.get(
                        "target_eef_site_in_base"
                    ),
                }
                for stage in stages
            ],
            "checks": checks,
            "dry_run": True,
            "pass": bool(stages) and all(checks.values()),
            "failure_phase": None if first_failed is None else first_failed.get("stage"),
            "failure_reason": (
                None
                if first_failed is None
                else first_failed.get("failure_reason")
                or first_failed.get("reason")
                or (first_failed.get("plan") or {}).get("status")
            ),
            **(
                {"prepared_stage_plans": prepared_stage_plans}
                if capture_prepared_plans
                else {}
            ),
        }

    def _make_world_place_payload_candidates(
        self,
        *,
        object_name: str,
        target_pos_world: np.ndarray,
        obs: dict[str, Any] | None,
        current_eef_world: np.ndarray | None,
        current_object_pos_world: np.ndarray | None,
        current_object_pose_world: np.ndarray | None,
        placement_reference: str | None,
        target_eef_rot_world: np.ndarray | None,
        placement_object_upright: bool,
        placement_yaw_offset_deg: Any | None,
        placement_yaw_offsets_deg: Any | None,
        placement_approach_angles_deg: Any | None,
        placement_entry_standoff: Any | None,
        placement_exit_standoff: Any | None,
        placement_entry_z_offset: Any | None,
        placement_exit_z_offset: Any | None,
        post_release_clearance_direction_world: Any | None = None,
        place_kwargs: dict[str, float],
    ) -> list[dict[str, Any]]:
        current_object_pos = current_object_pos_world
        current_object_pose = current_object_pose_world
        if current_object_pos is None:
            current_object_pos = self._virtual_object_pos_world(object_name)
        if current_object_pose is None:
            current_object_pose = self._virtual_object_pose_world(object_name)
        if current_object_pos is None:
            current_object_pos = self._object_pos_world(object_name)
        return make_world_place_target_pose_candidates(
            self.env,
            self.obs if obs is None else obs,
            object_name=object_name,
            target_pos_world=target_pos_world,
            current_eef_world=(
                self.stage_executor.current_eef_site_world()
                if current_eef_world is None
                else current_eef_world
            ),
            current_object_pos_world=current_object_pos,
            current_object_pose_world=current_object_pose,
            placement_reference=placement_reference,
            target_eef_rot_world=target_eef_rot_world,
            placement_object_upright=placement_object_upright,
            placement_yaw_offset_deg=placement_yaw_offset_deg,
            placement_yaw_offsets_deg=placement_yaw_offsets_deg,
            placement_approach_angles_deg=placement_approach_angles_deg,
            placement_entry_standoff=placement_entry_standoff,
            placement_exit_standoff=placement_exit_standoff,
            placement_entry_z_offset=(
                0.0 if placement_entry_z_offset is None else float(placement_entry_z_offset)
            ),
            placement_exit_z_offset=placement_exit_z_offset,
            post_release_clearance_direction_world=(
                post_release_clearance_direction_world
            ),
            **place_kwargs,
        )

    def return_arm_to_regular_pose(
        self,
        *,
        stage: str = "return_to_regular_pose",
        gripper_action: float | None = None,
        arm_qpos: list[float] | np.ndarray | None = None,
        settle_steps: int | None = None,
        qpos_threshold: float | None = None,
        post_step_callback: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        gripper_action_value = (
            1.0
            if gripper_action is None and self._virtual_held_object is not None
            else (-1.0 if gripper_action is None else float(gripper_action))
        )
        target_qpos = (
            np.array(arm_qpos, dtype=float)
            if arm_qpos is not None
            else self.stage_executor.regular_right_arm_qpos()
        )
        resolved_settle_steps = (
            self.runtime_config.settle_steps
            if settle_steps is None
            else max(0, int(settle_steps))
        )
        resolved_qpos_threshold = (
            self.runtime_config.qpos_threshold
            if qpos_threshold is None
            else float(qpos_threshold)
        )
        if not np.isfinite(resolved_qpos_threshold) or resolved_qpos_threshold <= 0.0:
            raise ValueError("qpos_threshold must be a finite positive value")
        if not self.runtime_config.execute:
            finite = bool(np.all(np.isfinite(target_qpos)))
            if finite:
                self.stage_executor.set_virtual_arm_qpos(target_qpos)
            return {
                "primitive": "return_arm_to_regular_pose",
                "stage": stage,
                "strategy": "reset_to_regular_arm_qpos",
                "target_qpos": target_qpos.astype(float).tolist(),
                "gripper_action": float(gripper_action_value),
                "settle_steps": int(resolved_settle_steps),
                "qpos_threshold": float(resolved_qpos_threshold),
                "dry_run": True,
                "checks": {"target_qpos_finite": finite},
                "pass": finite,
            }
        execution = self.driver.execute_trajectory(
            [target_qpos.astype(float).tolist()],
            max_joint_step=self.runtime_config.max_joint_step,
            settle_steps=int(resolved_settle_steps),
            gripper_action=float(gripper_action_value),
            post_step_callback=post_step_callback,
        )
        self._sync_obs(execution["obs"])
        self.stage_executor.clear_virtual_state()
        qpos_ok = float(execution["final_qpos_error"]) < resolved_qpos_threshold
        runtime_collision_free = not bool(execution.get("collision", {}).get("collision", False))
        return {
            "primitive": "return_arm_to_regular_pose",
            "stage": stage,
            "strategy": "reset_to_regular_arm_qpos",
            "target_qpos": target_qpos.astype(float).tolist(),
            "gripper_action": float(gripper_action_value),
            "settle_steps": int(resolved_settle_steps),
            "qpos_threshold": float(resolved_qpos_threshold),
            "execution": self._without_obs(execution),
            "checks": {
                "final_qpos": qpos_ok,
                "runtime_collision_free": runtime_collision_free,
            },
            "pass": bool(qpos_ok and runtime_collision_free),
        }

    def _execute_place_payload(
        self,
        object_name: str,
        *,
        target_label: str,
        target_payload: dict[str, Any],
        primitive: str,
        collision_options: dict[str, Any] | None = None,
        place_kwargs: dict[str, Any] | None = None,
        prepared_stage_plans: Mapping[str, dict[str, Any]] | None = None,
        pre_release_check: PreReleaseCheck | None = None,
        place_trajectory_stop_check: TrajectoryStopCheck | None = None,
    ) -> dict[str, Any]:
        place_kwargs = dict(place_kwargs or {})
        prepared_mode = prepared_stage_plans is not None
        prepared = dict(prepared_stage_plans or {})
        entry = None
        if "entry" in target_payload.get("poses", {}):
            entry = (
                self.stage_executor.run(
                    target_payload["poses"]["entry"]["base"],
                    stage="entry",
                    gripper_action=1.0,
                    holding_object=object_name,
                    collision_options=collision_options,
                    record_trajectory=True,
                    settle_steps=_stage_settle_steps(
                        place_kwargs,
                        "entry",
                        fallback_key="place_settle_steps",
                    ),
                    prepared_plan=prepared.get("entry"),
                )
                if not prepared_mode or "entry" in prepared
                else skipped_stage("entry", "selected_candidate_missing_prepared_plan")
            )
            self._accept_place_entry_near_miss(entry, kwargs=place_kwargs)
        place = (
            (
                self.stage_executor.run(
                    target_payload["poses"]["place"]["base"],
                    stage="place",
                    gripper_action=1.0,
                    holding_object=object_name,
                    collision_options=collision_options,
                    record_trajectory=True,
                    settle_steps=_stage_settle_steps(
                        place_kwargs,
                        "place",
                        fallback_key="place_settle_steps",
                    ),
                    prepared_plan=prepared.get("place"),
                    trajectory_stop_check=place_trajectory_stop_check,
                )
                if not prepared_mode or "place" in prepared
                else skipped_stage("place", "selected_candidate_missing_prepared_plan")
            )
            if entry is None or bool(entry.get("pass", False))
            else skipped_stage("place", "entry_failed")
        )
        self._accept_place_near_miss(place, kwargs=place_kwargs)
        pre_release = self._evaluate_pre_release_check(
            pre_release_check,
            place_passed=bool(place.get("pass", False)),
        )
        release_ready = bool(place.get("pass", False)) and bool(
            pre_release.get("pass", False)
        )
        release_result = (
            self._run_release_settle(
                release_steps=place_kwargs.get("release_steps"),
                release_settle_steps=place_kwargs.get("release_settle_steps"),
            )
            if release_ready
            else {
                "dry_run": not self.runtime_config.execute,
                "skipped": True,
                "steps": 0,
                "reason": (
                    "place_failed"
                    if not bool(place.get("pass", False))
                    else str(pre_release.get("reason") or "pre_release_check_failed")
                ),
            }
        )
        release_completed = bool(
            release_ready
            and (
                not self.runtime_config.execute
                or not bool(release_result.get("skipped", False))
            )
        )
        clearance = None
        if "clearance" in target_payload.get("poses", {}):
            clearance = (
                self.stage_executor.run(
                    target_payload["poses"]["clearance"]["base"],
                    stage="clearance",
                    gripper_action=-1.0,
                    holding_object=None,
                    record_trajectory=False,
                    settle_steps=_stage_settle_steps(
                        place_kwargs,
                        "clearance",
                        fallback_key="place_settle_steps",
                    ),
                )
                if release_completed
                else skipped_stage("clearance", "release_not_completed")
            )
            retreat = (
                self.stage_executor.run(
                    target_payload["poses"]["retreat"]["base"],
                    stage="retreat",
                    gripper_action=-1.0,
                    holding_object=None,
                    settle_steps=_stage_settle_steps(
                        place_kwargs,
                        "retreat",
                        fallback_key="place_settle_steps",
                    ),
                )
                if bool(clearance.get("pass", False))
                else skipped_stage("retreat", "clearance_failed")
            )
        else:
            retreat = (
                self._reverse_place_retreat(
                    place,
                    settle_steps=_stage_settle_steps(
                        place_kwargs,
                        "retreat",
                        fallback_key="place_settle_steps",
                    ),
                )
                if release_completed
                else skipped_stage("retreat", "release_not_completed")
            )
        stages = []
        if entry is not None:
            stages.append(self._without_stage_trajectory(entry))
        stages.append(self._without_stage_trajectory(place))
        if clearance is not None:
            stages.append(self._without_stage_trajectory(clearance))
        stages.append(retreat)
        checks = self._stage_checks(stages, command_name="release", command=release_result)
        checks["pre_release_check"] = bool(pre_release.get("pass", False))
        if bool(all(checks.values())):
            if not bool(self.runtime_config.execute):
                self._set_virtual_object_static_pos(
                    object_name,
                    target_payload.get("target_object_pos_world"),
                )
            self._clear_virtual_held_object(object_name)
        return {
            "primitive": primitive,
            "object_name": object_name,
            "target": target_label,
            "target_poses": target_payload["poses"],
            "pre_release_check": pre_release,
            "release_gripper": self._without_obs(release_result),
            "stages": stages,
            "checks": checks,
            "dry_run": not self.runtime_config.execute,
            "pass": bool(all(checks.values())),
        }

    @staticmethod
    def _evaluate_pre_release_check(
        check: PreReleaseCheck | None,
        *,
        place_passed: bool,
    ) -> dict[str, Any]:
        if not place_passed:
            return {
                "enabled": check is not None,
                "pass": False,
                "skipped": True,
                "reason": "place_failed",
            }
        if check is None:
            return {
                "enabled": False,
                "pass": True,
                "reason": "no_pre_release_check",
            }
        try:
            result = dict(check())
        except Exception as exc:
            return {
                "enabled": True,
                "pass": False,
                "reason": f"pre_release_check_error:{type(exc).__name__}: {exc}",
            }
        result["enabled"] = True
        result["pass"] = bool(result.get("pass", False))
        if not result["pass"]:
            result.setdefault("reason", "pre_release_check_failed")
        return result

    @staticmethod
    def _accept_place_entry_near_miss(
        entry_stage: dict[str, Any],
        *,
        kwargs: dict[str, Any],
    ) -> None:
        """Allow an explicitly configured pre-place entry to continue by pose.

        The release pose keeps its normal strict validation. This exception is
        only for a collision-free entry motion that reached a safe Cartesian
        neighborhood but missed the controller's joint tracking threshold.
        """
        if not _truthy(kwargs.get("allow_place_entry_near_miss"), default=False):
            return
        if bool(entry_stage.get("pass", False)) or entry_stage.get("stage") != "entry":
            return
        checks = entry_stage.get("checks") or {}
        execution = entry_stage.get("execution") or {}
        position_error = float(execution.get("final_position_error_m", float("inf")))
        orientation_error = float(
            execution.get("final_orientation_error_deg", float("inf"))
        )
        max_position_error = float(
            kwargs.get("place_entry_near_miss_max_position_error", 0.06)
        )
        max_orientation_error = float(
            kwargs.get("place_entry_near_miss_max_orientation_error_deg", 10.0)
        )
        if not bool(checks.get("plan_success", False)):
            return
        if not bool(checks.get("runtime_collision_free", False)):
            return
        if position_error > max_position_error or orientation_error > max_orientation_error:
            return
        entry_stage["accepted_near_miss_for_place_entry"] = {
            "reason": "safe_pre_place_pose_region_reached",
            "final_position_error_m": position_error,
            "max_position_error_m": max_position_error,
            "final_orientation_error_deg": orientation_error,
            "max_orientation_error_deg": max_orientation_error,
        }
        checks["final_position"] = True
        checks["final_orientation"] = True
        checks["final_qpos"] = True
        entry_stage["checks"] = checks
        entry_stage["failure_phase"] = None
        entry_stage["failure_reason"] = None
        entry_stage["pass"] = True

    @staticmethod
    def _without_stage_trajectory(stage: dict[str, Any]) -> dict[str, Any]:
        return {
            key: value
            for key, value in stage.items()
            if key not in {"executed_trajectory", "planned_trajectory"}
        }

    def _lift_current_eef_pose(
        self,
        object_name: str,
        *,
        stage: str,
        lift_height: float,
        gripper_action: float = 1.0,
    ) -> dict[str, Any]:
        target_world = np.array(self.stage_executor.current_eef_site_world(), copy=True)
        target_world[:3, 3] = target_world[:3, 3] + np.array(
            [0.0, 0.0, float(lift_height)],
            dtype=float,
        )
        base_world = matrix_from_base_obs(self.obs)
        target_base = pose_from_matrix(np.linalg.inv(base_world) @ target_world)
        result = self.run_planned_stage(
            target_base,
            stage=stage,
            gripper_action=float(gripper_action),
            holding_object=object_name,
        )
        result["strategy"] = "lift_current_eef_pose"
        result["lift_height"] = float(lift_height)
        return result

    def _reverse_pick_retreat(
        self,
        grasp_stage: dict[str, Any],
        *,
        gripper_action: float = 1.0,
    ) -> dict[str, Any]:
        reverse_trajectory = self._reverse_joint_trajectory_from_stage(grasp_stage)
        if len(reverse_trajectory) < 2:
            return {
                "stage": "retreat",
                "strategy": "reverse_grasp_trajectory",
                "pass": False,
                "failure_phase": "semantic",
                "side_effect_boundary": "none",
                "failure_reason": "grasp_trajectory_unavailable",
                "checks": {
                    "reverse_trajectory_available": False,
                    "final_qpos": False,
                    "runtime_collision_free": False,
                },
                "trajectory_len": len(reverse_trajectory),
            }

        retreat = self.execute_arm_trajectory(
            reverse_trajectory,
            stage="retreat",
            gripper_action=float(gripper_action),
        )
        retreat["strategy"] = "reverse_grasp_trajectory"
        retreat["trajectory_len"] = len(reverse_trajectory)
        checks = dict(retreat.get("checks") or {})
        checks["reverse_trajectory_available"] = True
        retreat["checks"] = checks
        if not bool(retreat.get("pass", False)) and retreat.get("failure_reason") is None:
            retreat["failure_reason"] = (
                "final_qpos"
                if not bool(checks.get("final_qpos", False))
                else "runtime_collision"
            )
        return retreat

    def _reverse_place_retreat(
        self,
        place_stage: dict[str, Any],
        *,
        settle_steps: int | None = None,
    ) -> dict[str, Any]:
        reverse_trajectory = self._reverse_joint_trajectory_from_stage(place_stage)
        if len(reverse_trajectory) < 2:
            return {
                "stage": "retreat",
                "strategy": "reverse_place_trajectory",
                "pass": False,
                "failure_phase": "semantic",
                "side_effect_boundary": "none",
                "failure_reason": "place_trajectory_unavailable",
                "checks": {
                    "reverse_trajectory_available": False,
                    "final_qpos": False,
                    "runtime_collision_free": False,
                },
                "trajectory_len": len(reverse_trajectory),
            }

        retreat = self.execute_arm_trajectory(
            reverse_trajectory,
            stage="retreat",
            gripper_action=-1.0,
            settle_steps=settle_steps,
        )
        retreat["strategy"] = "reverse_place_trajectory"
        retreat["trajectory_len"] = len(reverse_trajectory)
        checks = dict(retreat.get("checks") or {})
        checks["reverse_trajectory_available"] = True
        retreat["checks"] = checks
        if not bool(retreat.get("pass", False)) and retreat.get("failure_reason") is None:
            retreat["failure_reason"] = (
                "final_qpos"
                if not bool(checks.get("final_qpos", False))
                else "runtime_collision"
            )
        return retreat

    def _reverse_joint_trajectory_from_stage(
        self,
        stage_summary: dict[str, Any],
    ) -> list[list[float]]:
        joint_count = int(np.array(self.driver.current_arm_qpos(), dtype=float).size)
        trajectory = (
            stage_summary.get("executed_trajectory")
            or stage_summary.get("planned_trajectory")
            or []
        )
        reversed_waypoints: list[list[float]] = []
        for qpos in reversed(trajectory):
            qpos_array = np.array(qpos, dtype=float)
            if qpos_array.shape != (joint_count,) or not np.all(np.isfinite(qpos_array)):
                continue
            reversed_waypoints.append(qpos_array.astype(float).tolist())
        return reversed_waypoints

    def _run_gripper(
        self,
        action: float,
        steps: int,
        *,
        post_step_callback: Any | None = None,
    ) -> dict[str, Any]:
        if not self.runtime_config.execute:
            return {"dry_run": True, "skipped": True, "steps": 0}
        result = self.driver.run_gripper(
            arm_qpos=self.driver.current_arm_qpos(),
            gripper_action=float(action),
            steps=int(steps),
            post_step_callback=post_step_callback,
        )
        self._sync_obs(result["obs"])
        return result

    def _run_release_settle(
        self,
        *,
        release_steps: int | None = None,
        release_settle_steps: int | None = None,
    ) -> dict[str, Any]:
        resolved_release_steps = (
            self.runtime_config.release_steps
            if release_steps is None
            else max(0, int(release_steps))
        )
        resolved_settle_steps = (
            self.runtime_config.release_settle_steps
            if release_settle_steps is None
            else max(0, int(release_settle_steps))
        )
        release = self._run_gripper(-1.0, int(resolved_release_steps))
        if self.runtime_config.execute:
            for _ in range(int(resolved_settle_steps)):
                self._run_gripper(-1.0, 1)
        release["release_settle_steps"] = int(resolved_settle_steps)
        return release

    def _sync_obs(self, obs: dict[str, Any]) -> None:
        self.obs = obs
        self.stage_executor.obs = obs
        self.driver.obs = obs

    def _stage_checks(
        self,
        stages: list[dict[str, Any]],
        *,
        command_name: str,
        command: dict[str, Any],
    ) -> dict[str, bool]:
        checks = {stage["stage"]: bool(stage.get("pass", False)) for stage in stages}
        command_ok = (not self.runtime_config.execute) or not bool(command.get("skipped"))
        checks[f"{command_name}_command"] = command_ok
        return checks

    def _pick_grasp_validation(
        self,
        object_name: str,
        *,
        initial_object_pos: np.ndarray,
    ) -> dict[str, Any]:
        final_object_pos = self._object_pos_world(object_name)
        object_delta = final_object_pos - initial_object_pos
        lift_threshold = float(self.pick_place_config.lift_delta_threshold)
        result: dict[str, Any] = {
            "initial_object_pos": initial_object_pos.astype(float).tolist(),
            "final_object_pos": final_object_pos.astype(float).tolist(),
            "object_delta": object_delta.astype(float).tolist(),
            "object_lift_delta": float(object_delta[2]),
            "object_lift_threshold": lift_threshold,
            "object_lifted": (not self.runtime_config.execute)
            or float(object_delta[2]) >= lift_threshold,
        }
        try:
            result["contact"] = bool(
                self.env.check_contact(
                    self.env.robots[0].gripper["right"],
                    self.env.objects[object_name],
                )
            )
        except Exception as exc:
            result["contact_error"] = f"{type(exc).__name__}: {exc}"
        try:
            from robocasa.utils import object_utils as OU

            for threshold in (0.035, 0.6):
                try:
                    result[f"grasped_threshold_{threshold:g}"] = bool(
                        OU.check_obj_grasped(
                            self.env,
                            object_name,
                            threshold=threshold,
                        )
                    )
                except Exception as exc:
                    result[f"grasped_threshold_{threshold:g}_error"] = (
                        f"{type(exc).__name__}: {exc}"
                    )
        except Exception as exc:
            result["grasp_check_error"] = f"{type(exc).__name__}: {exc}"
        return result

    def _resolve_place_entry_target(
        self,
        anchor: str,
        *,
        holding: str | None,
        kwargs: dict[str, Any],
    ) -> dict[str, Any] | None:
        object_name = holding or self._virtual_held_object or kwargs.get("object_name") or anchor
        request = {
            "env": self.env,
            "obs": self.obs,
            "anchor": anchor,
            "purpose": "place",
            "holding": holding,
            "object_name": object_name,
            "kwargs": dict(kwargs),
            "controller": self,
        }
        if self.reachability_target_resolver is not None:
            resolved = self.reachability_target_resolver(request)
            if resolved is not None:
                target = dict(resolved)
                target.setdefault("mode", "world")
                target.setdefault("object_name", object_name)
                target.setdefault("target_name", anchor)
                return target

        if anchor not in getattr(self.env, "objects", {}):
            return None
        try:
            payload = make_receptacle_place_target_poses(
                self.env,
                self.obs,
                object_name=str(object_name),
                receptacle_name=anchor,
                current_eef_world=self.stage_executor.current_eef_site_world(),
                receptacle_pos_world=self._virtual_object_static_pos_world(anchor),
                **self.pick_place_config.place_target_kwargs(
                    place_height=kwargs.get("place_height"),
                    retreat_height=kwargs.get("retreat_height"),
                    retreat_x=kwargs.get("retreat_x"),
                    retreat_y=kwargs.get("retreat_y"),
                    release_offset_x=kwargs.get("release_offset_x"),
                    release_offset_y=kwargs.get("release_offset_y"),
                ),
            )
        except Exception:
            return None
        return {
            "mode": "receptacle",
            "object_name": str(object_name),
            "target_name": anchor,
            "payload": payload,
        }

    @staticmethod
    def _entry_stage_for_purpose(purpose: str) -> str | None:
        if purpose == "pick":
            return "pregrasp"
        if purpose == "place":
            return "place"
        return None

    @staticmethod
    def _dry_run_unavailable_entry(
        stage: str,
        *,
        reason: str,
        anchor: str,
        purpose: str,
        holding: str | None,
    ) -> dict[str, Any]:
        return {
            "stage": stage,
            "pass": False,
            "skipped": True,
            "reason": reason,
            "dry_run": True,
            "purpose": purpose,
            "anchor": anchor,
            "holding": holding,
        }

    @staticmethod
    def _tag_dry_run_entry_summary(
        summary: dict[str, Any],
        *,
        anchor: str,
        purpose: str,
        holding: str | None,
    ) -> dict[str, Any]:
        summary["dry_run"] = True
        summary["purpose"] = purpose
        summary["anchor"] = anchor
        summary["holding"] = holding
        return summary

    @staticmethod
    def _without_obs(result: dict[str, Any]) -> dict[str, Any]:
        return {key: value for key, value in result.items() if key != "obs"}

    def _set_virtual_held_object(
        self,
        object_name: str,
        eef_pose_world: dict[str, Any],
    ) -> None:
        eef_world = matrix_from_json_pose(eef_pose_world)
        self._virtual_held_object = str(object_name)
        self._virtual_object_from_eef = (
            np.linalg.inv(eef_world) @ self._object_pose_world(object_name)
        )

    def _virtual_object_pos_world(self, object_name: str) -> np.ndarray | None:
        pose = self._virtual_object_pose_world(object_name)
        if pose is None:
            return None
        return pose[:3, 3]

    def _virtual_object_pos_world_for_eef(
        self,
        object_name: str,
        eef_world: np.ndarray,
    ) -> np.ndarray | None:
        if self._virtual_held_object != str(object_name):
            return None
        if self._virtual_object_from_eef is None:
            return None
        return (np.array(eef_world, dtype=float) @ self._virtual_object_from_eef)[:3, 3]

    def _virtual_object_pose_world(self, object_name: str) -> np.ndarray | None:
        if self._virtual_held_object != str(object_name):
            return None
        if self._virtual_object_from_eef is None:
            return None
        return self.stage_executor.current_eef_site_world() @ self._virtual_object_from_eef

    def _clear_virtual_held_object(self, object_name: str) -> None:
        if self._virtual_held_object == str(object_name):
            self._virtual_held_object = None
            self._virtual_object_from_eef = None
    def _object_pos_world(self, object_name: str) -> np.ndarray:
        virtual = self._virtual_object_static_pos_world(object_name)
        if virtual is not None:
            return virtual
        return object_pos(self.env, object_name)

    def _object_pose_world(self, object_name: str) -> np.ndarray:
        pose = object_pose_matrix(self.env, object_name)
        static_pos = self._virtual_object_static_pos_world(object_name)
        if static_pos is not None:
            pose[:3, 3] = static_pos
        return pose

    def _virtual_object_static_pos_world(self, object_name: str) -> np.ndarray | None:
        value = getattr(self.env, "_bench_virtual_object_pos_world", {}).get(str(object_name))
        if value is None:
            return None
        arr = np.array(value, dtype=float).reshape(-1)
        if arr.size < 3 or not np.all(np.isfinite(arr[:3])):
            return None
        return arr[:3]

    def _set_virtual_object_static_pos(
        self,
        object_name: str,
        pos_world: Any,
    ) -> None:
        if pos_world is None:
            return
        pos = np.array(pos_world, dtype=float).reshape(-1)
        if pos.size < 3 or not np.all(np.isfinite(pos[:3])):
            return
        positions = dict(getattr(self.env, "_bench_virtual_object_pos_world", {}))
        positions[str(object_name)] = pos[:3].astype(float).tolist()
        self.env._bench_virtual_object_pos_world = positions

    def _obs_with_base_pose(
        self,
        *,
        base_pos: Any | None = None,
        base_quat_xyzw: Any | None = None,
    ) -> dict[str, Any]:
        obs = dict(self.obs)
        if base_pos is not None:
            base_pos_arr = np.array(base_pos, dtype=float).reshape(-1)
            if base_pos_arr.size < 3:
                current = np.array(obs["robot0_base_pos"], dtype=float).reshape(-1)
                base_pos_arr = np.array(
                    [base_pos_arr[0], base_pos_arr[1], current[2]],
                    dtype=float,
                )
            obs["robot0_base_pos"] = base_pos_arr[:3].astype(float)
        if base_quat_xyzw is not None:
            obs["robot0_base_quat"] = np.array(base_quat_xyzw, dtype=float).reshape(-1)[:4]
        return obs

    def _virtual_state_snapshot(self) -> dict[str, Any]:
        return {
            "held_object": self._virtual_held_object,
            "object_from_eef": (
                None
                if self._virtual_object_from_eef is None
                else np.array(self._virtual_object_from_eef, copy=True)
            ),
            "object_positions": copy.deepcopy(
                getattr(self.env, "_bench_virtual_object_pos_world", {})
            ),
            "arm_qpos": (
                None
                if self.stage_executor._virtual_arm_qpos is None
                else np.array(self.stage_executor._virtual_arm_qpos, copy=True)
            ),
            "eef_site_in_base": copy.deepcopy(
                self.stage_executor._virtual_eef_site_in_base
            ),
        }

    def _restore_virtual_state_snapshot(self, snapshot: dict[str, Any]) -> None:
        self._virtual_held_object = snapshot.get("held_object")
        object_from_eef = snapshot.get("object_from_eef")
        self._virtual_object_from_eef = (
            None if object_from_eef is None else np.array(object_from_eef, copy=True)
        )
        self.env._bench_virtual_object_pos_world = copy.deepcopy(
            snapshot.get("object_positions") or {}
        )
        arm_qpos = snapshot.get("arm_qpos")
        self.stage_executor._virtual_arm_qpos = (
            None if arm_qpos is None else np.array(arm_qpos, copy=True)
        )
        self.stage_executor._virtual_eef_site_in_base = copy.deepcopy(
            snapshot.get("eef_site_in_base")
        )

    @staticmethod
    def _dry_run_pick_place_summary(
        stages: list[dict[str, Any]],
        *,
        object_name: str,
    ) -> dict[str, Any]:
        checks = {str(stage.get("stage")): bool(stage.get("pass", False)) for stage in stages}
        first_failed = next((stage for stage in stages if not bool(stage.get("pass", False))), None)
        return {
            "primitive": "dry_run_pick_then_world_place",
            "object_name": object_name,
            "stages": stages,
            "checks": checks,
            "dry_run": True,
            "pass": bool(stages) and all(checks.values()),
            "failure_phase": None if first_failed is None else first_failed.get("stage"),
            "failure_reason": (
                None
                if first_failed is None
                else first_failed.get("failure_reason")
                or first_failed.get("reason")
                or (first_failed.get("plan") or {}).get("status")
            ),
        }


__all__ = [
    "ManipulationController",
    "ManipulationRuntimeConfig",
    "prepared_stage_plan_trajectory_metrics",
    "topdown_eef_rotation_world_for_base_yaw",
]
