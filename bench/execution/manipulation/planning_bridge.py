"""Planning-stage bridge used by the independent manipulation controller."""

from __future__ import annotations

import copy
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from bench.execution.manipulation.drivers import (
    ArmDriver,
    PostStepCallback,
    TrajectoryStopCheck,
    qpos_for_controller,
)
from bench.execution.planning import (
    build_static_scene_point_cloud,
    make_collision_monitor,
    matrix_from_base_obs,
    matrix_from_json_pose,
    matrix_from_site_pose,
    parse_keyword_csv,
    pose_error,
    pose_from_matrix,
    preflight_arm_trajectory_collisions,
)
from bench.execution.planning.sim_state import preserved_integration_state


PlannerFn = Callable[[dict[str, Any]], dict[str, Any]]


@dataclass(frozen=True)
class ManipulationRuntimeConfig:
    """Execution parameters for independent manipulation primitives."""

    execute: bool = True
    max_joint_step: float = 0.025
    settle_steps: int = 2
    position_threshold: float = 0.03
    orientation_threshold_deg: float = 8.0
    qpos_threshold: float = 0.05
    open_steps: int = 20
    close_steps: int = 90
    release_steps: int = 45
    release_settle_steps: int = 80
    planner_args: dict[str, Any] = field(default_factory=dict)
    static_scene_options: dict[str, Any] = field(default_factory=dict)


def default_planner(payload: dict[str, Any]) -> dict[str, Any]:
    from bench.execution.planning.mplib_planner import plan_tcp_target_in_mplib

    return plan_tcp_target_in_mplib(payload)


def stage_failure_reason(summary: dict[str, Any]) -> str | None:
    if summary.get("pass", False):
        return None
    if summary.get("failure_phase") == "plan":
        plan = summary.get("plan") or {}
        return str(plan.get("status") or "plan_failed")
    if summary.get("failure_phase") == "preflight":
        return "preflight_collision"
    for name, ok in (summary.get("checks") or {}).items():
        if not bool(ok):
            return str(name)
    return "execution_failed"


def skipped_stage(stage: str, reason: str) -> dict[str, Any]:
    return {"stage": stage, "pass": False, "skipped": True, "reason": reason}


def _controller_joint_names(env: Any, controller: Any, *, fallback_prefix: str) -> list[str]:
    joint_ids = getattr(controller, "joint_index", None)
    if joint_ids is not None and hasattr(env.sim.model, "joint_id2name"):
        names = []
        for joint_id in joint_ids:
            try:
                names.append(str(env.sim.model.joint_id2name(int(joint_id))))
            except Exception:
                names = []
                break
        if names:
            return names
    qpos_index = np.array(getattr(controller, "qpos_index", []), dtype=int).reshape(-1)
    return [f"{fallback_prefix}{i + 1}" for i in range(len(qpos_index))]


def _site_name(env: Any, site_id: int) -> str:
    try:
        return str(env.sim.model.site_id2name(int(site_id)))
    except Exception:
        return f"site_{int(site_id)}"


class PlannedStageExecutor:
    """Run one target-pose stage through planner, preflight, and driver."""

    def __init__(
        self,
        env: Any,
        obs: dict[str, Any],
        *,
        driver: ArmDriver,
        planner: PlannerFn,
        runtime_config: ManipulationRuntimeConfig,
        plan_template: dict[str, Any] | None = None,
        model_candidates: list[dict[str, Any]] | None = None,
    ) -> None:
        self.env = env
        self.obs = obs
        self.driver = driver
        self.planner = planner
        self.config = runtime_config
        self.plan_template = copy.deepcopy(plan_template)
        self.model_candidates = copy.deepcopy(model_candidates)
        self._virtual_arm_qpos: np.ndarray | None = None
        self._virtual_eef_site_in_base: dict[str, Any] | None = None

    def run(
        self,
        target_pose_base: dict[str, Any],
        *,
        stage: str,
        gripper_action: float,
        holding_object: str | None = None,
        collision_options: dict[str, Any] | None = None,
        execute: bool | None = None,
        record_trajectory: bool = False,
        commit_virtual_state: bool | None = None,
        post_step_callback: PostStepCallback | None = None,
        trajectory_stop_check: TrajectoryStopCheck | None = None,
        settle_steps: int | None = None,
        prepared_plan: dict[str, Any] | None = None,
        capture_prepared_plan: bool = False,
    ) -> dict[str, Any]:
        stage_execute = self.config.execute if execute is None else bool(execute)
        commit_virtual = (
            not stage_execute if commit_virtual_state is None else bool(commit_virtual_state)
        )
        collision_scene = None
        plan_source = "planner"
        prepared_error = None
        if prepared_plan is None:
            collision_scene = self._collision_scene_for_stage(stage)
            payload = self._stage_plan_payload(target_pose_base, stage, collision_scene)
            plan_payload = self.planner(payload)
        else:
            prepared_error = self._prepared_plan_error(
                prepared_plan,
                stage=stage,
                target_pose_base=target_pose_base,
            )
            plan_payload = (
                copy.deepcopy(prepared_plan.get("planner_result"))
                if isinstance(prepared_plan, dict)
                else None
            )
            plan_source = "prepared"
        plan = plan_payload.get("plan") if isinstance(plan_payload, dict) else None
        summary = {
            "stage": stage,
            "target_eef_site_in_base": target_pose_base,
            "plan_source": plan_source,
            "failure_phase": None,
            "failure_reason": None,
            "side_effect_boundary": "none",
        }
        if prepared_error is not None:
            summary.update(
                {
                    "plan": {"status": "invalid_prepared_plan"},
                    "pass": False,
                    "failure_phase": "plan",
                    "failure_reason": prepared_error,
                }
            )
            return summary
        if collision_scene is not None:
            summary["collision_scene"] = {
                key: value for key, value in collision_scene.items() if key != "points"
            }
        if not isinstance(plan, dict) or plan.get("status") != "Success":
            summary.update({"plan": plan, "pass": False, "failure_phase": "plan"})
            summary["failure_reason"] = stage_failure_reason(summary)
            return summary

        trajectory = plan.get("trajectory", [])
        monitor = self._collision_monitor(holding_object, collision_options=collision_options)
        preflight = self._preflight_if_enabled(trajectory, monitor, summary)
        if preflight is not None and preflight.get("abort", False):
            summary.update(self._plan_summary(plan, plan_payload))
            summary["checks"] = {
                "plan_success": True,
                "preflight_collision_free": False,
                "runtime_collision_free": False,
            }
            summary["pass"] = False
            summary["failure_phase"] = "preflight"
            summary["failure_reason"] = stage_failure_reason(summary)
            return summary

        summary.update(self._plan_summary(plan, plan_payload))
        if capture_prepared_plan:
            summary["prepared_plan"] = self._make_prepared_plan(
                stage=stage,
                target_pose_base=target_pose_base,
                planner_result=plan_payload,
            )
        if record_trajectory:
            summary["planned_trajectory"] = copy.deepcopy(trajectory)
        if not stage_execute:
            summary["dry_run"] = True
            summary["checks"] = {
                "plan_success": True,
                "preflight_collision_free": preflight is None
                or not bool(preflight.get("collision", False)),
            }
            summary["pass"] = True
            if commit_virtual:
                self._set_virtual_state_from_plan(trajectory, target_pose_base)
            return summary

        stage_settle_steps = self.config.settle_steps if settle_steps is None else int(settle_steps)
        execution = self.driver.execute_trajectory(
            trajectory,
            max_joint_step=self.config.max_joint_step,
            settle_steps=stage_settle_steps,
            gripper_action=float(gripper_action),
            collision_monitor=monitor,
            post_step_callback=post_step_callback,
            trajectory_stop_check=trajectory_stop_check,
        )
        self.obs = execution["obs"]
        self.clear_virtual_state()
        summary.update(self._execution_summary(execution, target_pose_base, gripper_action))
        summary["execution"]["settle_steps"] = int(stage_settle_steps)
        if record_trajectory:
            summary["executed_trajectory"] = copy.deepcopy(trajectory)
        return summary

    @staticmethod
    def _make_prepared_plan(
        *,
        stage: str,
        target_pose_base: dict[str, Any],
        planner_result: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "stage": str(stage),
            "target_eef_site_in_base": copy.deepcopy(target_pose_base),
            "planner_result": copy.deepcopy(planner_result),
        }

    @staticmethod
    def _prepared_plan_error(
        prepared_plan: dict[str, Any],
        *,
        stage: str,
        target_pose_base: dict[str, Any],
    ) -> str | None:
        if not isinstance(prepared_plan, dict):
            return "prepared_plan_not_a_mapping"
        if int(prepared_plan.get("schema_version", 0)) != 1:
            return "prepared_plan_schema_mismatch"
        if str(prepared_plan.get("stage")) != str(stage):
            return "prepared_plan_stage_mismatch"
        try:
            prepared_target = matrix_from_json_pose(
                prepared_plan["target_eef_site_in_base"]
            )
            requested_target = matrix_from_json_pose(target_pose_base)
        except Exception:
            return "prepared_plan_target_invalid"
        if not np.allclose(prepared_target, requested_target, atol=1e-9, rtol=0.0):
            return "prepared_plan_target_mismatch"
        if not isinstance(prepared_plan.get("planner_result"), dict):
            return "prepared_plan_result_missing"
        return None

    def collect_sample(self, name: str) -> dict[str, Any]:
        robot = self.env.robots[0]
        composite = robot.composite_controller
        right = composite.part_controllers["right"]
        gripper = composite.part_controllers.get("right_gripper")
        site_id = int(robot.eef_site_id["right"])
        site_world = matrix_from_site_pose(self.env, robot)
        base_world = matrix_from_base_obs(self.obs)
        eef_in_base = np.linalg.inv(base_world) @ site_world
        gripper_qpos = (
            qpos_for_controller(self.env, gripper).astype(float).tolist()
            if gripper is not None
            else []
        )
        sample = {
            "name": name,
            "right_arm_qpos": qpos_for_controller(self.env, right).astype(float).tolist(),
            "right_gripper_qpos": gripper_qpos,
            "base_world": pose_from_matrix(base_world),
            "eef_site_world": pose_from_matrix(site_world),
            "eef_site_in_base": pose_from_matrix(eef_in_base),
            "eef_site_name": _site_name(self.env, site_id),
            "eef_site_id": site_id,
        }
        if self._virtual_arm_qpos is not None and self._virtual_eef_site_in_base is not None:
            virtual_eef_in_base = matrix_from_json_pose(self._virtual_eef_site_in_base)
            sample.update(
                {
                    "right_arm_qpos": self._virtual_arm_qpos.astype(float).tolist(),
                    "eef_site_world": pose_from_matrix(base_world @ virtual_eef_in_base),
                    "eef_site_in_base": copy.deepcopy(self._virtual_eef_site_in_base),
                    "virtual_start_state": True,
                }
            )
        return sample

    def current_eef_site_world(self) -> np.ndarray:
        """Return the active EEF pose, including dry-run virtual pose if present."""
        if self._virtual_eef_site_in_base is None:
            return matrix_from_site_pose(self.env, self.env.robots[0])
        base_world = matrix_from_base_obs(self.obs)
        return base_world @ matrix_from_json_pose(self._virtual_eef_site_in_base)

    def clear_virtual_state(self) -> None:
        self._virtual_arm_qpos = None
        self._virtual_eef_site_in_base = None

    def set_virtual_arm_qpos(self, arm_qpos: np.ndarray) -> None:
        """Set a dry-run virtual arm state and derive its EEF pose from env FK."""
        right = self.env.robots[0].composite_controller.part_controllers["right"]
        qpos = np.array(arm_qpos, dtype=float).reshape(-1)
        current = qpos_for_controller(self.env, right)
        if qpos.shape != current.shape or not np.all(np.isfinite(qpos)):
            return
        with preserved_integration_state(self.env.sim):
            self.env.sim.data.qpos[right.qpos_index] = qpos
            self.env.sim.data.qvel[:] = 0.0
            self.env.sim.forward()
            base_world = matrix_from_base_obs(self.obs)
            eef_world = matrix_from_site_pose(self.env, self.env.robots[0])
            self._virtual_arm_qpos = qpos
            self._virtual_eef_site_in_base = pose_from_matrix(
                np.linalg.inv(base_world) @ eef_world
            )

    def _set_virtual_state_from_plan(
        self,
        trajectory: list[list[float]],
        target_pose_base: dict[str, Any],
    ) -> None:
        if not trajectory:
            return
        qpos = np.array(trajectory[-1], dtype=float).reshape(-1)
        if not np.all(np.isfinite(qpos)):
            return
        self._virtual_arm_qpos = qpos
        self._virtual_eef_site_in_base = copy.deepcopy(target_pose_base)

    def robocasa_payload(self) -> dict[str, Any]:
        robot = self.env.robots[0]
        composite = robot.composite_controller
        right = composite.part_controllers["right"]
        gripper = composite.part_controllers.get("right_gripper")
        site_id = int(robot.eef_site_id["right"])
        return {
            "robot_model_name": getattr(getattr(robot, "robot_model", None), "name", ""),
            "action_split": {
                key: list(value)
                for key, value in composite._action_split_indexes.items()
            },
            "right_arm_joint_names": _controller_joint_names(
                self.env,
                right,
                fallback_prefix="robot0_right_panda_joint",
            ),
            "right_arm_qpos_indexes": list(map(int, right.qpos_index)),
            "right_gripper_joint_names": (
                _controller_joint_names(
                    self.env,
                    gripper,
                    fallback_prefix="robot0_right_panda_finger_joint",
                )
                if gripper is not None
                else []
            ),
            "right_gripper_qpos_indexes": (
                list(map(int, gripper.qpos_index)) if gripper is not None else []
            ),
            "canonical_eef_site_name": _site_name(self.env, site_id),
            "canonical_eef_site_id": site_id,
        }

    def regular_right_arm_qpos(self) -> np.ndarray:
        right = self.env.robots[0].composite_controller.part_controllers["right"]
        current = qpos_for_controller(self.env, right)
        samples = self.plan_template.get("samples", []) if self.plan_template else []
        if samples:
            sample_qpos = np.array(samples[0].get("right_arm_qpos", []), dtype=float)
            if sample_qpos.shape == current.shape and np.all(np.isfinite(sample_qpos)):
                return sample_qpos
        initial_joint = np.array(getattr(right, "initial_joint", []), dtype=float)
        if initial_joint.shape == current.shape and np.all(np.isfinite(initial_joint)):
            return initial_joint
        robot_init = np.array(getattr(self.env.robots[0], "init_qpos", []), dtype=float)
        if robot_init.shape == current.shape and np.all(np.isfinite(robot_init)):
            return robot_init
        return np.array(current, copy=True)

    def _stage_plan_payload(
        self,
        target_pose_base: dict[str, Any],
        stage: str,
        collision_scene: dict[str, Any] | None,
    ) -> dict[str, Any]:
        current_sample = self.collect_sample(f"{stage}_start")
        if self.plan_template is not None:
            payload = copy.deepcopy(self.plan_template)
            payload["samples"] = copy.deepcopy(payload.get("samples", [])[:2])
            payload["samples"].append(current_sample)
            payload.setdefault("robocasa", self.robocasa_payload())
        else:
            payload = {
                "collector_environment": {"source": "bench.execution.manipulation"},
                "robocasa": self.robocasa_payload(),
                "samples": [current_sample],
            }
        payload["planner_args"] = dict(self.config.planner_args)
        payload["target"] = {"stage": stage, "target_eef_site_in_base": target_pose_base}
        if self._virtual_arm_qpos is not None:
            payload["virtual_start_state"] = {
                "enabled": True,
                "right_arm_qpos": self._virtual_arm_qpos.astype(float).tolist(),
                "eef_site_in_base": copy.deepcopy(self._virtual_eef_site_in_base),
            }
        if self.model_candidates is not None:
            payload["model_candidates"] = copy.deepcopy(self.model_candidates)
        if collision_scene is not None:
            payload["collision_scene"] = collision_scene
        return payload

    def _collision_scene_for_stage(self, stage: str) -> dict[str, Any] | None:
        options = self.config.static_scene_options
        if not bool(options.get("enabled", False)):
            return None
        stages = set(parse_keyword_csv(options.get("point_cloud_stages", "lift,retreat")))
        if stages and "all" not in stages and stage.lower() not in stages:
            return None
        return build_static_scene_point_cloud(
            self.env,
            self.obs,
            enabled=True,
            fixture_keywords=options.get("fixture_keywords", "cabinet"),
            exclude_keywords=options.get("exclude_keywords", "floor,wall"),
            spacing=float(options.get("spacing", 0.04)),
            inflate=float(options.get("inflate", 0.05)),
            point_radius=float(options.get("point_radius", 0.01)),
            faces=options.get("faces", "bottom"),
            min_z=options.get("min_z", 1.05),
            max_points=int(options.get("max_points", 4000)),
            workspace_min_base=options.get("workspace_min_base", "-0.2,-0.9,0.2"),
            workspace_max_base=options.get("workspace_max_base", "1.2,0.9,2.1"),
        )

    def _collision_monitor(
        self,
        holding_object: str | None,
        *,
        collision_options: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        options = self.config.static_scene_options
        stage_options = dict(collision_options or {})
        if not bool(
            stage_options.get(
                "validate_collisions",
                options.get("validate_collisions", False),
            )
        ):
            return None
        return make_collision_monitor(
            self.env,
            enabled=True,
            fixture_keywords=stage_options.get(
                "fixture_keywords",
                options.get("fixture_keywords", "cabinet"),
            ),
            exclude_keywords=stage_options.get(
                "exclude_keywords",
                options.get("exclude_keywords", "floor,wall"),
            ),
            abort_on_collision=bool(
                stage_options.get(
                    "abort_on_collision",
                    options.get("abort_on_collision", True),
                )
            ),
            holding_object=holding_object,
            attached_object_proxy=bool(options.get("attached_object_proxy", True)),
            attached_object_proxy_inflate=float(
                options.get("attached_object_proxy_inflate", 0.02)
            ),
            holding_allowed_fixture_keywords=stage_options.get(
                "holding_allowed_fixture_keywords",
                "",
            ),
            protected_object_names=stage_options.get(
                "protected_object_names",
                (),
            ),
        )

    def _preflight_if_enabled(
        self,
        trajectory: list[list[float]],
        monitor: dict[str, Any] | None,
        summary: dict[str, Any],
    ) -> dict[str, Any] | None:
        if monitor is None:
            return None
        result = preflight_arm_trajectory_collisions(
            self.env,
            trajectory,
            max_joint_step=self.config.max_joint_step,
            monitor=monitor,
        )
        summary["collision_preflight"] = result
        result["abort"] = bool(
            result.get("collision", False) and monitor.get("abort_on_collision", True)
        )
        return result

    def _execution_summary(
        self,
        execution: dict[str, Any],
        target_pose_base: dict[str, Any],
        gripper_action: float,
    ) -> dict[str, Any]:
        final_sample = self.collect_sample("stage_final")
        final_pose = matrix_from_json_pose(final_sample["eef_site_in_base"])
        target_pose = matrix_from_json_pose(target_pose_base)
        position_error, orientation_error = pose_error(final_pose, target_pose)
        qpos_ok = float(execution["final_qpos_error"]) < self.config.qpos_threshold
        pose_ok = (
            position_error < self.config.position_threshold
            and orientation_error < self.config.orientation_threshold_deg
        )
        runtime_collision_free = not bool(execution.get("collision", {}).get("collision", False))
        checks = {
            "plan_success": True,
            "final_position": position_error < self.config.position_threshold,
            "final_orientation": orientation_error < self.config.orientation_threshold_deg,
            "final_qpos": qpos_ok,
            "runtime_collision_free": runtime_collision_free,
        }
        summary = {
            "execution": {
                "executed_steps": execution["executed_steps"],
                "max_command_delta": execution["max_command_delta"],
                "final_qpos_error": execution["final_qpos_error"],
                "final_position_error_m": position_error,
                "final_orientation_error_deg": orientation_error,
                "gripper_action": float(gripper_action),
            },
            "final_eef_site_in_base": final_sample["eef_site_in_base"],
            "final_eef_site_world": final_sample["eef_site_world"],
            "checks": checks,
            "side_effect_boundary": "arm_moved",
            "pass": bool(pose_ok and qpos_ok and runtime_collision_free),
        }
        if execution.get("collision") is not None:
            summary["collision_runtime"] = execution["collision"]
        if execution.get("early_stop") is not None:
            summary["execution"]["early_stop"] = execution["early_stop"]
        if not summary["pass"]:
            summary["failure_phase"] = "execution"
            summary["failure_reason"] = stage_failure_reason(summary)
        return summary

    @staticmethod
    def _plan_summary(plan: dict[str, Any], plan_payload: dict[str, Any]) -> dict[str, Any]:
        return {
            "plan": {
                "status": plan.get("status"),
                "attempts": plan.get("attempts", []),
                "trajectory_len": plan.get("trajectory_len", len(plan.get("trajectory", []))),
                "duration": float(plan.get("duration", 0.0)),
            },
            "selected_tcp_candidate": plan_payload.get("selected_tcp_candidate"),
            "mplib_collision_scene": plan_payload.get("collision_scene"),
        }
