"""TM02 manipulation assembly with directly reused WA02 / WR01 primitives."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from bench.execution.manipulation import ManipulationController, ManipulationRuntimeConfig
from bench.execution.contracts import StageKind, StageResult
from bench.execution.planning import MplibPlannerConfig
from bench.execution.planning.mplib_planner import run_mplib_planner_subprocess
from bench.tasks.tm02.domain.contracts import COMPLETION_BUTTON
from bench.tasks.tm02.execution.water import operate_tm02_water_button
from bench.tasks.tm02.execution.completion_button import press_completion_button_physically
from bench.tasks.tm02.execution.door import move_slide_door, return_arm_after_cabinet_open
from bench.tasks.tm02.execution.retrieval import retrieve_item_from_cabinet


SUPPORTED_OPERATIONS = {
    "move_eef",
    "return_arm",
    "operate_water_button",
    "press_completion_button",
    "open_tm02_cabinet",
    "close_tm02_cabinet",
    "pick_can_from_cabinet",
    "place_can_in_marked_region",
}


@dataclass(frozen=True)
class ManipulationAssembly:
    controller: "TM02ManipulationController"
    planner_settings: Mapping[str, Any]


class TM02ManipulationController(ManipulationController):
    """Execute calibrated probe operations and reject uncalibrated ones."""

    def return_arm_to_regular_pose(self, **kwargs: Any) -> dict[str, Any]:
        if (
            kwargs.get("stage") == "return_arm_after_cabinet_open"
            and self.runtime_config.execute
        ):
            return return_arm_after_cabinet_open(
                self,
                lambda: super(TM02ManipulationController, self).return_arm_to_regular_pose(**kwargs),
            )
        return super().return_arm_to_regular_pose(**kwargs)

    def __init__(
        self,
        *args: Any,
        missing_resources: tuple[str, ...],
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.missing_resources = tuple(str(item) for item in missing_resources)

    def execute(self, operation: str, target: str, **kwargs: Any) -> StageResult:
        op = str(getattr(operation, "value", operation))
        if op in SUPPORTED_OPERATIONS:
            return super().execute(op, target, **kwargs)
        reason = "tm02_physical_manipulation_calibration_required"
        return StageResult(
            index=-1,
            kind=StageKind.OPERATE,
            success=False,
            details={
                "primitive": "tm02_calibration_gate",
                "operation": op,
                "target": str(target),
                "missing_resources": list(self.missing_resources),
                "semantic_shortcut_used": False,
                "pass": False,
                "reason": reason,
            },
            failure_code=reason,
        )


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
    """Build TM02 around validated WA02 and WR01 physical primitives."""

    planner = _section(settings, "planner")
    controller = TM02ManipulationController(
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
        missing_resources=missing_resources,
    )
    def water_button(target: str, **kwargs: Any) -> dict[str, Any]:
        del target
        return operate_tm02_water_button(controller, **kwargs)

    def completion_button(target: str, **kwargs: Any) -> dict[str, Any]:
        return _press_tm02_completion_button(
            controller,
            str(target),
            **kwargs,
        )

    def open_cabinet(target: str, **kwargs: Any) -> dict[str, Any]:
        del target
        return _operate_wr01_door(
            controller,
            pose_file=_required_resource(resources, "cabinet_door_pose"),
            opening=True,
            metadata=kwargs,
        )

    def close_cabinet(target: str, **kwargs: Any) -> dict[str, Any]:
        del target
        return _operate_wr01_door(
            controller,
            pose_file=_required_resource(resources, "cabinet_door_pose"),
            opening=False,
            metadata=kwargs,
        )

    def pick_can(target: str, **kwargs: Any) -> dict[str, Any]:
        if not controller.runtime_config.execute:
            return _dry_run("retrieve_item_from_cabinet", str(target))
        controller.refresh_obs()
        fraction = float(env.cabinet_open_fraction())
        threshold = float(env.physical_observer.config.cabinet_open_threshold)
        if fraction < threshold or env.semantic_state.absorbing_failure:
            return {
                "primitive": "retrieve_item_from_cabinet",
                "cabinet_open_fraction_after_navigation": fraction,
                "pass": False,
                "reason": "tm02_cabinet_not_open_after_navigation",
            }
        values = dict(kwargs)
        values.pop("mode", None)
        values.pop("require_physical_grasp", None)
        if "disable_collisions_during_transport" in values:
            raise ValueError(
                "TM02 is action-only; disable_collisions_during_transport is forbidden"
            )
        result = retrieve_item_from_cabinet(
            controller,
            str(target),
            lower_prepose_file=_required_resource(
                resources,
                "can_retrieval_prepose",
            ),
            front_clearance_distance=float(
                values.pop("front_clearance_distance", 0.10)
            ),
            front_pregrasp_distance=float(
                values.pop("front_pregrasp_distance", 0.10)
            ),
            front_clearance_z_offset=float(
                values.pop("front_clearance_z_offset", 0.10)
            ),
            grasp_z_offset=float(values.pop("grasp_z_offset", 0.010)),
            grasp_lateral_offset=float(
                values.pop("grasp_lateral_offset", -0.0067)
            ),
            grasp_inward_offset=float(
                values.pop("grasp_inward_offset", 0.0042)
            ),
            close_gripper_steps=int(values.pop("close_gripper_steps", 6)),
            post_close_hold_steps=int(
                values.pop("post_close_hold_steps", 12)
            ),
            held_gripper_action=float(
                values.pop("held_gripper_action", 0.0)
            ),
        )
        if values:
            result["unused_metadata"] = values
        result["cabinet_open_fraction_after_navigation"] = fraction
        return result

    def place_can(target: str, **kwargs: Any) -> dict[str, Any]:
        if not controller.runtime_config.execute:
            return _dry_run("place_can_in_marked_region", str(target))
        calibration = _load_can_place_calibration(
            _required_resource(resources, "can_place_pose")
        )
        live_target = _tm02_place_region_world(env)
        calibrated_target = calibration["anchor_world_xyz"]
        anchor_error = float(np.linalg.norm(live_target - calibrated_target))
        if anchor_error > 1e-6:
            return {
                "primitive": "place_can_in_marked_region",
                "object_name": str(target),
                "pass": False,
                "reason": "tm02_place_region_differs_from_calibration",
                "anchor_error_m": anchor_error,
                "live_target_world_xyz": live_target.tolist(),
                "calibrated_target_world_xyz": calibrated_target.tolist(),
            }

        values = dict(calibration["placement"])
        values.update(dict(kwargs))
        values.pop("require_release", None)
        values.pop("target_region", None)

        handoff: dict[str, Any] = {"pass": False, "performed": False}

        def release_to_physics() -> dict[str, Any]:
            in_region_before_release = bool(env.can_in_place_region())
            held_object = getattr(controller, "_virtual_held_object", None)
            ready = bool(
                in_region_before_release
                and held_object == str(target)
            )
            handoff.update(
                {
                    "pass": ready,
                    "performed": ready,
                    "can_in_place_region_before_release": in_region_before_release,
                    "held_object": held_object,
                    "mode": "physical_contact_hold",
                    "simulation_state_modified": False,
                }
            )
            if ready:
                controller._clear_virtual_held_object(str(target))
            return {
                "pass": ready,
                "reason": None if ready else "can_not_ready_for_physics_release",
                **handoff,
            }

        result = controller.place_at_world_pose(
            str(target),
            target_name="tm02_can_place_region",
            target_pos_world=live_target,
            pre_release_check=release_to_physics,
            **values,
        )
        controller.refresh_obs()
        final_position = np.asarray(
            env.object_world_position(str(target)), dtype=float
        )
        physical_checks = {
            "can_in_place_region": bool(env.can_in_place_region()),
            "can_released": bool(env.can_released()),
        }
        checks = dict(result.get("checks") or {})
        checks.update(physical_checks)
        result.update(
            {
                "primitive": "place_can_in_marked_region",
                "calibration_file": str(
                    _required_resource(resources, "can_place_pose")
                ),
                "anchor_error_m": anchor_error,
                "physics_handoff": handoff,
                "final_can_world_xyz": final_position.tolist(),
                "physical_checks": physical_checks,
                "checks": checks,
                "pass": bool(result.get("pass", False) and all(physical_checks.values())),
            }
        )
        if not result["pass"] and result.get("reason") is None:
            result["reason"] = "tm02_can_place_physical_validation_failed"
        return result

    controller.operation_handlers = {
        "operate_water_button": water_button,
        "press_completion_button": completion_button,
        "open_tm02_cabinet": open_cabinet,
        "close_tm02_cabinet": close_cabinet,
        "pick_can_from_cabinet": pick_can,
        "place_can_in_marked_region": place_can,
    }
    return ManipulationAssembly(controller=controller, planner_settings=planner)


def _press_tm02_completion_button(
    controller: TM02ManipulationController,
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
    """Physically press TM02's red final-success button."""

    del kwargs
    if button != COMPLETION_BUTTON:
        return {
            "primitive": "physical_tm02_completion_button_press",
            "button": button,
            "semantic_shortcut_used": False,
            "pass": False,
            "reason": "unknown_tm02_completion_button",
        }
    if not controller.runtime_config.execute:
        return press_completion_button_physically(
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

    env = controller.env
    state = env.semantic_state
    preconditions = {
        "a1_complete": bool(state.a1_complete),
        "a2_complete": bool(state.a2_complete),
        "b1_complete": bool(state.b1_complete),
        "b2_complete": bool(state.b2_complete),
        "faucet_off": not bool(env.faucet_is_on()),
        "pot_filled": bool(env.pot_is_filled()),
        "can_released_in_place_region": bool(
            env.can_in_place_region() and env.can_released()
        ),
        "cabinet_closed": bool(
            env.cabinet_open_fraction() <= env.final_closed_threshold
        ),
        "no_absorbing_failure": not bool(state.absorbing_failure),
        "not_previously_pressed": not bool(state.completion_button_pressed),
    }
    if not all(preconditions.values()):
        return {
            "primitive": "physical_tm02_completion_button_press",
            "button": button,
            "preconditions": preconditions,
            "semantic_shortcut_used": False,
            "pass": False,
            "reason": "tm02_completion_button_preconditions_not_met",
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
    diagnostics = env.task_diagnostics()
    checks = {
        "physical_press_operation": bool(actuation.get("pass", False)),
        "physical_button_motion": bool(
            actuation.get("physical_button_motion_observed", False)
        ),
        "completion_event_committed": bool(
            diagnostics["semantic"].get("completion_button_pressed", False)
        ),
        "task_success": bool(diagnostics.get("success", False)),
        "no_absorbing_failure": not bool(
            diagnostics["semantic"].get("absorbing_failure", False)
        ),
    }
    passed = bool(all(checks.values()))
    return {
        "primitive": "physical_tm02_completion_button_press",
        "button": button,
        "preconditions": preconditions,
        "actuation": actuation,
        "task_diagnostics": diagnostics,
        "checks": checks,
        "semantic_shortcut_used": False,
        "pass": passed,
        "reason": None if passed else "tm02_completion_button_press_failed",
    }


def _operate_wr01_door(
    controller: TM02ManipulationController,
    *,
    pose_file: Path,
    opening: bool,
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    if not controller.runtime_config.execute:
        return _dry_run(
            "open_storage_door" if opening else "close_storage_door",
            "lower",
        )
    values = dict(metadata)
    values.pop("control_mode", None)
    values.pop("open_fraction", None)
    max_attempts = max(1, int(values.pop("max_physical_attempts", 1)))
    parameters = {
        "pregrasp_distance": float(values.pop("pregrasp_distance", 0.14)),
        "retreat_distance": float(values.pop("retreat_distance", 0.10)),
        "open_distances": tuple(
            float(value)
            for value in values.pop("open_distances", (0.03, 0.12, 0.21, 0.34))
        ),
        "close_deltas": tuple(
            float(value)
            for value in values.pop("close_deltas", (0.12, 0.27, 0.36, 0.44))
        ),
        "gripper_open_steps": int(values.pop("gripper_open_steps", 7)),
        "gripper_close_steps": int(values.pop("gripper_close_steps", 15)),
        "gripper_release_steps": int(values.pop("gripper_release_steps", 10)),
        "motion_settle_steps": max(
            0, int(values.pop("motion_settle_steps", 2))
        ),
        "final_motion_settle_steps": max(
            0, int(values.pop("final_motion_settle_steps", 5))
        ),
        "retreat_settle_steps": max(
            0, int(values.pop("retreat_settle_steps", 5))
        ),
        "opened_threshold": float(values.pop("opened_threshold", 0.70)),
        "closed_threshold": float(values.pop("closed_threshold", 0.08)),
    }
    attempts: list[dict[str, Any]] = []
    for _ in range(max_attempts):
        result = move_slide_door(
            controller,
            level="lower",
            pose_file=pose_file,
            opening=opening,
            **parameters,
        )
        attempts.append(result)
        if bool(result.get("pass", False)):
            break
        if not bool((result.get("checks") or {}).get("planned_and_executed", False)):
            break
    result = dict(attempts[-1])
    if len(attempts) > 1:
        result["initial_metrics"] = attempts[0].get("initial_metrics")
        result["physical_attempts"] = attempts
    result["physical_attempt_count"] = len(attempts)
    result["max_physical_attempts"] = max_attempts
    if values:
        result["unused_metadata"] = values
    return result


def _required_resource(resources: Mapping[str, Path], name: str) -> Path:
    path = resources.get(name)
    if path is None or not Path(path).is_file():
        raise FileNotFoundError(f"missing TM02 execution resource {name!r}: {path}")
    return Path(path)


def _load_can_place_calibration(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("config_kind") != "tm02_can_place_region_release":
        raise ValueError("invalid TM02 can-place calibration kind")
    anchor = dict(payload.get("anchor") or {})
    placement = dict(payload.get("placement") or {})
    world = np.asarray(anchor.get("world_xyz"), dtype=float).reshape(-1)
    if world.size != 3 or not np.all(np.isfinite(world)):
        raise ValueError("TM02 can-place anchor must contain finite world_xyz")
    if not placement:
        raise ValueError("TM02 can-place calibration requires placement settings")
    return {
        "anchor_world_xyz": world,
        "placement": placement,
    }


def _tm02_place_region_world(env: Any) -> np.ndarray:
    world = np.asarray(env.can_place_region_world_xyz, dtype=float).reshape(-1)
    if world.size != 3 or not np.all(np.isfinite(world)):
        raise ValueError("TM02 live can-place region must be finite xyz")
    return world


def _dry_run(primitive: str, target: str) -> dict[str, Any]:
    return {
        "primitive": str(primitive),
        "object_name": str(target),
        "dry_run": True,
        "checks": {"deferred_to_execute": True},
        "pass": True,
    }


def _runtime_config(
    *,
    execute: bool,
    runtime: Mapping[str, Any],
    planner: Mapping[str, Any],
    static_scene: Mapping[str, Any],
) -> ManipulationRuntimeConfig:
    values = dict(runtime)
    planner_config = MplibPlannerConfig(
        planning_time=float(planner.get("planning_time", 5.0)),
        time_step=float(planner.get("time_step", 0.1)),
        qpos_step=float(planner.get("qpos_step", 0.02)),
        rrt_range=float(planner.get("rrt_range", 0.1)),
    )
    scene = dict(static_scene)
    static_options = {
        "enabled": bool(scene.pop("enabled", False)),
        "validate_collisions": bool(scene.pop("validate_collisions", False)),
    }
    if scene:
        raise ValueError(
            "unknown TM02 static_scene settings: " + ", ".join(sorted(scene))
        )
    config = ManipulationRuntimeConfig(
        execute=bool(execute),
        max_joint_step=float(values.pop("max_joint_step", 0.025)),
        settle_steps=int(values.pop("settle_steps", 20)),
        position_threshold=float(values.pop("position_threshold", 0.12)),
        orientation_threshold_deg=float(
            values.pop("orientation_threshold_deg", 15.0)
        ),
        qpos_threshold=float(values.pop("qpos_threshold", 0.24)),
        open_steps=int(values.pop("open_steps", 20)),
        close_steps=int(values.pop("close_steps", 90)),
        release_steps=int(values.pop("release_steps", 45)),
        release_settle_steps=int(values.pop("release_settle_steps", 20)),
        planner_args=planner_config.planner_args(),
        static_scene_options=static_options,
    )
    if values:
        raise ValueError("unknown TM02 runtime settings: " + ", ".join(sorted(values)))
    return config


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


def _section(payload: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = payload.get(name, {})
    if not isinstance(value, Mapping):
        raise ValueError(f"TM02 {name} must be a mapping")
    return value


__all__ = [
    "ManipulationAssembly",
    "SUPPORTED_OPERATIONS",
    "TM02ManipulationController",
    "build_manipulation_assembly",
]
