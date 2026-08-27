"""TM02 task-local RoboCasa runtime composition."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict
from typing import Any

from bench.runtime import TaskRuntimeAdapter, default_close
from bench.tasks.runtime_support import config_actions, manual_navigation, path_base_initialization, standard_plan, standard_scene, variant_prompt
from bench.tasks.tm02.config import CALIBRATION_RESOURCE_NAMES, load_task_config, missing_calibration_resources, require_production_calibration, resolve_action_config, resolve_observer_config, resolve_task_options
from bench.tasks.tm02.domain.episode import resolve_episode_plan, resolve_scene_config
from bench.tasks.tm02.evaluation import (
    REPORT_STAGE_PROFILE_ID,
    TASK_EVALUATOR_METADATA_KEY,
    TM02Evaluator,
    diagnostics_for_runtime,
    report_stages_for_runtime,
    success_for_runtime,
)
from bench.tasks.tm02.execution.navigation import build_navigation_config
from bench.tasks.tm02.expert.operations import build_operation_handlers
from bench.tasks.tm02.initialization import apply_manual_path_initial_base_pose
from bench.tasks.tm02.prompts import resolve_prompt_spec


def _environment(config: Any, scene: Any, plan: Any, resources: Mapping[str, Any]) -> Mapping[str, Any]:
    del plan, resources
    gripper = dict(config.manipulation.get("gripper", {}))
    return {
        "water_button_world_xyz": scene.water_button_world_xyz,
        "completion_button_world_xyz": scene.completion_button_world_xyz,
        "cabinet_world_xyz": scene.cabinet_world_xyz,
        "cabinet_yaw_deg": scene.cabinet_yaw_deg,
        "can_place_region_world_xyz": scene.can_place_region_world_xyz,
        "can_place_region_size_xy": scene.can_place_region_size_xy,
        "pot_name": scene.pot_name,
        "pot_xml": scene.pot_xml,
        "can_name": scene.can_name,
        "can_xml": scene.can_xml,
        "can_scale": scene.can_scale,
        "gripper_force_limit": float(gripper.get("force_limit", 20.0)),
        "gripper_pad_friction": tuple(
            float(value)
            for value in gripper.get("pad_friction", (2.0, 0.05, 0.0001))
        ),
        "observer_config": asdict(resolve_observer_config(config)),
    }


def _validate(config: Any, episode: Any, execute: bool) -> None:
    if execute and not episode.allow_partial_calibration:
        require_production_calibration(config)


def _release_facts(config: Any, episode: Any, resources: Mapping[str, Any], options: Mapping[str, Any]) -> Mapping[str, Any]:
    del resources, options
    return {
        "controller": {"max_joint_step": float(config.manipulation["runtime"]["max_joint_step"])},
        "missing_physical_calibration": list(missing_calibration_resources(config)),
        "allow_partial_calibration": bool(episode.allow_partial_calibration),
    }


_initialize_base = path_base_initialization(apply_manual_path_initial_base_pose)


def _initialize(descriptor: Any, env: Any, observation: Mapping[str, Any]):
    """Configure physical contact, initialize the base, then arm evaluation."""

    env.configure_canonical_physics()
    updated, initialization, metadata, targets = _initialize_base(
        descriptor,
        env,
        observation,
    )
    evaluator = TM02Evaluator()
    evaluator.reset(env.evaluation_snapshot())
    env._tm02_evaluator = evaluator
    return (
        updated,
        initialization,
        {**dict(metadata), TASK_EVALUATOR_METADATA_KEY: evaluator},
        targets,
    )


_ADAPTER = TaskRuntimeAdapter(
    task_id="TM02", load_task_config=load_task_config, resolve_task_options=resolve_task_options,
    scene=standard_scene(resolve_scene_config, uses_seed=False), plan=standard_plan(resolve_episode_plan),
    actions=config_actions(resolve_action_config), navigation=manual_navigation(build_navigation_config),
    prompt=variant_prompt(resolve_prompt_spec), environment=_environment,
    initialize=_initialize,
    diagnostics=diagnostics_for_runtime, success=success_for_runtime, close=default_close,
    report_stage_profile_id=REPORT_STAGE_PROFILE_ID,
    report_stages=report_stages_for_runtime,
    validate_descriptor=_validate, release_facts=_release_facts,
    expert_runtime=build_operation_handlers,
    task_validated_resource_names=tuple(CALIBRATION_RESOURCE_NAMES),
)
TASK_RUNTIME_API = _ADAPTER.api()
build_descriptor = _ADAPTER.build_descriptor
initialize_runtime = _ADAPTER.initialize_runtime
check_success = _ADAPTER.check_success
diagnostics = _ADAPTER.diagnostics
close_runtime = _ADAPTER.close_runtime
__all__ = ["TASK_RUNTIME_API", "build_descriptor", "check_success", "close_runtime", "diagnostics", "initialize_runtime"]
