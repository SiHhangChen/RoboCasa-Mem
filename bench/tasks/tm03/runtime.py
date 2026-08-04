"""TM03 task-local RoboCasa runtime composition."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, replace
from typing import Any

from bench.runtime import TaskRuntimeAdapter, default_close, default_diagnostics, default_success
from bench.tasks.runtime_support import manual_navigation, path_base_initialization, standard_plan, standard_scene, variant_prompt
from bench.tasks.tm03.config import CALIBRATION_RESOURCE_NAMES, load_task_config, missing_calibration_resources, require_production_calibration, resolve_action_config, resolve_observer_config, resolve_task_options
from bench.tasks.tm03.domain.episode import resolve_episode_plan, resolve_scene_config
from bench.tasks.tm03.execution.navigation import build_navigation_config
from bench.tasks.tm03.expert.operations import build_operation_handlers
from bench.tasks.tm03.initialization import apply_manual_path_initial_base_pose
from bench.tasks.tm03.prompts import resolve_prompt_spec


def _actions(config: Any, plan: Any, resources: Mapping[str, Any]) -> Any:
    actions = resolve_action_config(config)
    drawer = dict(actions.drawer_open_metadata)
    name = str(drawer.get("grasp_pose_resource", "")).strip()
    if name:
        drawer["grasp_pose_file"] = str(resources[name])
        actions = replace(actions, drawer_open_metadata=drawer)
    return actions


def _environment(config: Any, scene: Any, plan: Any, resources: Mapping[str, Any]) -> Mapping[str, Any]:
    del plan, resources
    return {
        "drawer_names": scene.drawer_names, "drawer_world_xyzs": scene.drawer_world_xyzs,
        "drawer_world_yaw_deg": scene.drawer_world_yaw_deg,
        "completion_button_world_xyzs": dict(scene.completion_button_world_xyzs),
        "expected_stove_asset": scene.expected_stove_asset, "knob_id": scene.knob_id,
        "burner_id": scene.burner_id, "pot_name": scene.pot_name, "pot_xml": scene.pot_xml,
        "expected_toaster_oven_asset": scene.expected_toaster_oven_asset,
        "toaster_oven_food_name": scene.toaster_oven_food_name,
        "toaster_oven_food_xml": scene.toaster_oven_food_xml,
        "toaster_oven_time_joint": scene.toaster_oven_time_joint,
        "toaster_oven_time_collision_geom": scene.toaster_oven_time_collision_geom,
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


_ADAPTER = TaskRuntimeAdapter(
    task_id="TM03", load_task_config=load_task_config, resolve_task_options=resolve_task_options,
    scene=standard_scene(resolve_scene_config, uses_seed=False), plan=standard_plan(resolve_episode_plan),
    actions=_actions, navigation=manual_navigation(build_navigation_config), prompt=variant_prompt(resolve_prompt_spec),
    environment=_environment, initialize=path_base_initialization(apply_manual_path_initial_base_pose),
    diagnostics=default_diagnostics, success=default_success, close=default_close,
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
