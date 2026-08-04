"""WA-01 task-local RoboCasa runtime composition."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from bench.runtime import TaskRuntimeAdapter, default_close, default_diagnostics, default_success
from bench.tasks.runtime_support import manual_station_navigation, standard_plan, standard_scene
from bench.tasks.wa01.config import load_pose_defaults, load_task_config, resolve_action_config
from bench.tasks.wa01.domain.episode import resolve_episode_plan, resolve_scene_config
from bench.tasks.wa01.domain.geometry import dining_region_config_summary
from bench.tasks.wa01.execution.navigation import build_navigation_config
from bench.tasks.wa01.expert.operations import build_operation_handlers
from bench.tasks.wa01.initialization import apply_fixed_initial_object_regions, apply_robot_base_initial_state
from bench.tasks.wa01.prompts import resolve_prompt_spec


def _actions(config: Any, plan: Any, resources: Mapping[str, Any]) -> Any:
    del resources
    return resolve_action_config(
        episode=plan,
        pose_defaults=load_pose_defaults(config),
        operations=config.operations,
    )


def _environment(config: Any, scene: Any, plan: Any, resources: Mapping[str, Any]) -> Mapping[str, Any]:
    del config, scene, plan, resources
    return {}


def _initialize(descriptor: Any, env: Any, observation: Mapping[str, Any]):
    objects, base = {}, {}
    objects_obs, objects = apply_fixed_initial_object_regions(
        env,
        seed=descriptor.episode_plan.seed,
        target_sugared_coffee_id=descriptor.episode_plan.target_sugared_coffee_id,
        **dict(descriptor.config.initial_state.objects),
    )
    env.set_target_sugared_coffee_id(descriptor.episode_plan.target_sugared_coffee_id)
    env.configure_dining_placement_regions(
        region_config=dining_region_config_summary(descriptor.actions.dining_region_config),
        regions_by_slot=descriptor.actions.dining_regions_by_slot,
        sampled_targets_by_slot=descriptor.actions.sampled_dining_targets_by_slot,
        planned_slot_assignment=descriptor.episode_plan.final_slot_assignment,
    )
    updated = env._get_observations(force_update=True)
    updated, base = apply_robot_base_initial_state(
        env, updated,
        config=descriptor.config.initial_state.robot_base,
        nav_config=descriptor.navigation,
        seed=descriptor.scene.scene_seed,
    )
    return updated, {"objects": objects, "base": base}, {}, None


_ADAPTER = TaskRuntimeAdapter(
    task_id="WA-01", load_task_config=load_task_config, resolve_task_options=None,
    scene=standard_scene(resolve_scene_config), plan=standard_plan(resolve_episode_plan),
    actions=_actions, navigation=manual_station_navigation(build_navigation_config),
    prompt=lambda plan, variant: resolve_prompt_spec(variant=variant), environment=_environment,
    initialize=_initialize, diagnostics=default_diagnostics, success=default_success, close=default_close,
    expert_runtime=build_operation_handlers,
)
TASK_RUNTIME_API = _ADAPTER.api()
build_descriptor = _ADAPTER.build_descriptor
initialize_runtime = _ADAPTER.initialize_runtime
check_success = _ADAPTER.check_success
diagnostics = _ADAPTER.diagnostics
close_runtime = _ADAPTER.close_runtime
__all__ = ["TASK_RUNTIME_API", "build_descriptor", "check_success", "close_runtime", "diagnostics", "initialize_runtime"]
