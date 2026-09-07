"""WX-02 task-local RoboCasa runtime composition."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from bench.runtime import TaskRuntimeAdapter, default_close
from bench.tasks.runtime_support import manual_station_navigation, standard_scene, variant_prompt
from bench.tasks.wx02.config import load_task_config, resolve_action_config
from bench.tasks.wx02.domain.contracts import CUP_NAMES, OBJECT_NAMES
from bench.tasks.wx02.domain.episode import resolve_episode_plan, resolve_scene_config
from bench.tasks.wx02.evaluation import (
    REPORT_STAGE_PROFILE_ID,
    TASK_EVALUATOR_METADATA_KEY,
    WX02Evaluator,
    diagnostics_for_runtime,
    report_stages_for_runtime,
    success_for_runtime,
)
from bench.tasks.wx02.execution.geometry import resolve_episode_targets
from bench.tasks.wx02.execution.navigation import build_navigation_config
from bench.tasks.wx02.expert.operations import build_operation_handlers
from bench.tasks.wx02.initialization import (
    apply_initial_object_plan,
    apply_robot_base_initial_state,
    initialize_cup_front_start_state,
)
from bench.tasks.wx02.prompts import resolve_prompt_spec


def _plan(config: Any, resolved: Any, options: Mapping[str, Any]) -> Any:
    if options:
        raise ValueError("WX-02 task options are YAML-owned")
    return resolve_episode_plan(
        seed=resolved.seed,
        episode=config.episode,
        object_settings={name: config.initial_state.objects[name] for name in OBJECT_NAMES},
    )


def _environment(config: Any, scene: Any, plan: Any, resources: Mapping[str, Any]) -> Mapping[str, Any]:
    del resources
    semantics = dict(config.semantics)
    return {"origin_threshold": float(semantics.get("origin_threshold", 0.02)),
            "tool_return_region_tolerance": float(
                semantics.get("tool_return_region_tolerance", 0.02)
            ),
            "wipe_radius": float(semantics.get("wipe_radius", 0.20)),
            "contact_separation": float(semantics.get("contact_separation", 0.02)),
            "minimum_contact_count": int(semantics.get("minimum_contact_count", 2)),
            "minimum_wipe_sweep": float(semantics.get("minimum_wipe_sweep", 0.07)),
            "release_distance": float(semantics.get("release_distance", 0.15)),
            "moved_threshold": float(semantics.get("moved_threshold", 0.10)),
            "stain_surface_z_offset": float(
                semantics.get("stain_surface_z_offset", 0.0015)
            ),
            "stain_color_rgba": semantics.get(
                "stain_color_rgba", (0.20, 0.055, 0.01, 0.92)
            ),
            "stain_spots": semantics.get("stain_spots", ()),
            "stain_center_xy_by_object": {
                name: plan.initial_object_poses[name].xy for name in CUP_NAMES
            },
            "wipe_direction_xy_by_object": {
                name: plan.wipe_metadata_by_object[name]["direction_xy"]
                for name in CUP_NAMES
            },
            "cup_scale": scene.cup_scale,
            "cup_placement_size": scene.cup_placement_size,
            "primary_cup_placement_pos": scene.primary_cup_placement_pos,
            "secondary_cup_placement_pos": scene.secondary_cup_placement_pos,
            "cup_rotation": scene.cup_rotation,
            "cleaning_tool_return_region_center_xy": scene.cleaning_tool_return_region_center_xy,
            "cleaning_tool_return_region_size_xy": scene.cleaning_tool_return_region_size_xy,
            "cleaning_tool_return_region_rgba": scene.cleaning_tool_return_region_rgba}


def _initialize(descriptor: Any, env: Any, observation: Mapping[str, Any]):
    updated, objects = apply_initial_object_plan(
        env, episode=descriptor.episode_plan,
        enabled=bool(descriptor.config.initial_state.objects.get("enabled", True)),
    )
    updated, base = apply_robot_base_initial_state(
        env, updated, config=descriptor.config.initial_state.robot_base,
        navigation=descriptor.navigation, seed=descriptor.episode_plan.seed,
    )
    targets = resolve_episode_targets(env, descriptor.episode_plan)
    updated, start = initialize_cup_front_start_state(
        env,
        targets=targets,
        config=descriptor.config.initial_state,
    )
    evaluator = WX02Evaluator()
    evaluator.reset(env.evaluation_snapshot())
    env._wx02_evaluator = evaluator
    return (
        updated,
        {"objects": objects, "base": base, "start": start},
        {TASK_EVALUATOR_METADATA_KEY: evaluator},
        targets,
    )


_ADAPTER = TaskRuntimeAdapter(
    task_id="WX-02", load_task_config=load_task_config, resolve_task_options=None,
    scene=standard_scene(resolve_scene_config), plan=_plan,
    actions=lambda config, plan, resources: resolve_action_config(episode=plan, operations=config.operations),
    navigation=manual_station_navigation(build_navigation_config),
    prompt=variant_prompt(resolve_prompt_spec), environment=_environment, initialize=_initialize,
    diagnostics=diagnostics_for_runtime, success=success_for_runtime, close=default_close,
    report_stage_profile_id=REPORT_STAGE_PROFILE_ID,
    report_stages=report_stages_for_runtime,
    expert_runtime=build_operation_handlers,
)
TASK_RUNTIME_API = _ADAPTER.api()
build_descriptor = _ADAPTER.build_descriptor
initialize_runtime = _ADAPTER.initialize_runtime
check_success = _ADAPTER.check_success
diagnostics = _ADAPTER.diagnostics
close_runtime = _ADAPTER.close_runtime
__all__ = ["TASK_RUNTIME_API", "build_descriptor", "check_success", "close_runtime", "diagnostics", "initialize_runtime"]
