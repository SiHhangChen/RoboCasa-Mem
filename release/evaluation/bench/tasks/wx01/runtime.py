"""WX-01 task-local RoboCasa runtime composition."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from bench.runtime import TaskRuntimeAdapter, default_close
from bench.tasks.runtime_support import manual_station_navigation, standard_scene, variant_prompt
from bench.tasks.wx01.config import load_task_config, resolve_action_config
from bench.tasks.wx01.domain.episode import resolve_episode_plan, resolve_scene_config
from bench.tasks.wx01.evaluation import (
    REPORT_STAGE_PROFILE_ID,
    TASK_EVALUATOR_METADATA_KEY,
    WX01Evaluator,
    diagnostics_for_runtime,
    report_stages_for_runtime,
    success_for_runtime,
)
from bench.tasks.wx01.execution.geometry import resolve_episode_targets
from bench.tasks.wx01.execution.navigation import build_navigation_config
from bench.tasks.wx01.initialization import apply_initial_object_plan, apply_robot_base_initial_state
from bench.tasks.wx01.prompts import resolve_prompt_spec


def _plan(config: Any, resolved: Any, options: Mapping[str, Any]) -> Any:
    if options:
        raise ValueError("WX-01 task options are YAML-owned")
    return resolve_episode_plan(
        seed=resolved.seed,
        episode=config.episode,
        object_settings={name: config.initial_state.objects[name] for name in ("blocking_object", "cleaning_tool")},
    )


def _environment(config: Any, scene: Any, plan: Any, resources: Mapping[str, Any]) -> Mapping[str, Any]:
    del resources
    semantics = dict(config.semantics)
    return {"cup_material_id": str(plan.cup_material_id),
            "sponge_type": str(plan.sponge_type),
            "origin_threshold": float(semantics.get("origin_threshold", 0.015)),
            "wipe_radius": float(semantics.get("wipe_radius", 0.20)),
            "contact_separation": float(semantics.get("contact_separation", 0.02)),
            "minimum_contact_count": int(semantics.get("minimum_contact_count", 2)),
            "minimum_wipe_sweep": float(semantics.get("minimum_wipe_sweep", 0.07)),
            "release_distance": float(semantics.get("release_distance", 0.15)),
            "blocking_object_scale": scene.blocking_object_scale,
            "blocking_object_placement_size": scene.blocking_object_placement_size,
            "blocking_object_placement_pos": scene.blocking_object_placement_pos,
            "blocking_object_rotation": scene.blocking_object_rotation,
            "stain_color_rgba": tuple(semantics.get("stain_color_rgba", (1.0, 1.0, 1.0, 1.0))),
            "stain_circle_radius": float(semantics.get("stain_circle_radius", 0.045)),
            "stain_z_offset": float(semantics.get("stain_z_offset", 0.0018))}


def _initialize(descriptor: Any, env: Any, observation: Mapping[str, Any]):
    configure_episode = getattr(env, "configure_episode", None)
    if callable(configure_episode):
        configure_episode(descriptor.episode_plan)
    updated, objects = apply_initial_object_plan(
        env, episode=descriptor.episode_plan,
        enabled=bool(descriptor.config.initial_state.objects.get("enabled", True)),
    )
    updated, base = apply_robot_base_initial_state(
        env, updated, config=descriptor.config.initial_state.robot_base,
        navigation=descriptor.navigation, seed=descriptor.episode_plan.seed,
    )
    targets = resolve_episode_targets(env, descriptor.episode_plan)
    from bench.runtime.setup_replay import replay_setup
    updated, short_start = replay_setup(env, descriptor)
    evaluator = WX01Evaluator()
    evaluator.reset(env.evaluation_snapshot())
    env._wx01_evaluator = evaluator
    return (
        updated,
        {"objects": objects, "base": base, "short_start": short_start},
        {TASK_EVALUATOR_METADATA_KEY: evaluator},
        targets,
    )


_ADAPTER = TaskRuntimeAdapter(
    task_id="WX-01", load_task_config=load_task_config, resolve_task_options=None,
    scene=standard_scene(resolve_scene_config), plan=_plan,
    actions=lambda config, plan, resources: resolve_action_config(episode=plan, operations=config.operations),
    navigation=manual_station_navigation(build_navigation_config),
    prompt=variant_prompt(resolve_prompt_spec), environment=_environment, initialize=_initialize,
    diagnostics=diagnostics_for_runtime, success=success_for_runtime, close=default_close,
    report_stage_profile_id=REPORT_STAGE_PROFILE_ID,
    report_stages=report_stages_for_runtime,
    release_facts=lambda config, episode, resources, options: {
        "cup_material_id": str(
            getattr(episode, "cup_material_id", None)
            or (getattr(episode, "facts", {}) or {}).get("cup_material_id", "")
        ),
        "sponge_type": str(
            getattr(episode, "sponge_type", None)
            or (getattr(episode, "facts", {}) or {}).get("sponge_type", "")
        ),
    },
    expert_runtime=None,
)
TASK_RUNTIME_API = _ADAPTER.api()
build_descriptor = _ADAPTER.build_descriptor
initialize_runtime = _ADAPTER.initialize_runtime
check_success = _ADAPTER.check_success
diagnostics = _ADAPTER.diagnostics
close_runtime = _ADAPTER.close_runtime
__all__ = ["TASK_RUNTIME_API", "build_descriptor", "check_success", "close_runtime", "diagnostics", "initialize_runtime"]
