"""TM01 task-local RoboCasa runtime composition."""

from __future__ import annotations

from dataclasses import asdict, replace
from collections.abc import Mapping
from typing import Any

from bench.runtime import TaskRuntimeAdapter, default_close
from bench.tasks.runtime_support import (
    manual_navigation,
    standard_plan,
    standard_scene,
    variant_prompt,
)
from bench.tasks.tm01.config import (
    CALIBRATION_RESOURCE_NAMES,
    load_task_config,
    missing_calibration_resources,
    require_production_calibration,
    resolve_action_config,
    resolve_observer_config,
    resolve_task_options,
)
from bench.tasks.tm01.domain.episode import resolve_episode_plan, resolve_scene_config
from bench.tasks.tm01.evaluation import (
    REPORT_STAGE_PROFILE_ID,
    TASK_EVALUATOR_METADATA_KEY,
    TM01Evaluator,
    diagnostics_for_runtime,
    report_stages_for_runtime,
    success_for_runtime,
)
from bench.tasks.tm01.execution.navigation import build_navigation_config
from bench.tasks.tm01.expert.operations import build_operation_handlers
from bench.tasks.tm01.initialization import apply_manual_path_initial_base_pose
from bench.tasks.tm01.prompts import resolve_prompt_spec


def _actions(config: Any, plan: Any, resources: Mapping[str, Any]) -> Any:
    actions = resolve_action_config(config)
    drawer = dict(actions.drawer_open_metadata)
    resource_name = str(drawer.get("grasp_pose_resource", "")).strip()
    if resource_name:
        drawer["grasp_pose_file"] = str(resources[resource_name])
        actions = replace(actions, drawer_open_metadata=drawer)
    return actions


def _environment(
    config: Any, scene: Any, plan: Any, resources: Mapping[str, Any]
) -> Mapping[str, Any]:
    del resources
    return {
        "drawer_names": scene.drawer_names,
        "drawer_world_xyzs": scene.drawer_world_xyzs,
        "drawer_world_yaw_deg": scene.drawer_world_yaw_deg,
        "drawer_material_by_fixture": dict(plan.drawer_material_by_fixture),
        "completion_button_world_xyzs": dict(scene.completion_button_world_xyzs),
        "expected_stove_asset": scene.expected_stove_asset,
        "knob_id": scene.knob_id,
        "burner_id": scene.burner_id,
        "pot_name": scene.pot_name,
        "pot_xml": scene.pot_xml,
        "observer_config": asdict(resolve_observer_config(config)),
    }


def _initialize_base(descriptor: Any, env: Any, observation: Mapping[str, Any]):
    updated, base = apply_manual_path_initial_base_pose(
        env,
        dict(observation),
        path_file=descriptor.resources["manual_base_path"],
        sample_index=int(
            descriptor.config.initial_state.get("robot_base_sample_index", 0)
        ),
        seed=int(descriptor.episode_plan.seed),
    )
    return updated, {"base": base}, {}, None


def _initialize(descriptor: Any, env: Any, observation: Mapping[str, Any]):
    updated, initialization, metadata, targets = _initialize_base(
        descriptor,
        env,
        observation,
    )
    evaluator = TM01Evaluator()
    evaluator.reset(env.evaluation_snapshot())
    env._tm01_evaluator = evaluator
    return (
        updated,
        initialization,
        {**dict(metadata), TASK_EVALUATOR_METADATA_KEY: evaluator},
        targets,
    )


def _validate(config: Any, episode: Any, execute: bool) -> None:
    if execute and not episode.allow_partial_calibration:
        require_production_calibration(config)


def _release_facts(
    config: Any, episode: Any, resources: Mapping[str, Any], options: Mapping[str, Any]
) -> Mapping[str, Any]:
    del resources, options
    return {
        "controller": {
            "max_joint_step": float(config.manipulation["runtime"]["max_joint_step"])
        },
        "missing_physical_calibration": list(missing_calibration_resources(config)),
        "allow_partial_calibration": bool(episode.allow_partial_calibration),
    }


_ADAPTER = TaskRuntimeAdapter(
    task_id="TM01",
    load_task_config=load_task_config,
    resolve_task_options=resolve_task_options,
    scene=standard_scene(resolve_scene_config, uses_seed=False),
    plan=standard_plan(resolve_episode_plan),
    actions=_actions,
    navigation=manual_navigation(build_navigation_config),
    prompt=variant_prompt(resolve_prompt_spec),
    environment=_environment,
    initialize=_initialize,
    diagnostics=diagnostics_for_runtime,
    success=success_for_runtime,
    close=default_close,
    report_stage_profile_id=REPORT_STAGE_PROFILE_ID,
    report_stages=report_stages_for_runtime,
    validate_descriptor=_validate,
    release_facts=_release_facts,
    expert_runtime=build_operation_handlers,
    task_validated_resource_names=tuple(CALIBRATION_RESOURCE_NAMES),
)

TASK_RUNTIME_API = _ADAPTER.api()
build_descriptor = _ADAPTER.build_descriptor
initialize_runtime = _ADAPTER.initialize_runtime
check_success = _ADAPTER.check_success
diagnostics = _ADAPTER.diagnostics
close_runtime = _ADAPTER.close_runtime

__all__ = [
    "TASK_RUNTIME_API",
    "build_descriptor",
    "check_success",
    "close_runtime",
    "diagnostics",
    "initialize_runtime",
]
