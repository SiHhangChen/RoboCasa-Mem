"""TS01 task-local RoboCasa runtime composition."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from bench.runtime import TaskRuntimeAdapter, default_close
from bench.tasks.runtime_support import episode_actions, manual_navigation, standard_plan, standard_scene, sugar_initialization, sugar_prompt
from bench.tasks.ts01.config import load_task_config, resolve_action_config, resolve_task_options
from bench.tasks.ts01.domain.episode import resolve_episode_plan, resolve_scene_config
from bench.tasks.ts01.evaluation import (
    REPORT_STAGE_PROFILE_ID,
    TASK_EVALUATOR_METADATA_KEY,
    TS01Evaluator,
    diagnostics_for_runtime,
    report_stages_for_runtime,
    success_for_runtime,
)
from bench.tasks.ts01.execution.navigation import build_navigation_config
from bench.tasks.ts01.expert.operations import build_operation_handlers
from bench.tasks.ts01.initialization import apply_fixed_initial_object_regions, apply_robot_base_initial_state
from bench.tasks.ts01.prompts import resolve_prompt_spec


CONFIRMED_SUGAR_DWELL_STEPS = 5


def _environment(config: Any, scene: Any, plan: Any, resources: Mapping[str, Any]) -> Mapping[str, Any]:
    operations = getattr(config, "operations", {})
    completion_button = (
        operations.get("completion_button", {})
        if isinstance(operations, Mapping)
        else {}
    )
    return {
        "target_sugar_count": plan.target_sugar_count,
        "total_sugar_count": plan.total_sugar_count,
        "sugar_names": plan.sugar_names,
        "target_sugar_names": plan.target_sugar_names,
        "sugar_plate_name": scene.sugar_plate_name,
        "receptacle_name": scene.receptacle_name,
        "completion_button_position_file": str(resources["completion_button_position"]),
        "confirmed_sugar_dwell_steps": CONFIRMED_SUGAR_DWELL_STEPS,
        "minimum_press_displacement_m": float(
            completion_button.get("minimum_press_displacement_m", 0.004)
        ),
    }


_initialize_sugar = sugar_initialization(
    apply_fixed_initial_object_regions,
    apply_robot_base_initial_state,
)


def _initialize(descriptor: Any, env: Any, observation: Mapping[str, Any]):
    updated, initialization, metadata, targets = _initialize_sugar(
        descriptor,
        env,
        observation,
    )
    evaluator = TS01Evaluator(
        target_sugar_count=env.target_sugar_count,
        total_sugar_count=env.total_sugar_count,
    )
    evaluator.reset(env.evaluation_snapshot())
    env._ts01_evaluator = evaluator
    return (
        updated,
        initialization,
        {**dict(metadata or {}), TASK_EVALUATOR_METADATA_KEY: evaluator},
        targets,
    )


_ADAPTER = TaskRuntimeAdapter(
    task_id="TS01", load_task_config=load_task_config, resolve_task_options=resolve_task_options,
    scene=standard_scene(resolve_scene_config), plan=standard_plan(resolve_episode_plan),
    actions=episode_actions(resolve_action_config), navigation=manual_navigation(build_navigation_config),
    prompt=sugar_prompt(resolve_prompt_spec), environment=_environment,
    initialize=_initialize,
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
