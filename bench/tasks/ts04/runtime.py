"""TS04 task-local RoboCasa runtime composition."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from typing import Any

from bench.runtime import TaskRuntimeAdapter, default_close
from bench.tasks.runtime_support import (
    drawer_initialization,
    standard_plan,
    standard_scene,
    station_navigation,
    variant_prompt,
)
from bench.tasks.ts04.config import (
    load_task_config,
    resolve_action_config,
    resolve_task_options,
)
from bench.tasks.ts04.domain.contracts import TASK_REVISION
from bench.tasks.ts04.domain.episode import resolve_episode_plan, resolve_scene_config
from bench.tasks.ts04.evaluation import (
    REPORT_STAGE_PROFILE_ID,
    TASK_EVALUATOR_METADATA_KEY,
    TS04Evaluator,
    diagnostics_for_runtime,
    report_stages_for_runtime,
    success_for_runtime,
)
from bench.tasks.ts04.execution.navigation import build_navigation_config
from bench.tasks.ts04.expert.operations import build_operation_handlers
from bench.tasks.ts04.initialization import (
    apply_initial_drawer_state,
    apply_robot_base_initial_state,
)
from bench.tasks.ts04.prompts import resolve_prompt_spec


def _actions(config: Any, plan: Any, resources: Mapping[str, Any]) -> Any:
    del plan
    actions = resolve_action_config(operations=config.operations)
    opened = dict(actions.drawer_open_metadata)
    resource_name = str(opened.get("grasp_pose_resource", "")).strip()
    if resource_name:
        opened["grasp_pose_file"] = str(resources[resource_name])
        actions = replace(actions, drawer_open_metadata=opened)
    return actions


def _environment(
    config: Any,
    scene: Any,
    plan: Any,
    resources: Mapping[str, Any],
) -> Mapping[str, Any]:
    del resources
    button = dict(config.operations.get("completion_button", {}))
    return {
        "fixture_names": scene.fixture_names,
        "drawer_world_xyzs": scene.drawer_world_xyzs,
        "drawer_world_yaw_deg": scene.drawer_world_yaw_deg,
        "completion_button_world_xyz": scene.completion_button_world_xyz,
        "target_fixture": plan.target_fixture,
        "open_threshold": scene.open_threshold,
        "closed_threshold": scene.closed_threshold,
        "button_pressed_threshold_m": float(
            button.get("press_depth_threshold", 0.004)
        ),
        "button_released_threshold_m": float(
            button.get("release_depth_threshold", 0.001)
        ),
        "button_release_dwell_steps": int(button.get("release_dwell_steps", 2)),
    }


_initialize_drawers = drawer_initialization(
    apply_initial_drawer_state,
    apply_robot_base_initial_state,
)


def _initialize(descriptor: Any, env: Any, observation: Mapping[str, Any]):
    updated, initialization, metadata, targets = _initialize_drawers(
        descriptor,
        env,
        observation,
    )
    evaluator = TS04Evaluator(
        fixture_names=env.fixture_names,
        target_fixture=env.target_fixture,
    )
    evaluator.reset(env.evaluation_snapshot())
    env._ts04_evaluator = evaluator
    return (
        updated,
        initialization,
        {**dict(metadata or {}), TASK_EVALUATOR_METADATA_KEY: evaluator},
        targets,
    )


_ADAPTER = TaskRuntimeAdapter(
    task_id="TS04",
    load_task_config=load_task_config,
    resolve_task_options=resolve_task_options,
    scene=standard_scene(resolve_scene_config),
    plan=standard_plan(resolve_episode_plan),
    actions=_actions,
    navigation=station_navigation(build_navigation_config),
    prompt=variant_prompt(resolve_prompt_spec),
    environment=_environment,
    initialize=_initialize,
    diagnostics=diagnostics_for_runtime,
    success=success_for_runtime,
    close=default_close,
    report_stage_profile_id=REPORT_STAGE_PROFILE_ID,
    report_stages=report_stages_for_runtime,
    release_facts=lambda config, episode, resources, options: {
        "task_revision": TASK_REVISION,
        "task_id": episode.task_id,
        "target_fixture": episode.facts["target_fixture"],
        "order_profile_id": (
            "left_right"
            if episode.facts.get("demo_fixture_order", ("",))[0]
            == "cabinet_slot_0"
            else "right_left"
        ),
    },
    expert_runtime=build_operation_handlers,
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
