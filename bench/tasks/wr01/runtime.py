"""WR-01 task-local RoboCasa runtime composition."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from bench.runtime import TaskRuntimeAdapter, default_close, default_diagnostics, default_success
from bench.tasks.runtime_support import cabinet_initialization, manual_navigation, operations_actions, standard_plan, standard_scene
from bench.tasks.wr01.config import load_task_config, resolve_action_config
from bench.tasks.wr01.domain.episode import resolve_episode_plan, resolve_scene_config
from bench.tasks.wr01.execution.navigation import build_navigation_config
from bench.tasks.wr01.expert.operations import build_operation_handlers
from bench.tasks.wr01.initialization import apply_robot_base_initial_state, apply_robot_torso_initial_state, initialize_episode_state
from bench.tasks.wr01.prompts import resolve_prompt_spec


def _environment(config: Any, scene: Any, plan: Any, resources: Mapping[str, Any]) -> Mapping[str, Any]:
    del scene, resources
    cabinet = dict(config.initial_state.cabinets)
    semantics = dict(config.semantics)
    return {"cabinet_world_xyz": tuple(float(v) for v in cabinet["world_xyz"]),
            "cabinet_yaw_deg": float(cabinet.get("yaw_deg", 0.0)),
            "upper_cabinet_z_offset": float(cabinet["upper_z_offset"]),
            "object_scale": float(config.initial_state.objects.get("object_scale", 0.90)),
            "item_placement_seed": int(plan.seed),
            "final_closed_threshold": float(semantics.get("final_closed_threshold", 0.08)),
            "cabinet_contains_margin": float(semantics.get("cabinet_contains_margin", 0.04))}


_ADAPTER = TaskRuntimeAdapter(
    task_id="WR-01", load_task_config=load_task_config, resolve_task_options=None,
    scene=standard_scene(resolve_scene_config), plan=standard_plan(resolve_episode_plan),
    actions=operations_actions(resolve_action_config), navigation=manual_navigation(build_navigation_config),
    prompt=lambda plan, variant: resolve_prompt_spec(target_item_id=plan.target_item_id, variant=variant),
    environment=_environment,
    initialize=cabinet_initialization(initialize_episode_state, apply_robot_base_initial_state, apply_robot_torso_initial_state),
    diagnostics=default_diagnostics, success=default_success, close=default_close,
    expert_runtime=build_operation_handlers,
)
TASK_RUNTIME_API = _ADAPTER.api()
build_descriptor = _ADAPTER.build_descriptor
initialize_runtime = _ADAPTER.initialize_runtime
check_success = _ADAPTER.check_success
diagnostics = _ADAPTER.diagnostics
close_runtime = _ADAPTER.close_runtime
__all__ = ["TASK_RUNTIME_API", "build_descriptor", "check_success", "close_runtime", "diagnostics", "initialize_runtime"]
