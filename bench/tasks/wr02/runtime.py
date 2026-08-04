"""WR-02 task-local RoboCasa runtime composition."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from bench.runtime import TaskRuntimeAdapter, default_close, default_diagnostics, default_success
from bench.tasks.runtime_support import keyword_config_actions, shelf_initialization, standard_plan, standard_scene, station_navigation, variant_prompt
from bench.tasks.wr02.config import load_task_config, resolve_action_config
from bench.tasks.wr02.domain.episode import resolve_episode_plan, resolve_scene_config
from bench.tasks.wr02.execution.geometry import resolve_episode_targets
from bench.tasks.wr02.execution.navigation import build_navigation_config
from bench.tasks.wr02.expert.operations import build_operation_handlers
from bench.tasks.wr02.initialization import initialize_episode_state
from bench.tasks.wr02.prompts import resolve_prompt_spec


def _environment(config: Any, scene: Any, plan: Any, resources: Mapping[str, Any]) -> Mapping[str, Any]:
    del scene, resources
    semantics = dict(config.semantics)
    return {"home_assignment": dict(plan.home_assignment),
            "home_region_placements": {k: dict(v) for k, v in config.initial_state.home_regions.items()},
            "use_assignment": dict(plan.use_assignment), "episode_case_id": plan.case_id,
            "home_threshold": float(semantics.get("home_threshold", 0.16)),
            "sink_stage_threshold": float(semantics.get("sink_stage_threshold", 0.20)),
            "return_region_xy_margin": float(semantics.get("return_region_xy_margin", 0.04)),
            "return_region_height_tolerance": float(semantics.get("return_region_height_tolerance", 0.10)),
            "return_level_min_separation": float(semantics.get("return_level_min_separation", 0.20))}


_ADAPTER = TaskRuntimeAdapter(
    task_id="WR-02", load_task_config=load_task_config, resolve_task_options=None,
    scene=standard_scene(resolve_scene_config), plan=standard_plan(resolve_episode_plan),
    actions=keyword_config_actions(resolve_action_config), navigation=station_navigation(build_navigation_config),
    prompt=variant_prompt(resolve_prompt_spec), environment=_environment,
    initialize=shelf_initialization(initialize_episode_state, resolve_episode_targets),
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
