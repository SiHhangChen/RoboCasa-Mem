"""WR-06 task-local RoboCasa runtime composition."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from bench.runtime import TaskRuntimeAdapter, default_close, default_diagnostics, default_success
from bench.tasks.runtime_support import keyword_config_actions, shelf_initialization, standard_plan, standard_scene, station_navigation, variant_prompt
from bench.tasks.wr06.config import load_task_config, resolve_action_config, resolve_task_options
from bench.tasks.wr06.domain.episode import resolve_episode_plan, resolve_scene_config
from bench.tasks.wr06.domain.contracts import TASK_REVISION
from bench.tasks.wr06.execution.geometry import resolve_episode_targets
from bench.tasks.wr06.execution.navigation import build_navigation_config
from bench.tasks.wr06.expert.operations import build_operation_handlers
from bench.tasks.wr06.initialization import initialize_episode_state
from bench.tasks.wr06.prompts import resolve_prompt_spec


def _environment(config: Any, scene: Any, plan: Any, resources: Mapping[str, Any]) -> Mapping[str, Any]:
    del scene, resources
    semantics = dict(config.semantics)
    return {"original_level": plan.demo_original_level, "plate_stage_region": plan.plate_stage_region,
            "placement_pair_seed": plan.placement_pair_seed,
            "home_region_placements": {k: dict(v) for k, v in config.initial_state.home_regions.items()},
            "plate_stage_threshold": float(semantics.get("plate_stage_threshold", 0.20)),
            "return_region_xy_margin": float(semantics.get("return_region_xy_margin", 0.04)),
            "return_region_height_tolerance": float(semantics.get("return_region_height_tolerance", 0.10)),
            "physical_event_dwell_steps": int(semantics.get("physical_event_dwell_steps", 3)),
            "terminal_verification_steps": int(semantics.get("terminal_verification_steps", 100))}


_ADAPTER = TaskRuntimeAdapter(
    task_id="WR-06", load_task_config=load_task_config, resolve_task_options=resolve_task_options,
    scene=standard_scene(resolve_scene_config), plan=standard_plan(resolve_episode_plan),
    actions=keyword_config_actions(resolve_action_config), navigation=station_navigation(build_navigation_config),
    prompt=variant_prompt(resolve_prompt_spec), environment=_environment,
    initialize=shelf_initialization(initialize_episode_state, resolve_episode_targets),
    diagnostics=default_diagnostics, success=default_success, close=default_close,
    release_facts=lambda config, episode, resources, options: {
        "task_revision": TASK_REVISION,
        "task_id": episode.task_id,
    },
    expert_runtime=build_operation_handlers,
)
TASK_RUNTIME_API = _ADAPTER.api()
build_descriptor = _ADAPTER.build_descriptor
initialize_runtime = _ADAPTER.initialize_runtime
check_success = _ADAPTER.check_success
diagnostics = _ADAPTER.diagnostics
close_runtime = _ADAPTER.close_runtime
__all__ = ["TASK_RUNTIME_API", "build_descriptor", "check_success", "close_runtime", "diagnostics", "initialize_runtime"]
