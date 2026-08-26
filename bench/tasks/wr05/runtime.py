"""WR-05 task-local RoboCasa runtime composition."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from bench.runtime import TaskRuntimeAdapter, default_close
from bench.tasks.runtime_support import keyword_config_actions, shelf_initialization, standard_plan, standard_scene, station_navigation, variant_prompt
from bench.tasks.wr05.config import load_task_config, resolve_action_config
from bench.tasks.wr05.domain.episode import resolve_episode_plan, resolve_scene_config
from bench.tasks.wr05.domain.contracts import TASK_REVISION
from bench.tasks.wr05.evaluation import (
    REPORT_STAGE_PROFILE_ID,
    TASK_EVALUATOR_METADATA_KEY,
    WR05Evaluator,
    diagnostics_for_runtime,
    report_stages_for_runtime,
    success_for_runtime,
)
from bench.tasks.wr05.execution.geometry import resolve_episode_targets
from bench.tasks.wr05.execution.navigation import build_navigation_config
from bench.tasks.wr05.expert.operations import build_operation_handlers
from bench.tasks.wr05.initialization import initialize_episode_state
from bench.tasks.wr05.prompts import resolve_prompt_spec


def _environment(config: Any, scene: Any, plan: Any, resources: Mapping[str, Any]) -> Mapping[str, Any]:
    del scene, resources
    semantics = dict(config.semantics)
    return {"home_assignment": dict(plan.home_assignment),
            "home_region_placements": {k: dict(v) for k, v in config.initial_state.home_regions.items()},
            "use_assignment": dict(plan.use_assignment), "episode_case_id": plan.case_id,
            "home_threshold": float(semantics.get("home_threshold", 0.16)),
            "plate_stage_threshold": float(semantics.get("plate_stage_threshold", 0.20)),
            "return_region_xy_margin": float(semantics.get("return_region_xy_margin", 0.04)),
            "return_region_height_tolerance": float(semantics.get("return_region_height_tolerance", 0.10)),
            "return_level_min_separation": float(semantics.get("return_level_min_separation", 0.20))}


_initialize_shelf = shelf_initialization(
    initialize_episode_state,
    resolve_episode_targets,
)


def _initialize(descriptor: Any, env: Any, observation: Mapping[str, Any]):
    updated, initialization, metadata, targets = _initialize_shelf(
        descriptor,
        env,
        observation,
    )
    evaluator = WR05Evaluator()
    evaluator.reset(env.evaluation_snapshot())
    env._wr05_evaluator = evaluator
    return (
        updated,
        initialization,
        {**dict(metadata), TASK_EVALUATOR_METADATA_KEY: evaluator},
        targets,
    )


_ADAPTER = TaskRuntimeAdapter(
    task_id="WR-05", load_task_config=load_task_config, resolve_task_options=None,
    scene=standard_scene(resolve_scene_config), plan=standard_plan(resolve_episode_plan),
    actions=keyword_config_actions(resolve_action_config), navigation=station_navigation(build_navigation_config),
    prompt=variant_prompt(resolve_prompt_spec), environment=_environment,
    initialize=_initialize,
    diagnostics=diagnostics_for_runtime, success=success_for_runtime, close=default_close,
    report_stage_profile_id=REPORT_STAGE_PROFILE_ID,
    report_stages=report_stages_for_runtime,
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
