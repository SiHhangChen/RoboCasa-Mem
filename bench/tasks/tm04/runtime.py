"""TM04 task-local runtime composition scaffold."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict
from typing import Any

from bench.runtime import TaskRuntimeAdapter, default_close, default_diagnostics, default_success
from bench.tasks.runtime_support import (
    config_actions,
    manual_navigation,
    standard_plan,
    standard_scene,
    variant_prompt,
)

from .config import (
    CALIBRATION_RESOURCE_NAMES,
    load_task_config,
    missing_calibration_resources,
    require_production_calibration,
    resolve_action_config,
    resolve_observer_config,
    resolve_task_options,
)
from .domain.episode import resolve_episode_plan, resolve_scene_config
from .expert.operations import build_operation_handlers
from .execution.navigation import build_navigation_config
from .initialization import initialize_scaffold
from .prompts import resolve_prompt_spec


def _environment(
    config: Any, scene: Any, plan: Any, resources: Mapping[str, Any]
) -> Mapping[str, Any]:
    del resources
    return {
        "water_button_world_xyz": scene.water_button_world_xyz,
        "completion_button_world_xyz": scene.completion_button_world_xyz,
        "cabinet_world_xyz": scene.cabinet_world_xyz,
        "cabinet_yaw_deg": scene.cabinet_yaw_deg,
        "can_place_region_world_xyz": scene.can_place_region_world_xyz,
        "can_place_region_size_xy": scene.can_place_region_size_xy,
        "water_pot_name": scene.water_pot_name,
        "can_name": scene.can_name,
        "sugar_plate_name": scene.sugar_plate_name,
        "hot_chocolate_name": scene.hot_chocolate_name,
        "sugar_names": scene.sugar_names,
        "sugar_plate_world_xyz": scene.sugar_plate_world_xyz,
        "hot_chocolate_world_xyz": scene.hot_chocolate_world_xyz,
        "selected_sugar_names": plan.selected_sugar_names,
        "target_sugar_count": plan.target_sugar_count,
        "total_sugar_count": plan.total_sugar_count,
        "observer_config": asdict(resolve_observer_config(config)),
    }


def _validate(config: Any, episode: Any, execute: bool) -> None:
    if execute and not episode.allow_partial_calibration:
        require_production_calibration(config)


def _release_facts(
    config: Any,
    episode: Any,
    resources: Mapping[str, Any],
    options: Mapping[str, Any],
) -> Mapping[str, Any]:
    del resources, options
    return {
        "route_count": 12,
        "target_sugar_count": 2,
        "total_sugar_count": 3,
        "missing_physical_calibration": list(missing_calibration_resources(config)),
        "missing_scene_positions": [
            name
            for name in ("sugar_plate_world_xyz", "hot_chocolate_world_xyz")
            if config.scene.get(name) is None
        ],
        "allow_partial_calibration": bool(episode.allow_partial_calibration),
    }


_ADAPTER = TaskRuntimeAdapter(
    task_id="TM04",
    load_task_config=load_task_config,
    resolve_task_options=resolve_task_options,
    scene=standard_scene(resolve_scene_config, uses_seed=False),
    plan=standard_plan(resolve_episode_plan),
    actions=config_actions(resolve_action_config),
    navigation=manual_navigation(build_navigation_config),
    prompt=variant_prompt(resolve_prompt_spec),
    environment=_environment,
    initialize=initialize_scaffold,
    diagnostics=default_diagnostics,
    success=default_success,
    close=default_close,
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
    "TASK_RUNTIME_API", "build_descriptor", "check_success", "close_runtime",
    "diagnostics", "initialize_runtime",
]
