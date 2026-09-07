"""WR-01 task-local RoboCasa runtime composition."""

from __future__ import annotations

from collections.abc import Mapping
import json
from pathlib import Path
from typing import Any

from bench.runtime import TaskRuntimeAdapter, default_close
from bench.tasks.runtime_support import cabinet_initialization, manual_navigation, operations_actions, standard_plan, standard_scene
from bench.tasks.wr01.config import load_task_config, resolve_action_config
from bench.tasks.wr01.domain.episode import resolve_episode_plan, resolve_scene_config
from bench.tasks.wr01.domain.contracts import TASK_REVISION
from bench.tasks.wr01.evaluation import (
    REPORT_STAGE_PROFILE_ID,
    TASK_EVALUATOR_METADATA_KEY,
    WR01Evaluator,
    diagnostics_for_runtime,
    report_stages_for_runtime,
    success_for_runtime,
)
from bench.tasks.wr01.execution.navigation import build_navigation_config
from bench.tasks.wr01.expert.operations import build_operation_handlers
from bench.tasks.wr01.initialization import apply_robot_base_initial_state, apply_robot_torso_initial_state, initialize_episode_state
from bench.tasks.wr01.prompts import resolve_prompt_spec


def _environment(config: Any, scene: Any, plan: Any, resources: Mapping[str, Any]) -> Mapping[str, Any]:
    del scene, resources
    cabinet = dict(config.initial_state.cabinets)
    semantics = dict(config.semantics)
    manipulation = dict(config.manipulation)
    gripper = dict(manipulation.get("gripper", {}))
    return {"cabinet_world_xyz": tuple(float(v) for v in cabinet["world_xyz"]),
            "cabinet_yaw_deg": float(cabinet.get("yaw_deg", 0.0)),
            "upper_cabinet_z_offset": float(cabinet["upper_z_offset"]),
            "object_scale": float(config.initial_state.objects.get("object_scale", 0.90)),
            "item_placement_seed": int(plan.seed),
            "final_closed_threshold": float(semantics.get("final_closed_threshold", 0.08)),
            "cabinet_contains_margin": float(semantics.get("cabinet_contains_margin", 0.04)),
            "physical_event_dwell_steps": int(semantics.get("physical_event_dwell_steps", 3)),
            "gripper_force_limit": float(gripper.get("force_limit", 20.0)),
            "gripper_pad_friction": tuple(
                float(value)
                for value in gripper.get("pad_friction", (2.0, 0.05, 0.0001))
            )}


_initialize_cabinet = cabinet_initialization(
    initialize_episode_state,
    apply_robot_base_initial_state,
    apply_robot_torso_initial_state,
)


def _initialize(descriptor: Any, env: Any, observation: Mapping[str, Any]):
    updated, initialization, metadata, targets = _initialize_cabinet(
        descriptor,
        env,
        observation,
    )
    env.configure_canonical_reset_reference(updated)
    _configure_canonical_reset_region(env, descriptor.resources["stations"])
    env.start_physical_observer()
    evaluator = WR01Evaluator(
        storage_order=descriptor.episode_plan.storage_order,
        target_item_id=descriptor.episode_plan.target_item_id,
        target_level=descriptor.episode_plan.target_level,
        close_order=descriptor.episode_plan.close_order,
    )
    evaluator.reset(env.evaluation_snapshot())
    env._wr01_evaluator = evaluator
    return (
        updated,
        initialization,
        {**dict(metadata), TASK_EVALUATOR_METADATA_KEY: evaluator},
        targets,
    )


def _configure_canonical_reset_region(env: Any, stations_path: Any) -> None:
    payload = json.loads(Path(stations_path).read_text(encoding="utf-8"))
    circle = dict(
        (payload.get("stations") or {}).get("cabinet_front_station", {}).get("circle")
        or {}
    )
    if not circle:
        raise ValueError("WR-01 stations must define cabinet_front_station.circle")
    env.configure_canonical_reset_region(
        center_xy=circle["center_xy"],
        radius_m=float(circle["radius_m"]),
        base_yaw=float(circle["base_yaw"]),
        torso_height_m=float(circle.get("torso_height", 0.20)),
    )


_ADAPTER = TaskRuntimeAdapter(
    task_id="WR-01", load_task_config=load_task_config, resolve_task_options=None,
    scene=standard_scene(resolve_scene_config), plan=standard_plan(resolve_episode_plan),
    actions=operations_actions(resolve_action_config), navigation=manual_navigation(build_navigation_config),
    prompt=lambda plan, variant: resolve_prompt_spec(target_item_id=plan.target_item_id, variant=variant),
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
