"""TS03 task-local RoboCasa runtime composition."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from bench.runtime import TaskRuntimeAdapter, default_close, default_diagnostics, default_success
from bench.tasks.runtime_support import episode_actions, manual_navigation, standard_plan, standard_scene, sugar_initialization, sugar_prompt
from bench.tasks.ts03.config import load_task_config, resolve_action_config, resolve_task_options
from bench.tasks.ts03.domain.episode import resolve_episode_plan, resolve_scene_config
from bench.tasks.ts03.domain.contracts import episode_profile_metadata
from bench.tasks.ts03.execution.navigation import build_navigation_config
from bench.tasks.ts03.expert.operations import build_operation_handlers
from bench.tasks.ts03.initialization import apply_fixed_initial_object_regions, apply_robot_base_initial_state
from bench.tasks.ts03.prompts import resolve_prompt_spec


def _environment(config: Any, scene: Any, plan: Any, resources: Mapping[str, Any]) -> Mapping[str, Any]:
    del config
    return {"target_sugar_count": plan.target_sugar_count,
            "total_sugar_count": plan.total_sugar_count,
            "sugar_names": plan.sugar_names,
            "target_sugar_names": plan.target_sugar_names,
            "sugar_plate_name": scene.sugar_plate_name,
            "receptacle_name": scene.receptacle_name,
            "completion_button_position_file": str(resources["completion_button_position"])}


_ADAPTER = TaskRuntimeAdapter(
    task_id="TS03", load_task_config=load_task_config, resolve_task_options=resolve_task_options,
    scene=standard_scene(resolve_scene_config), plan=standard_plan(resolve_episode_plan),
    actions=episode_actions(resolve_action_config), navigation=manual_navigation(build_navigation_config),
    prompt=sugar_prompt(resolve_prompt_spec), environment=_environment,
    initialize=sugar_initialization(apply_fixed_initial_object_regions, apply_robot_base_initial_state),
    diagnostics=default_diagnostics, success=default_success, close=default_close,
    release_facts=lambda config, episode, resources, options: {
        "task_revision": str(getattr(config, "task", {}).get("revision", "TS03-v1")),
        **episode_profile_metadata(type("EpisodeFacts", (), dict(episode.facts))()),
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
