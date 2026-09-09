"""Build WX-02's task-local navigation and manipulation runtime."""

from typing import Any

from bench.execution.runner import StageRunner
from bench.tasks.wx02.execution.geometry import make_cup_pick_payload_selector, resolve_episode_targets
from bench.tasks.wx02.execution.handlers import build_stage_handlers
from bench.tasks.wx02.execution.manipulation import build_manipulation_assembly
from bench.tasks.wx02.execution.navigation import build_navigation_assembly


def build_operation_handlers(descriptor: Any, env: Any, observation: dict[str, Any]) -> dict[str, Any]:
    targets = resolve_episode_targets(env, descriptor.episode_plan)
    manipulation = build_manipulation_assembly(
        env,
        observation,
        settings=descriptor.config.manipulation,
        planner_script=descriptor.resources["planner_script"],
        execute=descriptor.execute,
    )
    navigation = build_navigation_assembly(
        env,
        observation,
        config=descriptor.navigation,
        seed=descriptor.episode_plan.seed,
        holding_max_end_effector_distance=descriptor.config.initial_state.short_start.max_end_effector_distance,
        holding_max_relative_offset_drift=descriptor.config.initial_state.short_start.max_relative_offset_drift,
        holding_minimum_motion_ratio=descriptor.config.initial_state.short_start.minimum_motion_ratio,
    )
    manipulation.controller.operation_handlers = build_stage_handlers(
        manipulation.controller,
        navigation=navigation.controller,
        settings=descriptor.actions.handler_settings,
        targets=targets,
        cleaning_tool_grasp_pose_file=descriptor.resources[
            "cleaning_tool_grasp_pose"
        ],
    )
    manipulation.controller.pick_payload_selector = make_cup_pick_payload_selector(
        manipulation.controller,
        manual_candidates_file=descriptor.resources["cup_grasp_candidates"],
        maximum_base_x=float(descriptor.actions.pick_proxy_settings.get("object_max_base_x", 0.66)),
        dry_run_only=bool(descriptor.actions.pick_proxy_settings.get("dry_run_only", True)),
    )
    return {"runner": StageRunner(navigation=navigation.controller, manipulation=manipulation.controller)}


__all__ = ["build_operation_handlers"]
