"""Build TS03's task-local navigation and manipulation runtime."""

from typing import Any

from bench.execution.runner import StageRunner
from bench.tasks.ts03.execution.geometry import build_pick_payload_selector
from bench.tasks.ts03.execution.handlers import build_stage_handlers
from bench.tasks.ts03.execution.manipulation import build_manipulation_assembly
from bench.tasks.ts03.execution.navigation import build_navigation_assembly


def build_operation_handlers(descriptor: Any, env: Any, observation: dict[str, Any]) -> dict[str, Any]:
    navigation = build_navigation_assembly(
        env,
        observation,
        config=descriptor.navigation,
        seed=descriptor.episode_plan.seed,
    )
    manipulation = build_manipulation_assembly(env, observation, settings=descriptor.config.manipulation, planner_script=descriptor.resources["planner_script"], execute=descriptor.execute)
    manipulation.controller.pick_payload_selector = build_pick_payload_selector(
        descriptor.actions.grasp_yaw_offsets_deg,
        grasp_finger_offset_x=getattr(descriptor.actions, "grasp_finger_offset_x", 0.0),
        grasp_finger_offset_y=getattr(descriptor.actions, "grasp_finger_offset_y", 0.0),
    )
    manipulation.controller.operation_handlers = build_stage_handlers(manipulation.controller, handler_settings=descriptor.actions.handler_settings)
    return {"runner": StageRunner(navigation=navigation.controller, manipulation=manipulation.controller)}


__all__ = ["build_operation_handlers"]
