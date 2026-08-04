"""Build TS04's task-local navigation and manipulation runtime."""

from typing import Any

from bench.execution.runner import StageRunner
from bench.tasks.ts04.execution.handlers import build_stage_handlers
from bench.tasks.ts04.execution.manipulation import build_manipulation_assembly
from bench.tasks.ts04.execution.navigation import build_navigation_assembly


def build_operation_handlers(descriptor: Any, env: Any, observation: dict[str, Any]) -> dict[str, Any]:
    navigation = build_navigation_assembly(env, observation, config=descriptor.navigation, seed=descriptor.scene.scene_seed)
    manipulation = build_manipulation_assembly(env, observation, settings=descriptor.config.manipulation, planner_script=descriptor.resources["planner_script"], execute=descriptor.execute)
    manipulation.controller.operation_handlers = build_stage_handlers(manipulation.controller)
    return {"runner": StageRunner(navigation=navigation.controller, manipulation=manipulation.controller)}


__all__ = ["build_operation_handlers"]
