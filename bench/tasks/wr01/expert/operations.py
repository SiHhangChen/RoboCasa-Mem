"""Build WR-01's task-local navigation and manipulation runtime."""

from typing import Any

from bench.execution.runner import StageRunner
from bench.tasks.wr01.execution.handlers import build_stage_handlers
from bench.tasks.wr01.execution.manipulation import build_manipulation_assembly
from bench.tasks.wr01.execution.navigation import build_navigation_assembly


_CALIBRATION_RESOURCES = (
    "lower_door_grasp_pose", "lower_retrieval_prepose", "upper_door_grasp_pose",
)


def build_operation_handlers(descriptor: Any, env: Any, observation: dict[str, Any]) -> dict[str, Any]:
    navigation = build_navigation_assembly(env, observation, config=descriptor.navigation, seed=descriptor.episode_plan.seed, stations_file=descriptor.resources["stations"])
    manipulation = build_manipulation_assembly(env, observation, settings=descriptor.config.manipulation, planner_script=descriptor.resources["planner_script"], execute=descriptor.execute)
    manipulation.controller.operation_handlers = build_stage_handlers(manipulation.controller, resources={name: descriptor.resources[name] for name in _CALIBRATION_RESOURCES})
    return {"runner": StageRunner(navigation=navigation.controller, manipulation=manipulation.controller)}


__all__ = ["build_operation_handlers"]
