"""Build WA-01's task-local navigation and manipulation runtime."""

from typing import Any

from bench.execution.runner import StageRunner
from bench.tasks.wa01.execution.handlers import build_stage_handlers
from bench.tasks.wa01.execution.manipulation import build_manipulation_assembly
from bench.tasks.wa01.execution.navigation import (
    build_navigation_assembly,
    resolve_station_randomization,
)


def build_operation_handlers(descriptor: Any, env: Any, observation: dict[str, Any]) -> dict[str, Any]:
    navigation = build_navigation_assembly(
        env,
        observation,
        config=descriptor.navigation,
        seed=descriptor.episode_plan.seed,
        station_sampler=getattr(env, "_wa01_station_sampler", None),
        randomization=(
            getattr(env, "_wa01_station_randomization", None)
            or resolve_station_randomization(descriptor.config.navigation)
        ),
    )
    manipulation = build_manipulation_assembly(env, observation, settings=descriptor.config.manipulation, planner_script=descriptor.resources["planner_script"], execute=descriptor.execute)
    handler_settings = descriptor.actions.handler_settings
    manipulation.controller.operation_handlers = build_stage_handlers(
        manipulation.controller,
        accept_sugar_delivery=bool(
            handler_settings.get("accept_sugar_delivery", True)
        ),
        sugar_gripper_force_limit=float(
            handler_settings.get("sugar_gripper_force_limit", 5.0)
        ),
        default_gripper_force_limit=float(
            handler_settings.get("default_gripper_force_limit", 20.0)
        ),
    )
    return {"runner": StageRunner(navigation=navigation.controller, manipulation=manipulation.controller)}


__all__ = ["build_operation_handlers"]
