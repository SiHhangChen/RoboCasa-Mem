"""Build WR-05's task-local navigation and manipulation runtime."""

from typing import Any

from bench.execution.runner import StageRunner
from bench.tasks.wr05.execution.handlers import build_operation_handlers as build_stage_handlers
from bench.tasks.wr05.execution.manipulation import build_manipulation_assembly
from bench.tasks.wr05.execution.navigation import build_navigation_assembly
from bench.tasks.wr05.execution.geometry import resolve_episode_targets


def build_operation_handlers(descriptor: Any, env: Any, observation: dict[str, Any]) -> dict[str, Any]:
    targets = resolve_episode_targets(env, descriptor.episode_plan)
    navigation = build_navigation_assembly(
        env,
        observation,
        nav_config=descriptor.navigation,
        seed=descriptor.episode_plan.seed,
        return_target_xy_by_region={
            descriptor.episode_plan.home_assignment[name]: tuple(
                float(value) for value in targets.return_place_sampling[name]["sampled_target_xy"]
            )
            for name in descriptor.episode_plan.transport_order
        },
    )
    manipulation = build_manipulation_assembly(env, observation, settings=descriptor.config.manipulation, planner_script=descriptor.resources["planner_script"], execute=descriptor.execute)
    manipulation.controller.operation_handlers = build_stage_handlers(
        manipulation.controller,
        dry_run_only=bool(descriptor.actions.handler_settings.get("accessible_place_dry_run_only", True)),
        targets=targets,
    )
    return {"runner": StageRunner(navigation=navigation.controller, manipulation=manipulation.controller)}


__all__ = ["build_operation_handlers"]
