"""Generic callback adapters used by task-local runtime composition roots."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from bench.runtime.contracts import RuntimeDescriptor
from bench.tasks.contracts import ResolvedEpisode


def standard_scene(resolve_scene_config: Callable[..., Any], *, uses_seed: bool = True):
    def build(config: Any, episode: ResolvedEpisode) -> Any:
        values = {"env_split": episode.env_split, **dict(config.scene)}
        if uses_seed:
            values["seed"] = episode.seed
        return resolve_scene_config(**values)
    return build


def standard_plan(resolve_episode_plan: Callable[..., Any]):
    def build(config: Any, episode: ResolvedEpisode, options: Mapping[str, Any]) -> Any:
        return resolve_episode_plan(
            seed=episode.seed,
            **{**dict(config.episode), **dict(options)},
        )
    return build


def config_actions(resolve_action_config: Callable[..., Any]):
    return lambda config, plan, resources: resolve_action_config(config)


def operations_actions(resolve_action_config: Callable[..., Any]):
    return lambda config, plan, resources: resolve_action_config(operations=config.operations)


def episode_actions(resolve_action_config: Callable[..., Any]):
    return lambda config, plan, resources: resolve_action_config(
        episode=plan, operations=config.operations
    )


def keyword_config_actions(resolve_action_config: Callable[..., Any]):
    return lambda config, plan, resources: resolve_action_config(config=config)


def manual_navigation(build_navigation_config: Callable[..., Any]):
    return lambda config, resources: build_navigation_config(
        dict(config.navigation),
        manual_base_path_file=resources["manual_base_path"],
    )


def station_navigation(build_navigation_config: Callable[..., Any]):
    def build(config: Any, resources: Mapping[str, Path]) -> Any:
        return build_navigation_config(
            dict(config.navigation),
            manual_base_path_file=resources.get("manual_base_path"),
            station_base_pose_file=resources[
                "station_poses" if "station_poses" in resources else "stations"
            ],
        )
    return build


def manual_station_navigation(build_navigation_config: Callable[..., Any]):
    def build(config: Any, resources: Mapping[str, Path]) -> Any:
        return build_navigation_config(
            dict(config.navigation),
            manual_base_path_file=resources["manual_base_path"],
            station_base_pose_file=resources[
                "station_poses" if "station_poses" in resources else "stations"
            ],
        )
    return build


def variant_prompt(resolve_prompt_spec: Callable[..., Any]):
    return lambda plan, variant: resolve_prompt_spec(variant=variant)


def sugar_prompt(resolve_prompt_spec: Callable[..., Any]):
    return lambda plan, variant: resolve_prompt_spec(
        target_sugar_count=plan.target_sugar_count,
        variant=variant,
    )


def path_base_initialization(apply_manual_path_initial_base_pose: Callable[..., Any]):
    def initialize(descriptor: RuntimeDescriptor, env: Any, observation: Mapping[str, Any]):
        updated, base = apply_manual_path_initial_base_pose(
            env,
            dict(observation),
            path_file=descriptor.resources["manual_base_path"],
            sample_index=int(
                descriptor.config.initial_state.get("robot_base_sample_index", 0)
            ),
        )
        return updated, {"base": base}, {}, None
    return initialize


def sugar_initialization(
    apply_fixed_initial_object_regions: Callable[..., Any],
    apply_robot_base_initial_state: Callable[..., Any],
):
    def initialize(descriptor: RuntimeDescriptor, env: Any, observation: Mapping[str, Any]):
        settings = dict(descriptor.config.initial_state.objects)
        enabled = bool(settings.pop("enabled", True))
        updated, objects = apply_fixed_initial_object_regions(
            env,
            episode=descriptor.episode_plan,
            settings=settings,
            enabled=enabled,
        )
        updated, base = apply_robot_base_initial_state(
            env,
            dict(updated),
            config=descriptor.config.initial_state.robot_base,
            nav_config=descriptor.navigation,
            seed=descriptor.scene.scene_seed,
        )
        return updated, {"objects": objects, "base": base}, {}, None
    return initialize


def drawer_initialization(
    apply_initial_drawer_state: Callable[..., Any],
    apply_robot_base_initial_state: Callable[..., Any],
):
    def initialize(descriptor: RuntimeDescriptor, env: Any, observation: Mapping[str, Any]):
        settings = dict(descriptor.config.initial_state.drawers)
        enabled = bool(settings.pop("enabled", True))
        updated, drawers = apply_initial_drawer_state(env, enabled=enabled, **settings)
        updated, base = apply_robot_base_initial_state(
            env,
            dict(updated),
            config=descriptor.config.initial_state.robot_base,
            nav_config=descriptor.navigation,
            seed=descriptor.scene.scene_seed,
        )
        return updated, {"drawers": drawers, "base": base}, {}, None
    return initialize


def shelf_initialization(
    initialize_episode_state: Callable[..., Any],
    resolve_episode_targets: Callable[..., Any],
):
    def initialize(descriptor: RuntimeDescriptor, env: Any, observation: Mapping[str, Any]):
        updated, setup = initialize_episode_state(
            env,
            dict(observation),
            episode=descriptor.episode_plan,
            robot_base=descriptor.config.initial_state.robot_base,
            objects=dict(descriptor.config.initial_state.objects),
        )
        targets = resolve_episode_targets(env, descriptor.episode_plan)
        return updated, {"setup": setup}, {}, targets
    return initialize


def cabinet_initialization(
    initialize_episode_state: Callable[..., Any],
    apply_robot_base_initial_state: Callable[..., Any],
    apply_robot_torso_initial_state: Callable[..., Any],
):
    def initialize(descriptor: RuntimeDescriptor, env: Any, observation: Mapping[str, Any]):
        updated, setup = initialize_episode_state(
            env,
            episode=descriptor.episode_plan,
            cabinets=dict(descriptor.config.initial_state.cabinets),
        )
        updated, base = apply_robot_base_initial_state(
            env,
            dict(updated),
            config=descriptor.config.initial_state.robot_base,
        )
        updated, torso = apply_robot_torso_initial_state(
            env,
            dict(updated),
            target_height_m=descriptor.config.initial_state.robot_torso_height_m,
        )
        return updated, {"setup": setup, "base": base, "torso": torso}, {}, None
    return initialize


__all__ = [
    "cabinet_initialization",
    "config_actions",
    "drawer_initialization",
    "episode_actions",
    "keyword_config_actions",
    "manual_navigation",
    "manual_station_navigation",
    "operations_actions",
    "path_base_initialization",
    "shelf_initialization",
    "standard_plan",
    "standard_scene",
    "station_navigation",
    "sugar_initialization",
    "sugar_prompt",
    "variant_prompt",
]
