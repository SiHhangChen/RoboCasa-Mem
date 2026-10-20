"""Small helpers for task-local RoboCasa runtime descriptors.

The helpers only compose explicit task callbacks.  They never inspect task
IDs, import task packages, or create simulator objects.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from bench.runtime.contracts import EnvironmentDescriptor, RuntimeDescriptor
from bench.runtime.resolver import resolve_runtime_inputs
from bench.tasks.contracts import ResolvedEpisode


def build_descriptor(
    episode: ResolvedEpisode,
    *,
    task_id: str,
    load_task_config: Callable[[str | Path], Any],
    resolve_task_options: Callable[[Mapping[str, Any] | None], Mapping[str, Any]] | None,
    resolve_scene: Callable[[Any, ResolvedEpisode], Any],
    resolve_plan: Callable[[Any, ResolvedEpisode, Mapping[str, Any]], Any],
    resolve_actions: Callable[[Any, Any, Mapping[str, Path]], Any],
    resolve_navigation: Callable[[Any, Mapping[str, Path]], Any] | None,
    resolve_prompt: Callable[[Any, str], Any],
    env_kwargs: Callable[[Any, Any, Any, Mapping[str, Path]], Mapping[str, Any]],
    release_facts: Callable[[Any, Any, Mapping[str, Path]], Mapping[str, Any]] | None = None,
    task_validated_resource_names: tuple[str, ...] = (),
    execute: bool = True,
) -> RuntimeDescriptor:
    if episode.task_id != task_id:
        raise ValueError(f"{task_id} runtime cannot build {episode.task_id!r}")
    config, sensors, resources, options = resolve_runtime_inputs(
        episode,
        load_task_config=load_task_config,
        resolve_task_options=resolve_task_options,
        task_validated_resource_names=task_validated_resource_names,
    )
    scene = resolve_scene(config, episode)
    plan = resolve_plan(config, episode, options)
    actions = resolve_actions(config, plan, resources)
    if actions is None:
        raise TypeError(f"{task_id} action resolver returned None")
    navigation = None if resolve_navigation is None else resolve_navigation(config, resources)
    prompt = resolve_prompt(plan, episode.variant)
    facts = {} if release_facts is None else dict(release_facts(config, plan, resources))
    return RuntimeDescriptor(
        episode=episode,
        config=config,
        scene=scene,
        episode_plan=plan,
        actions=actions,
        navigation=navigation,
        sensors=sensors,
        prompt=prompt,
        resources=resources,
        environment=EnvironmentDescriptor(
            env_name=str(scene.env_name),
            env_split=str(scene.env_split),
            scene_seed=int(scene.scene_seed),
            layout_id=getattr(scene, "layout_id", None),
            style_id=getattr(scene, "style_id", None),
            env_kwargs=dict(env_kwargs(config, scene, plan, resources)),
        ),
        release_facts=facts,
        execute=bool(execute),
    )


def build_standard_descriptor(
    episode: ResolvedEpisode,
    *,
    task_id: str,
    load_task_config: Callable[[str | Path], Any],
    resolve_task_options: Callable[[Mapping[str, Any] | None], Mapping[str, Any]] | None,
    resolve_scene_config: Callable[..., Any],
    resolve_episode_plan: Callable[..., Any],
    resolve_action_config: Callable[..., Any],
    build_navigation_config: Callable[..., Any],
    resolve_prompt_spec: Callable[..., Any],
    env_kwargs: Callable[[Any, Any, Any, Mapping[str, Path]], Mapping[str, Any]],
    release_facts: Callable[[Any, Any, Mapping[str, Path]], Mapping[str, Any]] | None = None,
    scene_uses_seed: bool = True,
    action_mode: str = "config",
    navigation_mode: str = "manual",
    prompt_mode: str = "variant",
    task_validated_resource_names: tuple[str, ...] = (),
    execute: bool = True,
    plan_builder: Callable[[Any, ResolvedEpisode, Mapping[str, Any]], Any] | None = None,
    action_builder: Callable[[Any, Any, Mapping[str, Path]], Any] | None = None,
    navigation_builder: Callable[[Any, Mapping[str, Path]], Any] | None = None,
    prompt_builder: Callable[[Any, str], Any] | None = None,
) -> RuntimeDescriptor:
    """Build the common descriptor shape used by the shipped task leaves."""

    def scene(config: Any, resolved: ResolvedEpisode) -> Any:
        kwargs = {"env_split": resolved.env_split, **dict(config.scene)}
        if scene_uses_seed:
            kwargs["seed"] = resolved.seed
        return resolve_scene_config(**kwargs)

    def plan(config: Any, resolved: ResolvedEpisode, options: Mapping[str, Any]) -> Any:
        if plan_builder is not None:
            return plan_builder(config, resolved, options)
        return resolve_episode_plan(
            seed=resolved.seed,
            **{**dict(config.episode), **dict(options)},
        )

    def actions(config: Any, resolved_plan: Any, resources: Mapping[str, Path]) -> Any:
        if action_builder is not None:
            return action_builder(config, resolved_plan, resources)
        del resources
        if action_mode == "config":
            return resolve_action_config(config)
        if action_mode == "operations":
            return resolve_action_config(operations=config.operations)
        if action_mode == "episode_operations":
            return resolve_action_config(episode=resolved_plan, operations=config.operations)
        if action_mode == "keyword_config":
            return resolve_action_config(config=config)
        raise ValueError(f"unsupported action_mode {action_mode!r}")

    def navigation(config: Any, resources: Mapping[str, Path]) -> Any:
        if navigation_builder is not None:
            return navigation_builder(config, resources)
        kwargs: dict[str, Any] = {"manual_base_path_file": None}
        if navigation_mode in {"manual", "manual_stations"}:
            kwargs["manual_base_path_file"] = resources["manual_base_path"]
        if navigation_mode in {"stations", "manual_stations"}:
            kwargs["station_base_pose_file"] = resources["station_poses" if "station_poses" in resources else "stations"]
        return build_navigation_config(dict(config.navigation), **kwargs)

    def prompt(resolved_plan: Any, variant: str) -> Any:
        if prompt_builder is not None:
            return prompt_builder(resolved_plan, variant)
        if prompt_mode == "variant":
            return resolve_prompt_spec(variant=variant)
        if prompt_mode == "sugar_count":
            return resolve_prompt_spec(
                target_sugar_count=resolved_plan.target_sugar_count,
                variant=variant,
            )
        raise ValueError(f"unsupported prompt_mode {prompt_mode!r}")

    return build_descriptor(
        episode,
        task_id=task_id,
        load_task_config=load_task_config,
        resolve_task_options=resolve_task_options,
        resolve_scene=scene,
        resolve_plan=plan,
        resolve_actions=actions,
        resolve_navigation=navigation,
        resolve_prompt=prompt,
        env_kwargs=env_kwargs,
        release_facts=release_facts,
        task_validated_resource_names=task_validated_resource_names,
        execute=execute,
    )


__all__ = ["build_descriptor", "build_standard_descriptor"]
