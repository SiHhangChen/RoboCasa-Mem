"""Composable task-local RoboCasa runtime adapter.

This module owns only lifecycle plumbing.  Every task supplies its own scene,
episode, action, navigation, prompt, environment kwargs, initialization, and
semantic callbacks; no task ID or physical operation is encoded here.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from bench.runtime.contracts import (
    EnvironmentDescriptor,
    InitializedTaskRuntime,
    RuntimeDescriptor,
    TaskRuntimeAPI,
)
from bench.runtime.resolver import resolve_runtime_inputs
from bench.runtime.sensors import load_observation_profile
from bench.tasks.contracts import ResolvedEpisode


SceneBuilder = Callable[[Any, ResolvedEpisode], Any]
PlanBuilder = Callable[[Any, ResolvedEpisode, Mapping[str, Any]], Any]
ActionsBuilder = Callable[[Any, Any, Mapping[str, Path]], Any]
NavigationBuilder = Callable[[Any, Mapping[str, Path]], Any]
PromptBuilder = Callable[[Any, str], Any]
EnvironmentBuilder = Callable[[Any, Any, Any, Mapping[str, Path]], Mapping[str, Any]]
InitializationBuilder = Callable[
    [RuntimeDescriptor, Any, Mapping[str, Any]],
    tuple[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any], Any],
]
DiagnosticsBuilder = Callable[[InitializedTaskRuntime], Mapping[str, Any]]
ReportStagesBuilder = Callable[[InitializedTaskRuntime], Mapping[str, Any]]
SuccessBuilder = Callable[[InitializedTaskRuntime, Mapping[str, Any]], bool]
CloseBuilder = Callable[[InitializedTaskRuntime], None]
DescriptorValidator = Callable[[Any, ResolvedEpisode, bool], None]
ReleaseFactsBuilder = Callable[
    [Any, ResolvedEpisode, Mapping[str, Path], Mapping[str, Any]],
    Mapping[str, Any],
]
ExpertRuntimeBuilder = Callable[
    [RuntimeDescriptor, Any, Mapping[str, Any]], Mapping[str, Any]
]


def lazy_expert_runtime(symbol: str) -> ExpertRuntimeBuilder:
    """Load optional demonstration code only when an expert runtime is requested."""
    module_name, separator, attribute = symbol.partition(":")
    if not separator or not module_name or not attribute:
        raise ValueError("expert runtime reference must be 'module:attribute'")

    def build(*args: Any, **kwargs: Any) -> Mapping[str, Any]:
        from importlib import import_module

        return getattr(import_module(module_name), attribute)(*args, **kwargs)

    return build


class TaskRuntimeAdapter:
    """Bind task-owned pure and physical callbacks to the shared runtime API."""

    def __init__(
        self,
        *,
        task_id: str,
        load_task_config: Callable[[str | Path], Any],
        resolve_task_options: Callable[[Mapping[str, Any] | None], Mapping[str, Any]] | None,
        scene: SceneBuilder,
        plan: PlanBuilder,
        actions: ActionsBuilder,
        navigation: NavigationBuilder | None,
        prompt: PromptBuilder,
        environment: EnvironmentBuilder,
        initialize: InitializationBuilder,
        diagnostics: DiagnosticsBuilder,
        success: SuccessBuilder,
        close: CloseBuilder,
        report_stage_profile_id: str | None = None,
        report_stages: ReportStagesBuilder | None = None,
        validate_descriptor: DescriptorValidator | None = None,
        release_facts: ReleaseFactsBuilder | None = None,
        expert_runtime: ExpertRuntimeBuilder | None = None,
        subtask_scheduler: Callable[[Any, str], Any] | None = None,
        task_validated_resource_names: tuple[str, ...] = (),
    ) -> None:
        self.task_id = str(task_id)
        self._load_task_config = load_task_config
        self._resolve_task_options = resolve_task_options
        self._scene = scene
        self._plan = plan
        self._actions = actions
        self._navigation = navigation
        self._prompt = prompt
        self._environment = environment
        self._initialize = initialize
        self._diagnostics = diagnostics
        self._success = success
        self._close = close
        self._report_stage_profile_id = (
            None
            if report_stage_profile_id is None
            else str(report_stage_profile_id)
        )
        self._report_stages = report_stages
        self._validate_descriptor = validate_descriptor
        self._release_facts = release_facts
        self._expert_runtime = expert_runtime
        self._subtask_scheduler = subtask_scheduler
        self._validated = tuple(task_validated_resource_names)
        if not callable(actions):
            raise TypeError(f"{self.task_id} runtime requires an ActionConfig resolver")
        if (self._report_stage_profile_id is None) != (self._report_stages is None):
            raise ValueError(
                f"{self.task_id} runtime must define report_stage_profile_id and "
                "report_stages together"
            )

    def build_descriptor(self, episode: ResolvedEpisode, execute: bool = True) -> RuntimeDescriptor:
        if episode.task_id != self.task_id:
            raise ValueError(f"{self.task_id} runtime cannot build {episode.task_id!r}")
        config, sensors, resources, options = resolve_runtime_inputs(
            episode,
            load_task_config=self._load_task_config,
            resolve_task_options=self._resolve_task_options,
            task_validated_resource_names=self._validated,
        )
        if self._validate_descriptor is not None:
            self._validate_descriptor(config, episode, bool(execute))
        scene = self._scene(config, episode)
        plan = self._plan(config, episode, options)
        actions = self._actions(config, plan, resources)
        if actions is None:
            raise TypeError(f"{self.task_id} action resolver returned None")
        navigation = None if self._navigation is None else self._navigation(config, resources)
        prompt = self._prompt(plan, episode.variant)
        generic_release_facts: dict[str, Any] = {}
        manipulation = getattr(config, "manipulation", None)
        if isinstance(manipulation, Mapping):
            runtime_settings = manipulation.get("runtime", {})
            if isinstance(runtime_settings, Mapping) and "max_joint_step" in runtime_settings:
                generic_release_facts["controller"] = {
                    "max_joint_step": float(runtime_settings["max_joint_step"])
                }
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
                env_kwargs=dict(self._environment(config, scene, plan, resources)),
            ),
            release_facts={
                "task_id": self.task_id,
                "config_hash": episode.config_hash,
                "resource_hash": episode.resource_hash,
                **generic_release_facts,
                **(
                    dict(self._release_facts(config, episode, resources, options))
                    if self._release_facts is not None
                    else {}
                ),
            },
            execute=bool(execute),
        )

    def initialize_runtime(
        self,
        descriptor: RuntimeDescriptor,
        env: Any,
        observation: Mapping[str, Any],
        metadata: Mapping[str, Any],
    ) -> InitializedTaskRuntime:
        updated, initialization, extra_metadata, targets = self._initialize(
            descriptor, env, observation
        )
        runtime_metadata = dict(metadata)
        runtime_metadata.update(dict(extra_metadata))
        return InitializedTaskRuntime(
            descriptor=descriptor,
            env=env,
            observation=dict(updated),
            targets=targets,
            initialization=dict(initialization),
            metadata=runtime_metadata,
        )

    def check_success(self, runtime: InitializedTaskRuntime, info: Mapping[str, Any]) -> bool:
        return bool(self._success(runtime, info))

    def diagnostics(self, runtime: InitializedTaskRuntime) -> Mapping[str, Any]:
        return dict(self._diagnostics(runtime) or {})

    def report_stages(
        self,
        runtime: InitializedTaskRuntime,
    ) -> Mapping[str, Any] | None:
        if self._report_stages is None:
            return None
        return dict(self._report_stages(runtime) or {})

    def close_runtime(self, runtime: InitializedTaskRuntime) -> None:
        self._close(runtime)

    def api(self) -> TaskRuntimeAPI:
        return TaskRuntimeAPI(
            build_descriptor=self.build_descriptor,
            initialize_runtime=self.initialize_runtime,
            check_success=self.check_success,
            diagnostics=self.diagnostics,
            close_runtime=self.close_runtime,
            report_stage_profile_id=self._report_stage_profile_id,
            report_stages=(
                self.report_stages if self._report_stages is not None else None
            ),
            build_expert_runtime=self._expert_runtime,
            build_subtask_scheduler=self._subtask_scheduler,
        )


def default_success(runtime: InitializedTaskRuntime, info: Mapping[str, Any]) -> bool:
    del info
    checker = getattr(runtime.env, "_check_success", None)
    if not callable(checker):
        raise AttributeError("task environment has no _check_success()")
    return bool(checker())


def default_diagnostics(runtime: InitializedTaskRuntime) -> Mapping[str, Any]:
    for name in ("task_diagnostics",):
        method = getattr(runtime.env, name, None)
        if callable(method):
            return dict(method() or {})
    return {"success": default_success(runtime, {})}


def default_close(runtime: InitializedTaskRuntime) -> None:
    close = getattr(runtime.env, "close", None)
    if callable(close):
        close()


__all__ = [
    "TaskRuntimeAdapter",
    "default_close",
    "default_diagnostics",
    "default_success",
]
