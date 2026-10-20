"""RoboCasa-backed implementation of the shared runtime contract.

The factory consumes only a canonical ``ResolvedEpisode`` and an explicit
task-local ``TaskRuntimeAPI``.  No unsupported context, bundle, or workflow object
is created on this path.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from bench.runtime.contracts import (
    CONTROL_FREQUENCY_HZ,
    InitializedTaskRuntime,
    InitializationReport,
    RuntimeDescriptor,
    RuntimeStep,
    TaskRuntimeAPI,
    TaskDiagnostics,
)
from bench.tasks.contracts import ResolvedEpisode
from bench.runtime.data_contract import validate_action


class RoboCasaEpisodeRuntime:
    """Wrap one initialized RoboCasa environment as a canonical runtime."""

    def __init__(
        self,
        episode: ResolvedEpisode,
        initialized: InitializedTaskRuntime,
        runtime_api: TaskRuntimeAPI,
        *,
        expert: bool = False,
    ) -> None:
        self.episode = episode
        self._initialized = initialized
        self._runtime_api = runtime_api
        self.env = initialized.env
        self.observation = dict(initialized.observation or {})
        self._closed = False
        self._expert = bool(expert)
        self._execution_stopped = False

    @property
    def initialized(self) -> Any:
        return self._initialized

    def reset(self) -> Mapping[str, Any]:
        if self._closed:
            raise RuntimeError("cannot reset a closed episode runtime")
        from bench.env.robocasa_runtime import reset_env_with_initial_arm_qpos

        observation = reset_env_with_initial_arm_qpos(self.env)
        metadata = dict(self._initialized.metadata)
        reinitialized = self._runtime_api.initialize_runtime(
            self._initialized.descriptor,
            self.env,
            self._normalize_observation(observation),
            metadata,
        )
        if not isinstance(reinitialized, InitializedTaskRuntime):
            raise TypeError("task initialize_runtime() must return InitializedTaskRuntime")
        self._initialized = reinitialized
        self._attach_expert_runtime()
        self.observation = self._normalize_observation(reinitialized.observation)
        self._execution_stopped = False
        return self.observation

    def initialize(self) -> InitializationReport:
        facts = getattr(self._initialized, "initialization", {}) or {}
        return InitializationReport.applied(facts=facts)

    def observe(self) -> Mapping[str, Any]:
        if self._closed:
            raise RuntimeError("cannot observe a closed episode runtime")
        getter = getattr(self.env, "_get_observations", None)
        if callable(getter):
            try:
                value = getter(force_update=True)
            except TypeError:
                value = getter()
            self.observation = self._normalize_observation(value)
        return self.observation

    def step(self, action: Sequence[float] | Any) -> RuntimeStep:
        if self._closed:
            raise RuntimeError("cannot step a closed episode runtime")
        if self._execution_stopped:
            raise RuntimeError("task stopped execution; reset before stepping again")
        # The runtime is the single action boundary for policy evaluation and
        # therefore validates the exact vector that is handed to RoboSuite.
        action_value = validate_action(action)
        result = self.env.step(action_value)
        if not isinstance(result, tuple):
            raise TypeError("RoboCasa env.step() must return a tuple")
        if len(result) == 5:
            observation, reward, terminated, truncated, info = result
        elif len(result) == 4:
            observation, reward, done, info = result
            terminated, truncated = bool(done), False
        else:
            raise ValueError(f"unsupported env.step() result length: {len(result)}")
        self.observation = self._normalize_observation(observation)
        diagnostics = self.diagnostics()
        return RuntimeStep(
            observation=self.observation,
            reward=float(reward),
            terminated=bool(terminated),
            truncated=bool(truncated),
            info=dict(info or {}) if isinstance(info, Mapping) else {},
            diagnostics=diagnostics,
        )

    def diagnostics(self) -> TaskDiagnostics:
        values = self._runtime_api.diagnostics(self._initialized)
        details = dict(values or {}) if isinstance(values, Mapping) else {}
        if "success" in details:
            success = bool(details["success"])
        else:
            success = bool(self._runtime_api.check_success(self._initialized, {}))
        failure = details.get("failure_code")
        stop_execution = bool(details.get("stop_execution", False))
        self._execution_stopped = self._execution_stopped or stop_execution
        return TaskDiagnostics(
            success=success,
            terminal=stop_execution or success or bool(details.get("terminal", False)),
            failure_code=None if failure is None else str(failure),
            details=details,
            stop_execution=stop_execution,
        )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._runtime_api.close_runtime(self._initialized)

    def _attach_expert_runtime(self) -> None:
        builder = self._runtime_api.build_expert_runtime
        if not self._expert or builder is None:
            return
        parts = dict(
            builder(
                self._initialized.descriptor,
                self.env,
                dict(self._initialized.observation),
            )
            or {}
        )
        self._initialized.runner = parts.get("runner")
        self._initialized.metadata.update(dict(parts.get("metadata", {}) or {}))

    @staticmethod
    def _normalize_observation(value: Any) -> dict[str, Any]:
        if not isinstance(value, Mapping):
            raise TypeError("RoboCasa observation must be a mapping")
        return dict(value)


class RoboCasaRuntimeFactory:
    """Create active runtimes from immutable episodes and task-local hooks."""

    def __init__(self, runtime_api: TaskRuntimeAPI) -> None:
        self.runtime_api = runtime_api

    def create(
        self,
        episode: ResolvedEpisode,
        *,
        execute: bool = True,
        expert: bool = False,
        context: Any | None = None,
    ) -> RoboCasaEpisodeRuntime:
        if context is not None:
            raise TypeError("RoboCasaRuntimeFactory no longer accepts unsupported context")
        descriptor = self.runtime_api.build_descriptor(episode, bool(execute))
        return self.create_from_descriptor(descriptor, expert=expert)

    def create_from_descriptor(
        self,
        descriptor: RuntimeDescriptor,
        *,
        expert: bool = False,
    ) -> RoboCasaEpisodeRuntime:
        """Create a simulator from an already-resolved descriptor.

        Expert callers build and validate their plan between descriptor
        resolution and this method.  Keeping this entry point separate also
        guarantees that ActionConfig is resolved exactly once.
        """

        if not isinstance(descriptor, RuntimeDescriptor):
            raise TypeError("task build_descriptor() must return RuntimeDescriptor")
        from bench.env.robocasa_runtime import (
            make_joint_position_controller_config,
            make_robocasa_env,
            reset_env_with_initial_arm_qpos,
        )

        controller_config = make_joint_position_controller_config()
        environment = descriptor.environment
        sensors = descriptor.sensors
        env = make_robocasa_env(
            env_name=environment.env_name,
            split=environment.env_split,
            seed=environment.scene_seed,
            controller_config=controller_config,
            env_kwargs=dict(environment.env_kwargs),
            camera_names=sensors.camera_names,
            camera_widths=sensors.width,
            camera_heights=sensors.height,
            layout_id=environment.layout_id,
            style_id=environment.style_id,
            has_renderer=sensors.has_renderer,
            has_offscreen_renderer=sensors.has_offscreen_renderer,
            use_camera_obs=sensors.use_camera_observations,
            camera_depths=sensors.camera_depths,
            control_freq=descriptor.control_frequency_hz or CONTROL_FREQUENCY_HZ,
        )
        try:
            observation = reset_env_with_initial_arm_qpos(env)
            initialized = self.runtime_api.initialize_runtime(
                descriptor, env, observation, {"controller_config": controller_config}
            )
        except Exception:
            close = getattr(env, "close", None)
            if callable(close):
                close()
            raise
        if not isinstance(initialized, InitializedTaskRuntime):
            raise TypeError("task initialize_runtime() must return InitializedTaskRuntime")
        runtime = RoboCasaEpisodeRuntime(
            descriptor.episode,
            initialized,
            self.runtime_api,
            expert=expert,
        )
        runtime._attach_expert_runtime()
        return runtime


__all__ = ["RoboCasaEpisodeRuntime", "RoboCasaRuntimeFactory"]
