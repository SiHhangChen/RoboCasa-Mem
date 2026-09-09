"""Canonical expert recording over one shared :class:`EpisodeRuntime`."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from bench.execution.contracts import ExpertPlan, NavigateStage, StageResult
from bench.execution.runner import ExpertRunResult, StageRunner
from bench.recording.config import RecordingConfig
from bench.runtime.contracts import EpisodeRuntime
from bench.runtime.data_contract import ACTION_SCHEMA, OBSERVATION_PROFILE
from bench.tasks.contracts import ResolvedEpisode


def build_recording_config(
    runtime: EpisodeRuntime,
    *,
    output_dir: str | Path,
    cameras: list[str] | tuple[str, ...] | None = None,
    fps: int | None = None,
    save_navigation_snapshots: bool = False,
    record_depth: bool | None = None,
) -> RecordingConfig:
    """Resolve recorder settings from the same canonical runtime descriptor."""

    initialized = getattr(runtime, "initialized", None)
    descriptor = getattr(initialized, "descriptor", None)
    if descriptor is None:
        raise TypeError("canonical recording requires runtime descriptor access")
    sensors = descriptor.sensors
    if str(descriptor.episode.action_schema) != ACTION_SCHEMA:
        raise ValueError(
            f"episode action schema must be {ACTION_SCHEMA}, got {descriptor.episode.action_schema!r}"
        )
    if sensors.profile_id != OBSERVATION_PROFILE:
        raise ValueError(
            f"recording requires observation profile {OBSERVATION_PROFILE}, got {sensors.profile_id!r}"
        )
    requested_fps = descriptor.control_frequency_hz if fps is None else int(fps)
    if requested_fps != descriptor.control_frequency_hz:
        raise ValueError(
            f"recording fps must match {descriptor.control_frequency_hz} Hz runtime"
        )
    camera_names = _recording_camera_names(sensors, cameras)
    recording = dict(getattr(descriptor.config, "recording", {}) or {})
    requested_depth = bool(recording.get("record_depth", True)) if record_depth is None else bool(record_depth)
    if requested_depth and not sensors.camera_depths:
        raise ValueError("record_depth requires a depth-enabled observation profile")
    return RecordingConfig(
        output_dir=Path(output_dir),
        cameras=camera_names,
        camera_width=sensors.width,
        camera_height=sensors.height,
        fps=requested_fps,
        save_navigation_snapshots=bool(save_navigation_snapshots),
        record_depth=requested_depth,
        depth_dtype=str(recording.get("depth_dtype", "uint16")),
        depth_scale=float(recording.get("depth_scale", 0.001)),
        env_meta=_runtime_env_meta(descriptor, initialized),
        env_name=descriptor.environment.env_name,
    )


class RecordingRunner:
    """Record a canonical ExpertPlan against an already initialized runtime."""

    def __init__(
        self,
        *,
        runtime: EpisodeRuntime,
        runner: StageRunner,
        config: RecordingConfig,
    ) -> None:
        self._runtime = runtime
        self._runner = runner
        self._config = config

    def run(self, plan: ExpertPlan) -> tuple[ExpertRunResult, Path]:
        episode = self._runtime.episode
        if plan.task_id != episode.task_id:
            raise ValueError(
                f"expert plan {plan.task_id!r} does not match episode {episode.task_id!r}"
            )
        started_at = time.time()
        env = _runtime_env(self._runtime)

        from bench.recording.writers import EpisodeRecorder

        recorder = EpisodeRecorder(
            env,
            output_dir=self._config.output_dir,
            cameras=self._config.cameras,
            camera_width=self._config.camera_width,
            camera_height=self._config.camera_height,
            fps=self._config.fps,
            record_depth=self._config.record_depth,
            depth_dtype=self._config.depth_dtype,
            depth_scale=self._config.depth_scale,
            env_meta=self._config.env_meta,
            env_name=self._config.env_name,
        )
        recorder.install()
        recorder.record_initial_observation(dict(self._runtime.observe()))

        def on_stage_end(index: int, stage: object, result: StageResult) -> None:
            observation = dict(self._runtime.observe())
            recorder.record_stage_observation(observation)
            if not self._config.save_navigation_snapshots or not isinstance(
                stage, NavigateStage
            ):
                return
            snapshots = recorder.write_stage_snapshot(
                stage_index=index,
                stage_kind="navigate",
                label=stage.name or stage.target.station_id,
                obs=observation,
            )
            if snapshots:
                result.details["navigation_snapshots"] = snapshots

        try:
            expert_result = self._runner.run(plan, on_stage_end=on_stage_end)
        finally:
            recorder.close()

        diagnostics = self._runtime.diagnostics()
        recorder.set_final_diagnostics(_diagnostics_dict(diagnostics))
        semantic_required = bool(
            getattr(getattr(self._runtime, "initialized", None), "descriptor", None)
            and self._runtime.initialized.descriptor.execute
        )
        accepted = bool(
            expert_result.success
            and (diagnostics.success if semantic_required else True)
        )
        result = ExpertRunResult(
            task_id=expert_result.task_id,
            success=accepted,
            stages=expert_result.stages,
            details={
                **dict(expert_result.details),
                "expert_stages_success": expert_result.success,
                "task_diagnostics": _diagnostics_dict(diagnostics),
            },
        )
        if result.success:
            recorder.finalize_terminal(
                terminated=True,
                truncated=False,
                success=True,
                reason="task_success",
            )
        elif diagnostics.terminal:
            recorder.finalize_terminal(
                terminated=True,
                truncated=False,
                success=False,
                reason=str(diagnostics.failure_code or "task_failure"),
            )
        elif len(recorder.records) >= int(episode.max_horizon):
            recorder.finalize_terminal(
                terminated=False,
                truncated=True,
                success=False,
                reason="horizon",
            )
        else:
            recorder.finalize_terminal(
                terminated=False,
                truncated=True,
                success=False,
                reason=str(diagnostics.failure_code or "expert_failure"),
            )
        stage_events = [_stage_event(item) for item in result.stages]
        planner_summary = {
            "pass": result.success,
            "task_id": episode.task_id,
            "episode_identity": episode.identity,
            "expert_plan_version": plan.version,
            "episode_result": {
                "success": result.success,
                "stage_count": len(result.stages),
                "stages": [item.success for item in result.stages],
            },
            "rule_policy": {"operations": stage_events},
            "rollout": {"stages": stage_events},
            "final_diagnostics": _diagnostics_dict(diagnostics),
        }
        recorder.write_episode(
            metadata=_episode_metadata(
                episode,
                plan=plan,
                runtime=self._runtime,
                started_at=started_at,
            ),
            planner_summary=planner_summary,
            accepted=result.success,
        )
        return result, self._config.output_dir


def _runtime_env(runtime: EpisodeRuntime) -> Any:
    env = getattr(runtime, "env", None)
    if env is None:
        raise TypeError("recording requires an EpisodeRuntime exposing its robot environment")
    return env


def _recording_camera_names(sensors: Any, requested: list[str] | tuple[str, ...] | None) -> list[str]:
    if requested is None:
        return list(sensors.camera_names)
    values = tuple(str(value) for value in requested)
    if values == tuple(sensors.camera_aliases):
        return list(sensors.camera_names)
    if values != tuple(sensors.camera_names):
        raise ValueError("cameras cannot override the selected observation profile")
    return list(values)


def _runtime_env_meta(descriptor: Any, initialized: Any) -> dict[str, Any]:
    from bench.env.robocasa_runtime import build_env_meta

    environment = descriptor.environment
    sensors = descriptor.sensors
    controller_config = dict(initialized.metadata.get("controller_config", {}) or {})
    return build_env_meta(
        env_name=environment.env_name,
        controller_configs=controller_config,
        camera_names=sensors.camera_names,
        camera_width=sensors.width,
        camera_height=sensors.height,
        split=environment.env_split,
        layout_id=environment.layout_id,
        style_id=environment.style_id,
        env_kwargs=dict(environment.env_kwargs),
        camera_depths=sensors.camera_depths,
        control_freq=descriptor.control_frequency_hz,
    )


def _stage_event(result: StageResult) -> dict[str, Any]:
    return {
        "stage_index": result.index,
        "stage_kind": result.kind.value,
        "success": result.success,
        "failure_code": result.failure_code,
        "details": dict(result.details),
    }


def _diagnostics_dict(value: Any) -> dict[str, Any]:
    return {
        "success": bool(value.success),
        "terminal": bool(value.terminal),
        "failure_code": value.failure_code,
        **dict(value.details),
    }


def _episode_metadata(
    episode: ResolvedEpisode,
    *,
    plan: ExpertPlan,
    runtime: EpisodeRuntime,
    started_at: float,
) -> dict[str, Any]:
    initialized = getattr(runtime, "initialized", None)
    descriptor = getattr(initialized, "descriptor", None)
    prompt = getattr(descriptor, "prompt", None)
    initial_prompt = getattr(prompt, "initial_prompt", None)
    policy_prompt = getattr(prompt, "policy_prompt", None)
    return {
        "task_id": episode.task_id,
        "task_version": episode.task_version,
        "episode_identity": episode.identity,
        "seed": episode.seed,
        "variant": episode.variant,
        "split": episode.split,
        "env_split": episode.env_split,
        "max_horizon": episode.max_horizon,
        "started_at_unix": started_at,
        "resolved_episode": episode.to_dict(),
        "expert_plan": {
            "version": plan.version,
            "stage_count": len(plan.stages),
            "metadata": dict(plan.metadata),
        },
        "initialization": dict(getattr(initialized, "initialization", {}) or {}),
        "release_facts": dict(getattr(descriptor, "release_facts", {}) or {}),
        "initial_prompt": None if initial_prompt is None else str(initial_prompt),
        "policy_prompt": None if policy_prompt is None else str(policy_prompt),
    }


__all__ = ["RecordingRunner", "build_recording_config"]
