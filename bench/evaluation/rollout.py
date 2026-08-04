"""Policy-neutral closed-loop rollout runner."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
import time
import traceback
from typing import Any

import numpy as np

from bench.evaluation.action import ActionChunkQueue
from bench.evaluation.adapter import EpisodeRuntime, EvaluationAdapter, preflight_adapter
from bench.evaluation.artifacts import ArtifactWriter, EpisodeArtifactSession
from bench.evaluation.client import RemotePolicyClient
from bench.evaluation.suite import EvalSuiteConfig


@dataclass
class EpisodeResult:
    seed: int
    episode_id: str
    task_id: str
    suite_version: str
    status: str
    success: bool
    success_step: int | None
    env_steps: int
    policy_calls: int
    reward_sum: float
    failure_reason: str | None = None
    error: str | None = None
    diagnostics: Mapping[str, Any] = field(default_factory=dict)
    inference_latency_ms: list[float] = field(default_factory=list)
    action_mean: list[float] = field(default_factory=list)
    action_std: list[float] = field(default_factory=list)
    action_min: list[float] = field(default_factory=list)
    action_max: list[float] = field(default_factory=list)
    policy_actions_returned: int = 0
    policy_actions_consumed: int = 0
    duration_s: float = 0.0
    episode_metadata: Mapping[str, Any] = field(default_factory=dict)


class RolloutRunner:
    def __init__(
        self,
        *,
        client: RemotePolicyClient,
        suite: EvalSuiteConfig,
        artifacts: ArtifactWriter | None = None,
    ) -> None:
        self.client = client
        self.suite = suite
        self.artifacts = artifacts
        if suite.rollout.replan_steps > client.contract.action.horizon:
            raise ValueError(
                f"replan_steps={suite.rollout.replan_steps} exceeds policy horizon "
                f"{client.contract.action.horizon}"
            )

    def run_episode(
        self,
        adapter: EvaluationAdapter,
        seed: int,
    ) -> EpisodeResult:
        preflight_adapter(
            adapter,
            suite=self.suite,
            policy_contract=self.client.contract,
        )
        started_at = time.monotonic()
        runtime: EpisodeRuntime | None = None
        session = self._start_artifact_session(seed)
        actions: list[np.ndarray] = []
        latencies: list[float] = []
        env_steps = 0
        policy_calls = 0
        reward_sum = 0.0
        success = False
        success_step: int | None = None
        failure_reason: str | None = None
        policy_actions_returned = 0
        policy_actions_consumed = 0
        episode_id = f"{self.suite.task_id.lower()}-seed-{seed:06d}"
        result: EpisodeResult | None = None
        interrupted: KeyboardInterrupt | None = None
        try:
            runtime = adapter.create_episode(
                seed=seed,
                policy_contract=self.client.contract,
                suite=self.suite,
            )
            episode_id = runtime.episode_id
            self.client.reset(episode_id)
            queue = ActionChunkQueue()
            while env_steps < self.suite.rollout.max_steps:
                if len(queue) == 0:
                    observation = adapter.build_observation(runtime)
                    start = time.monotonic()
                    response = self.client.infer(
                        observation,
                        episode_id=episode_id,
                        step=env_steps,
                    )
                    client_latency_ms = (time.monotonic() - start) * 1000.0
                    latencies.append(client_latency_ms)
                    policy_calls += 1
                    policy_actions_returned += int(response.actions.shape[0])
                    queued_count = queue.push(
                        response.actions,
                        limit=self.suite.rollout.replan_steps,
                    )
                    if session is not None:
                        session.append_request_metric(
                            {
                                "request_id": response.request_id,
                                "step": env_steps,
                                "client_latency_ms": client_latency_ms,
                                "returned_action_count": int(response.actions.shape[0]),
                                "queued_action_count": int(queued_count),
                                "server_timing": dict(response.timing),
                            }
                        )
                action = queue.pop()
                policy_actions_consumed += 1
                for _ in range(self.suite.rollout.action_repeat):
                    if env_steps >= self.suite.rollout.max_steps:
                        break
                    shared_runtime = runtime.metadata.get("shared_runtime")
                    if shared_runtime is not None:
                        transition = shared_runtime.step(action)
                        observation = transition.observation
                        reward = transition.reward
                        done = transition.terminal
                        info = transition.info
                    else:
                        step_result = runtime.env.step(action)
                        observation, reward, done, info = _normalize_env_step(step_result)
                    runtime.observation = observation
                    env_steps += 1
                    reward_sum += reward
                    actions.append(action.copy())
                    if session is not None:
                        session.append_action(action)
                        if self.suite.artifacts.record_video:
                            capture_frames = getattr(adapter, "artifact_frames", None)
                            if callable(capture_frames):
                                frames = capture_frames(
                                    runtime,
                                    self.suite.artifacts.video_cameras,
                                )
                            else:
                                frames = adapter.build_observation(runtime).images
                            session.append_frames(frames)
                    if adapter.check_success(runtime, info):
                        success = True
                        if success_step is None:
                            success_step = env_steps
                        if self.suite.rollout.stop_on_success:
                            break
                    if done and not self.suite.rollout.ignore_done:
                        failure_reason = "env_done"
                        break
                if success and self.suite.rollout.stop_on_success:
                    break
                if failure_reason == "env_done":
                    break
            if not success and failure_reason is None:
                failure_reason = "max_steps"
            result = self._make_result(
                seed=seed,
                episode_id=episode_id,
                status="success" if success else "failure",
                success=success,
                success_step=success_step,
                env_steps=env_steps,
                policy_calls=policy_calls,
                reward_sum=reward_sum,
                failure_reason=failure_reason,
                error=None,
                diagnostics=adapter.diagnostics(runtime),
                latencies=latencies,
                actions=actions,
                policy_actions_returned=policy_actions_returned,
                policy_actions_consumed=policy_actions_consumed,
                episode_metadata=runtime.result_metadata,
            )
        except KeyboardInterrupt as exc:
            interrupted = exc
            result = self._make_result(
                seed=seed,
                episode_id=episode_id,
                status="cancelled",
                success=False,
                success_step=success_step,
                env_steps=env_steps,
                policy_calls=policy_calls,
                reward_sum=reward_sum,
                failure_reason=None,
                error="KeyboardInterrupt",
                diagnostics=self._safe_diagnostics(adapter, runtime),
                latencies=latencies,
                actions=actions,
                policy_actions_returned=policy_actions_returned,
                policy_actions_consumed=policy_actions_consumed,
                episode_metadata=(
                    runtime.result_metadata if runtime is not None else {}
                ),
            )
        except Exception as exc:
            result = self._make_result(
                seed=seed,
                episode_id=episode_id,
                status="error",
                success=False,
                success_step=success_step,
                env_steps=env_steps,
                policy_calls=policy_calls,
                reward_sum=reward_sum,
                failure_reason=None,
                error="".join(
                    traceback.format_exception_only(type(exc), exc)
                ).strip(),
                diagnostics=self._safe_diagnostics(adapter, runtime),
                latencies=latencies,
                actions=actions,
                policy_actions_returned=policy_actions_returned,
                policy_actions_consumed=policy_actions_consumed,
                episode_metadata=(
                    runtime.result_metadata if runtime is not None else {}
                ),
            )
        finally:
            if runtime is not None:
                try:
                    adapter.close(runtime)
                except Exception as exc:
                    close_error = f"adapter.close:{type(exc).__name__}: {exc}"
                    if result is None:
                        result = self._make_result(
                            seed=seed,
                            episode_id=episode_id,
                            status="error",
                            success=False,
                            success_step=success_step,
                            env_steps=env_steps,
                            policy_calls=policy_calls,
                            reward_sum=reward_sum,
                            failure_reason=None,
                            error=close_error,
                            diagnostics={},
                            latencies=latencies,
                            actions=actions,
                            policy_actions_returned=policy_actions_returned,
                            policy_actions_consumed=policy_actions_consumed,
                            episode_metadata=runtime.result_metadata,
                        )
                    else:
                        result.status = "error"
                        result.success = False
                        result.failure_reason = None
                        result.error = (
                            close_error
                            if result.error is None
                            else f"{result.error}; {close_error}"
                        )
        assert result is not None
        result.duration_s = time.monotonic() - started_at
        if session is not None:
            try:
                session.finish(asdict(result))
            except BaseException:
                session.abort()
                raise
        if self.artifacts is not None:
            self.artifacts.write_summary()
        if interrupted is not None:
            raise interrupted
        return result

    def run_suite(
        self,
        adapter: EvaluationAdapter,
        *,
        resume: bool = True,
        retry_errors: bool = False,
    ) -> list[EpisodeResult]:
        completed = (
            self.artifacts.completed_seeds(retry_errors=retry_errors)
            if resume and self.artifacts is not None
            else set()
        )
        results = [
            self.run_episode(adapter, seed)
            for seed in self.suite.seeds
            if seed not in completed
        ]
        if self.artifacts is not None:
            self.artifacts.write_summary()
        return results

    def _start_artifact_session(self, seed: int) -> EpisodeArtifactSession | None:
        if self.artifacts is None:
            return None
        return self.artifacts.start_episode(
            seed=seed,
            record_video=self.suite.artifacts.record_video,
            video_cameras=self.suite.artifacts.video_cameras,
            video_fps=self.suite.artifacts.video_fps,
        )

    def _make_result(
        self,
        *,
        seed: int,
        episode_id: str,
        status: str,
        success: bool,
        success_step: int | None,
        env_steps: int,
        policy_calls: int,
        reward_sum: float,
        failure_reason: str | None,
        error: str | None,
        diagnostics: Mapping[str, Any],
        latencies: list[float],
        actions: list[np.ndarray],
        policy_actions_returned: int,
        policy_actions_consumed: int,
        episode_metadata: Mapping[str, Any],
    ) -> EpisodeResult:
        if actions:
            array = np.stack(actions)
            action_mean = array.mean(axis=0).tolist()
            action_std = array.std(axis=0).tolist()
            action_min = array.min(axis=0).tolist()
            action_max = array.max(axis=0).tolist()
        else:
            action_mean = action_std = action_min = action_max = []
        return EpisodeResult(
            seed=seed,
            episode_id=episode_id,
            task_id=self.suite.task_id,
            suite_version=self.suite.suite_version,
            status=status,
            success=success,
            success_step=success_step,
            env_steps=env_steps,
            policy_calls=policy_calls,
            reward_sum=reward_sum,
            failure_reason=failure_reason,
            error=error,
            diagnostics=dict(diagnostics),
            inference_latency_ms=latencies,
            action_mean=action_mean,
            action_std=action_std,
            action_min=action_min,
            action_max=action_max,
            policy_actions_returned=int(policy_actions_returned),
            policy_actions_consumed=int(policy_actions_consumed),
            episode_metadata=dict(episode_metadata),
        )

    @staticmethod
    def _safe_diagnostics(
        adapter: EvaluationAdapter,
        runtime: EpisodeRuntime | None,
    ) -> dict[str, Any]:
        if runtime is None:
            return {}
        try:
            return dict(adapter.diagnostics(runtime))
        except Exception as exc:
            return {"diagnostics_error": f"{type(exc).__name__}: {exc}"}


def _normalize_env_step(
    value: Any,
) -> tuple[Mapping[str, Any], float, bool, Mapping[str, Any]]:
    if not isinstance(value, tuple):
        raise TypeError("env.step() must return a tuple")
    if len(value) == 4:
        observation, reward, done, info = value
    elif len(value) == 5:
        observation, reward, terminated, truncated, info = value
        done = bool(terminated or truncated)
    else:
        raise ValueError(f"env.step() returned {len(value)} values")
    if not isinstance(observation, Mapping) or not isinstance(info, Mapping):
        raise TypeError("env.step() observation and info must be mappings")
    return observation, float(reward), bool(done), info
