"""MemER high-level memory policy composed with an OpenPI action policy."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import time
from typing import Any

from bench.policy.backends.openpi import OpenPIBackend
from bench.policy.protocol import (
    CanonicalObservation,
    InferenceRequest,
    InferenceResponse,
    PolicyContract,
)


class MemEROpenPIBackend:
    """Run MemER for subtask selection and OpenPI for low-level actions.

    The backend is intentionally local and stateful: MemER's episodic memory is
    reset with the policy protocol episode, while OpenPI keeps its own temporal
    state.  The wire contract remains the low-level OpenPI action contract.
    """

    def __init__(
        self,
        memer_policy: Any,
        low_level: OpenPIBackend,
        *,
        instruction: str,
        allowed_subtasks: tuple[str, ...] = (),
        server_id: str | None = None,
    ) -> None:
        if not instruction.strip():
            raise ValueError("MemER instruction must be non-empty")
        self.memer_policy = memer_policy
        self.low_level = low_level
        self.instruction = instruction
        self.allowed_subtasks = tuple(allowed_subtasks)
        self._allowed_subtask_set = frozenset(self.allowed_subtasks)
        self._subtask_indices = {
            label: index for index, label in enumerate(self.allowed_subtasks)
        }
        self._last_subtask: str | None = None
        self._active_episode: str | None = None
        self._contract = replace(
            low_level.contract,
            server_id=server_id or f"memer-openpi-{low_level.contract.server_id}",
            policy={
                "family": "memer_openpi",
                "name": "MemER high-level memory + OpenPI low-level action",
                "high_level": "memer",
                "low_level": dict(low_level.contract.policy),
                "subtasks": list(self.allowed_subtasks),
            },
        )

    @property
    def contract(self) -> PolicyContract:
        return self._contract

    def reset(self, episode_id: str) -> None:
        self._active_episode = episode_id
        self._last_subtask = None
        self.memer_policy.reset(instruction=self.instruction)
        self.low_level.reset(episode_id)

    def infer(self, request: InferenceRequest) -> InferenceResponse:
        if request.episode_id != self._active_episode:
            raise ValueError(
                f"Reset episode {request.episode_id!r} before MemER inference"
            )

        memer_observation = {
            "observation.images.robot0_agentview_left": request.observation.images[
                "agentview_left"
            ],
            "observation.images.robot0_agentview_right": request.observation.images[
                "agentview_right"
            ],
            "observation.images.robot0_eye_in_hand": request.observation.images[
                "eye_in_hand"
            ],
        }
        memer_start = time.monotonic()
        high_level_result = self.memer_policy.step(memer_observation)
        memer_ms = (time.monotonic() - memer_start) * 1000.0

        predicted = _normalize_subtask(high_level_result.current_subtask)
        if predicted and (
            not self._allowed_subtask_set or predicted in self._allowed_subtask_set
        ):
            subtask = predicted
            fallback = 0.0
        elif self._last_subtask is not None:
            subtask = self._last_subtask
            fallback = 1.0
        elif self.allowed_subtasks:
            # WA-01 has a deterministic first stage.  Keep the low-level prompt
            # inside its training vocabulary even if the first VLM output fails.
            subtask = self.allowed_subtasks[0]
            fallback = 1.0
        else:
            subtask = request.observation.prompt
            fallback = 1.0
        self._last_subtask = subtask

        low_level_request = InferenceRequest(
            request_id=request.request_id,
            episode_id=request.episode_id,
            step=request.step,
            observation=CanonicalObservation(
                task_id=request.observation.task_id,
                state=request.observation.state,
                images=request.observation.images,
                prompt=subtask,
                extra_modalities=request.observation.extra_modalities,
            ),
        )
        low_level_response = self.low_level.infer(low_level_request)
        timing = dict(low_level_response.timing)
        timing.update(
            {
                "memer_ms": memer_ms,
                "subtask_fallback": fallback,
                "subtask_parse_ok": float(bool(high_level_result.parse_ok)),
                "subtask_memory_count": float(
                    len(high_level_result.memory_indices_after)
                ),
                "subtask_index": float(self._subtask_indices.get(subtask, -1)),
            }
        )
        return InferenceResponse(
            request_id=low_level_response.request_id,
            episode_id=low_level_response.episode_id,
            actions=low_level_response.actions,
            timing=timing,
        )

    def close(self) -> None:
        self.low_level.close()
        self._active_episode = None


def load_allowed_subtasks(path: Path) -> tuple[str, ...]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    subtasks = payload.get("subtasks", {})
    if not isinstance(subtasks, dict):
        raise ValueError(f"subtasks must be an object in {path}")
    labels = tuple(
        _normalize_subtask(value.get("label"))
        for value in subtasks.values()
        if isinstance(value, dict) and value.get("label")
    )
    if not labels:
        raise ValueError(f"no subtask labels found in {path}")
    return labels


def _normalize_subtask(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


__all__ = ["MemEROpenPIBackend", "load_allowed_subtasks"]
