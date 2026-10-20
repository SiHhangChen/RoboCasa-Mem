"""Interface implemented by every remote policy backend."""

from __future__ import annotations

from typing import Protocol

from bench.policy.protocol import InferenceRequest, InferenceResponse, PolicyContract


class PolicyBackend(Protocol):
    @property
    def contract(self) -> PolicyContract: ...

    def reset(self, episode_id: str) -> None: ...

    def infer(self, request: InferenceRequest) -> InferenceResponse: ...

    def close(self) -> None: ...
