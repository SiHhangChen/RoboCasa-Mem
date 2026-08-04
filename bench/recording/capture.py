"""Step-aligned native capture buffer."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class CaptureBuffer:
    actions: list[Any] = field(default_factory=list)
    observations: list[Any] = field(default_factory=list)
    timestamps: list[float] = field(default_factory=list)
    stage_events: list[dict[str, Any]] = field(default_factory=list)

    def append(self, *, action: Any, observation: Any, timestamp: float) -> None:
        self.actions.append(action)
        self.observations.append(observation)
        self.timestamps.append(float(timestamp))

    def append_stage_event(self, event: dict[str, Any]) -> None:
        self.stage_events.append(dict(event))

    def fields(self) -> dict[str, list[Any]]:
        return {
            "actions": self.actions,
            "observations": self.observations,
            "timestamps": self.timestamps,
        }


__all__ = ["CaptureBuffer"]
