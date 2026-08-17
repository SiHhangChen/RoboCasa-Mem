"""Observation and task-state compatibility surface for WR-06."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def observe_task_state(runtime: Any) -> Mapping[str, Any]:
    observation = getattr(runtime, "observation", None)
    if observation is not None:
        return dict(observation)
    env = getattr(runtime, "env", None)
    if env is not None and hasattr(env, "_get_observations"):
        return dict(env._get_observations(force_update=True))
    return {}


__all__ = ["observe_task_state"]
