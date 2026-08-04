"""Task-state observation boundary for WR-03."""

from collections.abc import Mapping
from typing import Any


def observe_task_state(runtime: Any) -> Mapping[str, Any]:
    if isinstance(runtime, Mapping):
        return dict(runtime)
    observation = getattr(runtime, "observation", None)
    if isinstance(observation, Mapping):
        return dict(observation)
    env = getattr(runtime, "env", None)
    getter = getattr(env, "_get_observations", None)
    if callable(getter):
        value = getter(force_update=True)
        return dict(value) if isinstance(value, Mapping) else {}
    return {}


__all__ = ["observe_task_state"]

