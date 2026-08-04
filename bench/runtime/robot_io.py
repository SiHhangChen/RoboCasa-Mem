"""Standard observe/step adapters for RoboCasa robot environments."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def observe(env: Any) -> Mapping[str, Any]:
    getter = getattr(env, "_get_observations", None)
    if not callable(getter):
        raise AttributeError("environment has no _get_observations()")
    try:
        value = getter(force_update=True)
    except TypeError:
        value = getter()
    if not isinstance(value, Mapping):
        raise TypeError("environment observation must be a mapping")
    return value


def step(env: Any, action: Any) -> tuple[Mapping[str, Any], float, bool, dict[str, Any]]:
    result = env.step(action)
    if len(result) == 5:
        observation, reward, terminated, truncated, info = result
        done = bool(terminated or truncated)
    elif len(result) == 4:
        observation, reward, done, info = result
    else:
        raise ValueError(f"unsupported environment step result length: {len(result)}")
    if not isinstance(observation, Mapping):
        raise TypeError("environment observation must be a mapping")
    return observation, float(reward), bool(done), dict(info or {})


__all__ = ["observe", "step"]
