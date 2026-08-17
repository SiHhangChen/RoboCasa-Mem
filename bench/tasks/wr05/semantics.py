"""Diagnostics boundary for WR-05."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def evaluate_task_state(runtime: Any, state: Any = None) -> Mapping[str, Any]:
    del state
    env = getattr(runtime, "env", None)
    if env is not None and hasattr(env, "task_diagnostics"):
        return dict(env.task_diagnostics())
    if env is not None and hasattr(env, "_check_success"):
        return {"success": bool(env._check_success())}
    return {}


__all__ = ["evaluate_task_state"]
