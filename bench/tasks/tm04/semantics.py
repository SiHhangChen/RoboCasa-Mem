"""Success and diagnostics boundary for TM04."""

from collections.abc import Mapping
from typing import Any


def evaluate_task_state(runtime: Any, state: Mapping[str, Any] | None = None) -> Mapping[str, Any]:
    del state
    for owner in (runtime, getattr(runtime, "env", None)):
        method = getattr(owner, "task_diagnostics", None)
        if callable(method):
            value = method()
            return dict(value) if isinstance(value, Mapping) else {"success": bool(value)}
    return {"success": False, "reason": "tm04_physical_runtime_not_initialized"}


__all__ = ["evaluate_task_state"]
