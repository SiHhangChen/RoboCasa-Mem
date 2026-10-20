"""Success, failure, and diagnostics boundary for TM02."""

from collections.abc import Mapping
from typing import Any


def evaluate_task_state(runtime: Any, state: Mapping[str, Any] | None = None) -> Mapping[str, Any]:
    del state
    for name in ("task_diagnostics",):
        method = getattr(runtime, name, None)
        if callable(method):
            value = method()
            return dict(value) if isinstance(value, Mapping) else {"success": bool(value)}
    env = getattr(runtime, "env", None)
    for name in ("task_diagnostics",):
        method = getattr(env, name, None)
        if callable(method):
            value = method()
            return dict(value) if isinstance(value, Mapping) else {"success": bool(value)}
    check = getattr(env, "_check_success", None)
    return {"success": bool(check())} if callable(check) else {"success": False, "reason": "runtime_not_initialized"}


__all__ = ["evaluate_task_state"]
