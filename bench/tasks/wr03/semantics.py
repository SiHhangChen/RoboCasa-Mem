"""Success, failure, and diagnostics boundary for WR-03."""

from collections.abc import Mapping
from typing import Any

from bench.tasks.wr03.evaluation import (
    TASK_EVALUATOR_METADATA_KEY,
    WR03Evaluator,
    diagnostics_for_runtime,
)


def evaluate_task_state(runtime: Any, state: Mapping[str, Any] | None = None) -> Mapping[str, Any]:
    del state
    initialized = getattr(runtime, "initialized", None) or runtime
    metadata = getattr(initialized, "metadata", {}) or {}
    env = getattr(initialized, "env", None) or getattr(runtime, "env", None)
    evaluator = metadata.get(TASK_EVALUATOR_METADATA_KEY) or getattr(
        env,
        "_wr03_evaluator",
        None,
    )
    if isinstance(evaluator, WR03Evaluator):
        return diagnostics_for_runtime(runtime)
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
