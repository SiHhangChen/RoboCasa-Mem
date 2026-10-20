"""Cross-cutting errors and failure categories.

The core package deliberately knows nothing about RoboCasa, a task id, or a
policy implementation.  Product-specific code can attach structured details
to these errors without changing the shared vocabulary.
"""

from __future__ import annotations

from enum import Enum
from typing import Any


class FailureCode(str, Enum):
    """Stable failure reasons written to recording and evaluation artifacts."""

    INVALID_REQUEST = "invalid_request"
    RESOLUTION_ERROR = "resolution_error"
    INITIALIZATION_ERROR = "initialization_error"
    OBSERVATION_ERROR = "observation_error"
    ACTION_ERROR = "action_error"
    STAGE_ERROR = "stage_error"
    TIMEOUT = "timeout"
    ENVIRONMENT_ERROR = "environment_error"
    TASK_FAILURE = "task_failure"
    INTERNAL_ERROR = "internal_error"


class BenchmarkError(RuntimeError):
    """An error that is safe to serialize into a benchmark artifact."""

    def __init__(
        self,
        message: str,
        *,
        code: FailureCode = FailureCode.INTERNAL_ERROR,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = FailureCode(code)
        self.details = dict(details or {})


__all__ = ["BenchmarkError", "FailureCode"]
