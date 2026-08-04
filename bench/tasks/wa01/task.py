"""Canonical composition root for WA-01."""

from bench.tasks.catalog import get_task_definition

TASK = get_task_definition("WA-01")

__all__ = ["TASK"]
