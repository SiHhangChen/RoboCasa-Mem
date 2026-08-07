"""Canonical composition root for WA-03."""

from bench.tasks.catalog import get_task_definition

TASK = get_task_definition("WA-03")

__all__ = ["TASK"]
