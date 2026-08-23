"""Canonical composition root for WA-05."""

from bench.tasks.catalog import get_task_definition

TASK = get_task_definition("WA-05")

__all__ = ["TASK"]
