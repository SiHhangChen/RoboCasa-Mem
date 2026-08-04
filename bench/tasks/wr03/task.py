"""Canonical composition root for WR-03."""

from bench.tasks.catalog import get_task_definition

TASK = get_task_definition("WR-03")

__all__ = ["TASK"]
