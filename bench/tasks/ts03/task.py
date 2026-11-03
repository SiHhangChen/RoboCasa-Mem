"""Canonical composition root for TS03."""

from bench.tasks.catalog import get_task_definition

TASK = get_task_definition("TS-03")

__all__ = ["TASK"]

