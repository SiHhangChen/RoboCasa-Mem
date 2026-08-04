"""Canonical composition root for TM03."""

from bench.tasks.catalog import get_task_definition

TASK = get_task_definition("TM03")

__all__ = ["TASK"]

