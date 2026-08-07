"""Canonical composition root for WA-04."""

from bench.tasks.catalog import get_task_definition

TASK = get_task_definition("WA-04")

__all__ = ["TASK"]
