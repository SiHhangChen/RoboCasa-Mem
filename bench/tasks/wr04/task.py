"""Canonical composition root for WR-04."""

from bench.tasks.catalog import get_task_definition

TASK = get_task_definition("WR-04")

__all__ = ["TASK"]
