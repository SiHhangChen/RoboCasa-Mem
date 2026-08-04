"""Canonical composition root for WR-01."""

from bench.tasks.catalog import get_task_definition

TASK = get_task_definition("WR-01")

__all__ = ["TASK"]
