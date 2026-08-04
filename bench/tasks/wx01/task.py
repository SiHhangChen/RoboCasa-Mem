"""Canonical composition root for WX-01."""

from bench.tasks.catalog import get_task_definition

TASK = get_task_definition("WX-01")

__all__ = ["TASK"]
