"""Canonical composition root for TS01."""

from bench.tasks.catalog import get_task_definition

TASK = get_task_definition("TS01")

__all__ = ["TASK"]

