"""Canonical composition root for TM01."""

from bench.tasks.catalog import get_task_definition

TASK = get_task_definition("TM-01")

__all__ = ["TASK"]

