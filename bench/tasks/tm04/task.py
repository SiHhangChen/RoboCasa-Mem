"""Canonical composition root for TM04."""

from bench.tasks.catalog import get_task_definition

TASK = get_task_definition("TM04")

__all__ = ["TASK"]
