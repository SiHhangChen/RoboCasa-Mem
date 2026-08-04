"""Canonical composition root for TS04."""

from bench.tasks.catalog import get_task_definition

TASK = get_task_definition("TS04")

__all__ = ["TASK"]

