"""Canonical composition root for TM02."""

from bench.tasks.catalog import get_task_definition

TASK = get_task_definition("TM-02")

__all__ = ["TASK"]

