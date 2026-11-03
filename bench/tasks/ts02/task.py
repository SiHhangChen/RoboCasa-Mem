"""Canonical composition root for TS02."""

from bench.tasks.catalog import get_task_definition

TASK = get_task_definition("TS-02")

__all__ = ["TASK"]

