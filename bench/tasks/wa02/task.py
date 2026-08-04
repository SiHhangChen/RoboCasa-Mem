"""Canonical composition root for WA-02."""

from bench.tasks.catalog import get_task_definition

TASK = get_task_definition("WA-02")

__all__ = ["TASK"]
