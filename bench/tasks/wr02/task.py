"""Canonical composition root for WR-02."""

from bench.tasks.catalog import get_task_definition

TASK = get_task_definition("WR-02")

__all__ = ["TASK"]
