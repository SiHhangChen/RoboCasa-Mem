"""Canonical composition root for WX-02."""

from bench.tasks.catalog import get_task_definition

TASK = get_task_definition("WX-02")

__all__ = ["TASK"]
