"""WR-06 package identity and composition root."""

from __future__ import annotations

from bench.tasks.catalog import get_task_definition

TASK = get_task_definition("WR-06")

__all__ = ["TASK"]
