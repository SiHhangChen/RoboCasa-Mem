"""WR-07 package identity and composition root."""

from bench.tasks.catalog import get_task_definition

TASK = get_task_definition("WR-07")

__all__ = ["TASK"]
