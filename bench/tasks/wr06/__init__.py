"""Independent single-bottle plate-staging shelf-memory task."""

from bench.tasks.wr06.task import TASK

TASK_ID = TASK.task_id
TASK_VERSION = TASK.version
ENV_NAME = TASK.env_name

__all__ = ["ENV_NAME", "TASK", "TASK_ID", "TASK_VERSION"]
