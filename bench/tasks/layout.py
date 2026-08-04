"""Validation for the mandatory leaf-task package layout."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


TASK_PACKAGE_LAYOUT_VERSION = "membench.task_package/v1"

REQUIRED_PYTHON_FILES = (
    "__init__.py", "README.md", "task.py", "config.py", "episode.py",
    "environment.py", "initialization.py", "observation.py", "semantics.py", "prompts.py",
    "expert/__init__.py", "expert/spec.py", "expert/stations.py", "expert/targets.py", "expert/operations.py",
)
REQUIRED_RESOURCE_FILES = (
    "resources/task.yaml", "resources/sampling/regions.json",
    "resources/navigation/stations.json", "resources/navigation/routes.json",
    "resources/interactions/calibrations.json", "resources/scene/placements.json",
    "resources/assets/manifest.yaml",
)


def required_task_files() -> tuple[str, ...]:
    return REQUIRED_PYTHON_FILES + REQUIRED_RESOURCE_FILES


def validate_task_package(path: str | Path) -> list[str]:
    """Return missing paths; raise no imports and perform no simulator work."""

    root = Path(path)
    return [relative for relative in required_task_files() if not (root / relative).is_file()]


def validate_placeholder_resources(path: str | Path) -> None:
    """Validate the empty resource form used by tasks without a resource yet."""

    root = Path(path)
    for relative in REQUIRED_RESOURCE_FILES:
        source = root / relative
        if relative.endswith(".json"):
            payload = json.loads(source.read_text(encoding="utf-8"))
            if not isinstance(payload, dict) or "schema_version" not in payload or "entries" not in payload:
                raise ValueError(f"invalid resource schema: {source}")


__all__ = [
    "REQUIRED_PYTHON_FILES",
    "REQUIRED_RESOURCE_FILES",
    "TASK_PACKAGE_LAYOUT_VERSION",
    "required_task_files",
    "validate_placeholder_resources",
    "validate_task_package",
]
