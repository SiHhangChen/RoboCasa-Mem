"""Atomic, manifest-first episode artifact helpers."""

from __future__ import annotations

import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from bench.core.identity import canonical_hash
from bench.runtime.data_contract import NATIVE_EPISODE_SCHEMA


def atomic_json_commit(path: str | Path, payload: Any) -> Path:
    """Write JSON through a temporary sibling and atomically replace the target."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=f".{target.name}-", dir=target.parent) as tmp:
        staged = Path(tmp) / target.name
        staged.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
        os.replace(staged, target)
    return target


def artifact_manifest(*, task_id: str, episode_id: str, files: dict[str, str], provenance: dict[str, Any] | None = None) -> dict[str, Any]:
    manifest = {
        "schema_version": NATIVE_EPISODE_SCHEMA,
        "task_id": str(task_id),
        "episode_id": str(episode_id),
        "files": dict(sorted(files.items())),
        "provenance": dict(provenance or {}),
    }
    manifest["manifest_hash"] = canonical_hash(manifest)
    return manifest


__all__ = ["artifact_manifest", "atomic_json_commit"]
