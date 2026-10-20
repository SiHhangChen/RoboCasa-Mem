"""Checkpoint identity helpers used by serving backends."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from bench.policy.protocol import CheckpointIdentity


_METADATA_NAMES = ("membench_policy.json", "policy_config.json", "config.json")


def read_checkpoint_metadata(checkpoint_dir: Path) -> dict[str, Any]:
    directory = checkpoint_dir.expanduser().resolve()
    for name in _METADATA_NAMES:
        path = directory / name
        if path.is_file():
            payload = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                raise ValueError(f"Checkpoint metadata must be a map: {path}")
            return payload
    return {}


def fingerprint_checkpoint(checkpoint_dir: Path) -> str:
    directory = checkpoint_dir.expanduser().resolve()
    if not directory.is_dir():
        raise FileNotFoundError(f"Checkpoint directory not found: {directory}")
    digest = hashlib.sha256()
    files = sorted(
        path
        for path in directory.rglob("*")
        if path.is_file() and not path.is_symlink()
    )
    if not files:
        raise ValueError(f"Checkpoint directory contains no files: {directory}")
    for path in files:
        relative = path.relative_to(directory).as_posix().encode()
        stat = path.stat()
        digest.update(relative)
        digest.update(b"\0")
        digest.update(str(stat.st_size).encode())
        digest.update(b"\0")
        if path.name in _METADATA_NAMES or stat.st_size <= 1024 * 1024:
            digest.update(path.read_bytes())
        else:
            with path.open("rb") as handle:
                digest.update(handle.read(64 * 1024))
                handle.seek(max(0, stat.st_size - 64 * 1024))
                digest.update(handle.read(64 * 1024))
    return f"sha256-manifest:{digest.hexdigest()}"


def resolve_checkpoint_identity(
    checkpoint_dir: Path,
    *,
    uri: str | None = None,
    step: int | None = None,
) -> CheckpointIdentity:
    directory = checkpoint_dir.expanduser().resolve()
    metadata = read_checkpoint_metadata(directory)
    resolved_step = step if step is not None else _metadata_step(metadata)
    return CheckpointIdentity(
        uri=uri or f"file://{directory}",
        step=resolved_step,
        fingerprint=fingerprint_checkpoint(directory),
    )


def _metadata_step(metadata: dict[str, Any]) -> int | None:
    value = metadata.get("step", metadata.get("training_step"))
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("Checkpoint metadata step must be an integer")
    return value
