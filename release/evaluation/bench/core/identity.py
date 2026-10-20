"""Deterministic identities for tasks, resources, code and releases."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in sorted(value.items(), key=lambda x: str(x[0]))}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, set | frozenset):
        return sorted(_jsonable(item) for item in value)
    if hasattr(value, "tolist"):
        return _jsonable(value.tolist())
    raise TypeError(f"value is not JSON serializable: {type(value).__name__}")


def canonical_json(value: Any) -> str:
    """Return the canonical JSON representation used for hashes."""

    return json.dumps(
        _jsonable(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )


def canonical_hash(value: Any) -> str:
    """Hash a JSON-like value with a stable, explicit prefix."""

    digest = hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


def file_sha256(path: str | Path) -> str:
    """Hash one file without following directories or implicit globs."""

    source = Path(path).expanduser().resolve()
    return f"sha256:{hashlib.sha256(source.read_bytes()).hexdigest()}"


def release_identity(*, release_id: str, contexts: Any, code_revision: str = "unknown") -> str:
    """Build the immutable identity of one benchmark release."""

    return canonical_hash(
        {
            "schema_version": "membench.release/v1",
            "release_id": str(release_id),
            "code_revision": str(code_revision),
            "contexts": contexts,
        }
    )


__all__ = ["canonical_hash", "canonical_json", "file_sha256", "release_identity"]
