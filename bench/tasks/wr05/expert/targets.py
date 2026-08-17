"""WR-05 motion target compatibility surface (targets are resolved live)."""

from __future__ import annotations

from typing import Any


def resolve_motion_targets(episode: Any, *, operation: str | None = None) -> dict[str, Any]:
    del operation
    return dict(episode.facts.get("targets", ()))


__all__ = ["resolve_motion_targets"]
