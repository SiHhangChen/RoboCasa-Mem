"""Task-local MotionSegment target resolution for WA-05."""

from typing import Any


def resolve_motion_targets(episode: Any, *, operation: str | None = None):
    del operation
    return dict(getattr(episode, "facts", {}) or {}).get("targets", ())


__all__ = ["resolve_motion_targets"]
