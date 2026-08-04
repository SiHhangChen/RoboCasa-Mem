"""Task-local Station and route resolution for TS01."""

from typing import Any


def resolve_station_targets(episode: Any, *, purpose: str | None = None):
    del purpose
    return dict(getattr(episode, "facts", {}) or {}).get("stations", ())


__all__ = ["resolve_station_targets"]

