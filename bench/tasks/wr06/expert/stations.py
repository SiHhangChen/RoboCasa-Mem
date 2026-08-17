"""WR-06 station target compatibility surface (navigation facts are live)."""

from __future__ import annotations

from typing import Any


def resolve_station_targets(episode: Any, *, purpose: str | None = None) -> dict[str, Any]:
    del purpose
    return dict(episode.facts.get("stations", ()))


__all__ = ["resolve_station_targets"]
