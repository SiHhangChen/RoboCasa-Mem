"""Expert target compatibility surface for WR-07."""

from __future__ import annotations

from typing import Any


def resolve_station_targets(episode: Any, *, purpose: str | None = None) -> dict[str, Any]:
    del purpose
    facts = dict(getattr(episode, "facts", {}) or {})
    return {
        "source_slot": facts.get("source_slot_by_item", {}).get("milk_carton"),
        "source_color": facts.get("source_color_by_item", {}).get("milk_carton"),
        "target_slot": facts.get("target_slot_by_color", {}).get(
            facts.get("source_color_by_item", {}).get("milk_carton")
        ),
    }


__all__ = ["resolve_station_targets"]
