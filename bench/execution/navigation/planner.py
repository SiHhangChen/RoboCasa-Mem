"""Simple route planner over already resolved waypoints."""

from __future__ import annotations

from collections.abc import Iterable

from .contracts import PathResult, Waypoint


def plan_path(route: Iterable[Waypoint]) -> PathResult:
    waypoints = tuple(route)
    if not waypoints:
        return PathResult(False, reason="empty_route")
    return PathResult(True, waypoints=waypoints)


__all__ = ["plan_path"]
