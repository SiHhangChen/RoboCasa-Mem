"""Navigation data contracts without task semantics."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class StationTarget:
    station_id: str
    pose: tuple[float, float, float] | None = None
    route: tuple[tuple[float, float], ...] = ()
    # These are execution hints, not task semantics.  Leaf expert modules may
    # use them to select a calibrated station entry or carrying constraint;
    # generic navigation only forwards them to the station resource.
    purpose: str = ""
    holding: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Waypoint:
    x: float
    y: float
    yaw: float = 0.0


@dataclass(frozen=True, slots=True)
class PathResult:
    success: bool
    waypoints: tuple[Waypoint, ...] = ()
    reason: str | None = None


@dataclass(slots=True)
class Pose:
    """A mobile-base pose selected from task-owned station resources."""

    position: tuple[float, float, float]
    quaternion: tuple[float, float, float, float] | None = None
    torso_height: float | None = None

    @classmethod
    def xyz(cls, x: float, y: float, z: float) -> "Pose":
        return cls(position=(float(x), float(y), float(z)))


@dataclass(slots=True)
class StationSpec:
    """Candidate base poses for one task-owned station identifier."""

    name: str
    base_candidates: list[Pose]
    description: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


__all__ = ["PathResult", "Pose", "StationSpec", "StationTarget", "Waypoint"]
