"""Pure geometry and assignment validation for WA-05."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from bench.tasks.wa05.domain.contracts import CUP_IDS, SERVICE_SLOTS


@dataclass(frozen=True)
class CupSamplingRegion:
    """One cup's independent counter-local reset region."""

    center_xy: tuple[float, float]
    size_xy: tuple[float, float]
    yaw_rad: float

    @property
    def x_bounds(self) -> tuple[float, float]:
        half = 0.5 * self.size_xy[0]
        return (self.center_xy[0] - half, self.center_xy[0] + half)

    @property
    def y_bounds(self) -> tuple[float, float]:
        half = 0.5 * self.size_xy[1]
        return (self.center_xy[1] - half, self.center_xy[1] + half)

    def inset(self, inset_xy: Sequence[float]) -> "CupSamplingRegion":
        """Return a centered sampling region inset from this legal region."""

        inset = _xy(inset_xy, "cup_region_constraints.sampling_inset_xy")
        if any(value < 0.0 for value in inset):
            raise ValueError("WA-05 cup sampling inset must be non-negative")
        sampled_size = tuple(
            self.size_xy[index] - 2.0 * inset[index]
            for index in range(2)
        )
        if any(value <= 0.0 for value in sampled_size):
            raise ValueError("WA-05 cup sampling inset consumes the legal region")
        return CupSamplingRegion(
            center_xy=self.center_xy,
            size_xy=sampled_size,
            yaw_rad=self.yaw_rad,
        )


@dataclass(frozen=True)
class CupRegionConstraints:
    """Hard layout constraints protecting the sink and separating the cups."""

    sink_exterior_local_x_max: float
    minimum_sink_clearance: float
    minimum_inter_region_gap: float
    sampling_inset_xy: tuple[float, float] = (0.0, 0.0)


@dataclass(frozen=True)
class ServicePlacementRegion:
    """One neutral service slot's counter-local placement region."""

    center_xy: tuple[float, float]
    size_xy: tuple[float, float]

    @property
    def x_bounds(self) -> tuple[float, float]:
        half = 0.5 * self.size_xy[0]
        return (self.center_xy[0] - half, self.center_xy[0] + half)

    @property
    def y_bounds(self) -> tuple[float, float]:
        half = 0.5 * self.size_xy[1]
        return (self.center_xy[1] - half, self.center_xy[1] + half)


@dataclass(frozen=True)
class ServiceSlotLayout:
    """Validated placement region for the fixed service slot."""

    regions_by_slot: Mapping[str, ServicePlacementRegion]
    surface_z_offset: float
    minimum_xy_separation: float
    minimum_inter_region_gap: float
    placement_center_margin_xy: tuple[float, float]


def other_cup_id(cup_id: str) -> str:
    raise ValueError("WA-05 has only one cup")


def validate_service_slot_assignment(value: Mapping[str, Any]) -> dict[str, str]:
    """Require the single cup to use the single fixed service slot."""

    assignment = {str(key): str(item) for key, item in dict(value).items()}
    if set(assignment) != set(CUP_IDS):
        raise ValueError("WA-05 service slot assignment must contain cup_A")
    if set(assignment.values()) != set(SERVICE_SLOTS):
        raise ValueError("WA-05 service slot assignment must use slot_1")
    return assignment


def service_slot_assignment_for_seed(seed: int, *, swap: bool | None = None) -> dict[str, str]:
    """Return the fixed slot mapping; seed is irrelevant for WA-05."""

    del seed, swap
    return validate_service_slot_assignment({"cup_A": "slot_1"})


def resolve_cup_sampling_regions(
    value: Mapping[str, Any],
) -> dict[str, CupSamplingRegion]:
    """Parse the task-owned fixture-local cup region."""

    regions: dict[str, CupSamplingRegion] = {}
    for cup in CUP_IDS:
        raw = value.get(cup)
        if not isinstance(raw, Mapping):
            raise ValueError(f"WA-05 initial_state.objects.cups.{cup} must be a mapping")
        center = _xy(raw.get("region_center_xy"), f"cups.{cup}.region_center_xy")
        size = _xy(raw.get("region_size_xy"), f"cups.{cup}.region_size_xy")
        if any(item <= 0.0 for item in size):
            raise ValueError(f"WA-05 cups.{cup}.region_size_xy must be positive")
        yaw = float(raw.get("yaw_rad"))
        if not np.isfinite(yaw):
            raise ValueError(f"WA-05 cups.{cup}.yaw_rad must be finite")
        regions[cup] = CupSamplingRegion(
            center_xy=center,
            size_xy=size,
            yaw_rad=yaw,
        )
    return regions


def resolve_cup_region_constraints(value: Mapping[str, Any]) -> CupRegionConstraints:
    """Parse sink clearance and inter-region gap requirements."""

    result = CupRegionConstraints(
        sink_exterior_local_x_max=_finite(
            value.get("sink_exterior_local_x_max"),
            "cup_region_constraints.sink_exterior_local_x_max",
        ),
        minimum_sink_clearance=_positive(
            value.get("minimum_sink_clearance"),
            "cup_region_constraints.minimum_sink_clearance",
        ),
        minimum_inter_region_gap=_positive(
            value.get("minimum_inter_region_gap"),
            "cup_region_constraints.minimum_inter_region_gap",
        ),
        sampling_inset_xy=_xy(
            value.get("sampling_inset_xy", (0.0, 0.0)),
            "cup_region_constraints.sampling_inset_xy",
        ),
    )
    if any(item < 0.0 for item in result.sampling_inset_xy):
        raise ValueError("WA-05 cup sampling inset must be non-negative")
    return result


def validate_cup_region_layout(
    regions: Mapping[str, CupSamplingRegion],
    constraints: CupRegionConstraints,
) -> None:
    """Enforce sink clearance for the single cup region."""

    if set(regions) != set(CUP_IDS):
        raise ValueError("WA-05 cup regions must contain cup_A")
    for cup_id, region in regions.items():
        try:
            region.inset(constraints.sampling_inset_xy)
        except ValueError as exc:
            raise ValueError(
                f"WA-05 cup sampling inset is invalid for {cup_id}"
            ) from exc
    cup = regions["cup_A"]
    sink_clearance = cup.x_bounds[0] - constraints.sink_exterior_local_x_max
    if sink_clearance + 1e-9 < constraints.minimum_sink_clearance:
        raise ValueError(
            "WA-05 cup_A region violates sink clearance: "
            f"{sink_clearance:.4f} < {constraints.minimum_sink_clearance:.4f}"
        )



CUP_PLACEMENT_SALTS = {
    "cup_A": 0x57413032,
}
SERVICE_PLACEMENT_SALTS = {
    "slot_1": 0x53563132,
}


def cup_placement_rng(seed: int, cup_id: str) -> np.random.Generator:
    """Return one deterministic RNG stream owned only by the selected cup."""

    cup = str(cup_id)
    if cup not in CUP_IDS:
        raise ValueError(f"invalid WA-05 cup id: {cup_id!r}")
    return np.random.default_rng(
        np.random.SeedSequence([int(seed), int(CUP_PLACEMENT_SALTS[cup])])
    )


def service_placement_rng(seed: int, slot: str) -> np.random.Generator:
    """Return an independent deterministic stream for one service region."""

    name = str(slot)
    if name not in SERVICE_SLOTS:
        raise ValueError(f"invalid WA-05 service slot: {slot!r}")
    return np.random.default_rng(
        np.random.SeedSequence([int(seed), int(SERVICE_PLACEMENT_SALTS[name])])
    )


def resolve_service_slot_layout(value: Mapping[str, Any]) -> ServiceSlotLayout:
    """Parse the fixed service placement region."""

    raw_regions = value.get("regions")
    if not isinstance(raw_regions, Mapping):
        raise ValueError("WA-05 service_slots.regions must be a mapping")
    regions: dict[str, ServicePlacementRegion] = {}
    for slot in SERVICE_SLOTS:
        raw = raw_regions.get(slot)
        if not isinstance(raw, Mapping):
            raise ValueError(f"WA-05 service_slots.regions.{slot} must be a mapping")
        center = _xy(
            raw.get("region_center_xy"),
            f"service_slots.regions.{slot}.region_center_xy",
        )
        size = _xy(
            raw.get("region_size_xy"),
            f"service_slots.regions.{slot}.region_size_xy",
        )
        if any(item <= 0.0 for item in size):
            raise ValueError(
                f"WA-05 service_slots.regions.{slot}.region_size_xy must be positive"
            )
        regions[slot] = ServicePlacementRegion(center_xy=center, size_xy=size)

    minimum = float(value.get("minimum_xy_separation", 0.20))
    if not np.isfinite(minimum) or minimum <= 0.0:
        raise ValueError("WA-05 service slot minimum_xy_separation must be positive")
    minimum_gap = _positive(
        value.get("minimum_inter_region_gap", 0.07),
        "service_slots.minimum_inter_region_gap",
    )
    margin = _xy(
        value.get("placement_center_margin_xy", (0.04, 0.04)),
        "service_slots.placement_center_margin_xy",
    )
    if any(item < 0.0 for item in margin):
        raise ValueError("WA-05 service placement margins must be non-negative")
    for slot, region in regions.items():
        if any(2.0 * margin[index] >= region.size_xy[index] for index in range(2)):
            raise ValueError(
                f"WA-05 service placement margin consumes region {slot}"
            )
    z_offset = float(value.get("surface_z_offset", 0.003))
    if not np.isfinite(z_offset):
        raise ValueError("WA-05 service_slots.surface_z_offset must be finite")
    return ServiceSlotLayout(
        regions_by_slot=regions,
        surface_z_offset=z_offset,
        minimum_xy_separation=minimum,
        minimum_inter_region_gap=minimum_gap,
        placement_center_margin_xy=margin,
    )


def sample_service_slot_relative_xy(
    layout: ServiceSlotLayout,
    *,
    seed: int,
) -> dict[str, tuple[float, float]]:
    """Return the fixed center of each service region as its placement target."""

    # Keep ``seed`` in the public API for compatibility with existing callers. Service
    # placement is intentionally deterministic now, regardless of the episode seed.
    del seed
    result: dict[str, tuple[float, float]] = {}
    for slot in SERVICE_SLOTS:
        region = layout.regions_by_slot[slot]
        result[slot] = tuple(region.center_xy)
    return result


def _xy(value: Any, label: str) -> tuple[float, float]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or len(value) != 2:
        raise ValueError(f"WA-05 {label} must contain two numeric values")
    result = (float(value[0]), float(value[1]))
    if not all(np.isfinite(item) for item in result):
        raise ValueError(f"WA-05 {label} must be finite")
    return result


def _finite(value: Any, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"WA-05 {label} must be numeric") from exc
    if not np.isfinite(result):
        raise ValueError(f"WA-05 {label} must be finite")
    return result


def _positive(value: Any, label: str) -> float:
    result = _finite(value, label)
    if result <= 0.0:
        raise ValueError(f"WA-05 {label} must be positive")
    return result


__all__ = [
    "CUP_PLACEMENT_SALTS",
    "SERVICE_PLACEMENT_SALTS",
    "CupRegionConstraints",
    "CupSamplingRegion",
    "ServiceSlotLayout",
    "ServicePlacementRegion",
    "cup_placement_rng",
    "other_cup_id",
    "resolve_cup_region_constraints",
    "resolve_cup_sampling_regions",
    "resolve_service_slot_layout",
    "sample_service_slot_relative_xy",
    "service_placement_rng",
    "service_slot_assignment_for_seed",
    "validate_cup_region_layout",
    "validate_service_slot_assignment",
]
