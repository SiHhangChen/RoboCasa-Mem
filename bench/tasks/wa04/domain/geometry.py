"""Pure geometry and assignment validation for WA-04."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import combinations, permutations
from typing import Any

import numpy as np

from bench.tasks.wa04.domain.contracts import CUP_IDS, SERVICE_SLOTS


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
            raise ValueError("WA-04 cup sampling inset must be non-negative")
        sampled_size = tuple(
            self.size_xy[index] - 2.0 * inset[index]
            for index in range(2)
        )
        if any(value <= 0.0 for value in sampled_size):
            raise ValueError("WA-04 cup sampling inset consumes the legal region")
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
    """Validated independent placement regions for three neutral slots."""

    regions_by_slot: Mapping[str, ServicePlacementRegion]
    surface_z_offset: float
    minimum_xy_separation: float
    minimum_inter_region_gap: float
    placement_center_margin_xy: tuple[float, float]


def validate_service_slot_assignment(value: Mapping[str, Any]) -> dict[str, str]:
    """Require a one-to-one mapping from all cups to all neutral slots."""

    assignment = {str(key): str(item) for key, item in dict(value).items()}
    if set(assignment) != set(CUP_IDS):
        raise ValueError("WA-04 service slot assignment must contain cup_A, cup_B, and cup_C")
    if set(assignment.values()) != set(SERVICE_SLOTS):
        raise ValueError("WA-04 service slot assignment must use all three slots once")
    return assignment


def service_slot_assignment_for_seed(
    seed: int,
    *,
    permutation_index: int | None = None,
) -> dict[str, str]:
    """Return one of all six cup-to-slot permutations."""

    index = int(seed) % 6 if permutation_index is None else int(permutation_index)
    slot_order = tuple(permutations(SERVICE_SLOTS))[index % 6]
    return validate_service_slot_assignment(dict(zip(CUP_IDS, slot_order, strict=True)))


def resolve_cup_sampling_regions(
    value: Mapping[str, Any],
) -> dict[str, CupSamplingRegion]:
    """Parse the three task-owned, independent fixture-local cup regions."""

    regions: dict[str, CupSamplingRegion] = {}
    for cup in CUP_IDS:
        raw = value.get(cup)
        if not isinstance(raw, Mapping):
            raise ValueError(f"WA-04 initial_state.objects.cups.{cup} must be a mapping")
        center = _xy(raw.get("region_center_xy"), f"cups.{cup}.region_center_xy")
        size = _xy(raw.get("region_size_xy"), f"cups.{cup}.region_size_xy")
        if any(item <= 0.0 for item in size):
            raise ValueError(f"WA-04 cups.{cup}.region_size_xy must be positive")
        yaw = float(raw.get("yaw_rad"))
        if not np.isfinite(yaw):
            raise ValueError(f"WA-04 cups.{cup}.yaw_rad must be finite")
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
        raise ValueError("WA-04 cup sampling inset must be non-negative")
    return result


def validate_cup_region_layout(
    regions: Mapping[str, CupSamplingRegion],
    constraints: CupRegionConstraints,
) -> None:
    """Enforce the left-to-right sink / A / B / C parallel geometry."""

    if set(regions) != set(CUP_IDS):
        raise ValueError("WA-04 cup regions must contain cup_A, cup_B, and cup_C")
    for cup_id, region in regions.items():
        try:
            region.inset(constraints.sampling_inset_xy)
        except ValueError as exc:
            raise ValueError(
                f"WA-04 cup sampling inset is invalid for {cup_id}"
            ) from exc
    ordered = [regions[cup] for cup in CUP_IDS]
    if any(
        not np.isclose(ordered[0].center_xy[1], region.center_xy[1], atol=1e-9)
        for region in ordered[1:]
    ):
        raise ValueError("WA-04 cup regions must share one counter-local y center")

    if any(
        ordered[index].center_xy[0] >= ordered[index + 1].center_xy[0]
        for index in range(len(ordered) - 1)
    ):
        raise ValueError("WA-04 cup regions must be ordered cup_A, cup_B, cup_C")

    sink_clearance = ordered[0].x_bounds[0] - constraints.sink_exterior_local_x_max
    if sink_clearance + 1e-9 < constraints.minimum_sink_clearance:
        raise ValueError(
            "WA-04 cup_A region violates sink clearance: "
            f"{sink_clearance:.4f} < {constraints.minimum_sink_clearance:.4f}"
        )

    for left, right in zip(ordered, ordered[1:]):
        region_gap = right.x_bounds[0] - left.x_bounds[1]
        if region_gap + 1e-9 < constraints.minimum_inter_region_gap:
            raise ValueError(
                "WA-04 cup regions violate minimum gap: "
                f"{region_gap:.4f} < {constraints.minimum_inter_region_gap:.4f}"
            )


CUP_PLACEMENT_SALTS = {
    "cup_A": 0x57413034,
    "cup_B": 0x57423034,
    "cup_C": 0x57433034,
}
SERVICE_PLACEMENT_SALTS = {
    "slot_1": 0x53563134,
    "slot_2": 0x53563234,
    "slot_3": 0x53563334,
}


def cup_placement_rng(seed: int, cup_id: str) -> np.random.Generator:
    """Return one deterministic RNG stream owned only by the selected cup."""

    cup = str(cup_id)
    if cup not in CUP_IDS:
        raise ValueError(f"invalid WA-04 cup id: {cup_id!r}")
    return np.random.default_rng(
        np.random.SeedSequence([int(seed), int(CUP_PLACEMENT_SALTS[cup])])
    )


def service_placement_rng(seed: int, slot: str) -> np.random.Generator:
    """Return an independent deterministic stream for one service region."""

    name = str(slot)
    if name not in SERVICE_SLOTS:
        raise ValueError(f"invalid WA-04 service slot: {slot!r}")
    return np.random.default_rng(
        np.random.SeedSequence([int(seed), int(SERVICE_PLACEMENT_SALTS[name])])
    )


def resolve_service_slot_layout(value: Mapping[str, Any]) -> ServiceSlotLayout:
    """Parse three cloned, independent service placement regions."""

    raw_regions = value.get("regions")
    if not isinstance(raw_regions, Mapping):
        raise ValueError("WA-04 service_slots.regions must be a mapping")
    regions: dict[str, ServicePlacementRegion] = {}
    for slot in SERVICE_SLOTS:
        raw = raw_regions.get(slot)
        if not isinstance(raw, Mapping):
            raise ValueError(f"WA-04 service_slots.regions.{slot} must be a mapping")
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
                f"WA-04 service_slots.regions.{slot}.region_size_xy must be positive"
            )
        regions[slot] = ServicePlacementRegion(center_xy=center, size_xy=size)

    minimum = float(value.get("minimum_xy_separation", 0.20))
    if not np.isfinite(minimum) or minimum <= 0.0:
        raise ValueError("WA-04 service slot minimum_xy_separation must be positive")
    for first, second in combinations(regions.values(), 2):
        separation = float(
            np.linalg.norm(np.asarray(first.center_xy) - np.asarray(second.center_xy))
        )
        if separation + 1.0e-9 < minimum:
            raise ValueError(
                "WA-04 service slots violate minimum separation: "
                f"{separation:.4f} < {minimum:.4f}"
            )
    minimum_gap = _positive(
        value.get("minimum_inter_region_gap", 0.07),
        "service_slots.minimum_inter_region_gap",
    )
    ordered = sorted(regions.values(), key=lambda region: region.center_xy[0])
    for left, right in zip(ordered, ordered[1:]):
        region_gap = right.x_bounds[0] - left.x_bounds[1]
        if region_gap + 1.0e-9 < minimum_gap:
            raise ValueError(
                "WA-04 service regions violate minimum gap: "
                f"{region_gap:.4f} < {minimum_gap:.4f}"
            )
    margin = _xy(
        value.get("placement_center_margin_xy", (0.04, 0.04)),
        "service_slots.placement_center_margin_xy",
    )
    if any(item < 0.0 for item in margin):
        raise ValueError("WA-04 service placement margins must be non-negative")
    for slot, region in regions.items():
        if any(2.0 * margin[index] >= region.size_xy[index] for index in range(2)):
            raise ValueError(
                f"WA-04 service placement margin consumes region {slot}"
            )
    z_offset = float(value.get("surface_z_offset", 0.003))
    if not np.isfinite(z_offset):
        raise ValueError("WA-04 service_slots.surface_z_offset must be finite")
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
        raise ValueError(f"WA-04 {label} must contain two numeric values")
    result = (float(value[0]), float(value[1]))
    if not all(np.isfinite(item) for item in result):
        raise ValueError(f"WA-04 {label} must be finite")
    return result


def _finite(value: Any, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"WA-04 {label} must be numeric") from exc
    if not np.isfinite(result):
        raise ValueError(f"WA-04 {label} must be finite")
    return result


def _positive(value: Any, label: str) -> float:
    result = _finite(value, label)
    if result <= 0.0:
        raise ValueError(f"WA-04 {label} must be positive")
    return result


__all__ = [
    "CUP_PLACEMENT_SALTS",
    "SERVICE_PLACEMENT_SALTS",
    "CupRegionConstraints",
    "CupSamplingRegion",
    "ServiceSlotLayout",
    "ServicePlacementRegion",
    "cup_placement_rng",
    "resolve_cup_region_constraints",
    "resolve_cup_sampling_regions",
    "resolve_service_slot_layout",
    "sample_service_slot_relative_xy",
    "service_placement_rng",
    "service_slot_assignment_for_seed",
    "validate_cup_region_layout",
    "validate_service_slot_assignment",
]
