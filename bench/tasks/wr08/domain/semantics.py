"""History-aware semantic events for WR-08 color-matched transport."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

from bench.tasks.wr08.domain.contracts import ITEM_IDS, SOURCE_SLOTS, TARGET_SLOTS


@dataclass
class SemanticState:
    source_slot_by_item: dict[str, str] = field(default_factory=dict)
    source_color_by_item: dict[str, str] = field(default_factory=dict)
    item_picked_by_item: dict[str, bool] = field(
        default_factory=lambda: {item: False for item in ITEM_IDS}
    )
    source_color_memory_latched_by_item: dict[str, bool] = field(
        default_factory=lambda: {item: False for item in ITEM_IDS}
    )
    target_slot_attempted_by_item: dict[str, str] = field(default_factory=dict)
    item_placed_by_item: dict[str, bool] = field(
        default_factory=lambda: {item: False for item in ITEM_IDS}
    )
    target_color_matches_by_item: dict[str, bool] = field(
        default_factory=lambda: {item: False for item in ITEM_IDS}
    )
    protocol_violation: bool = False
    stage_regressions: list[str] = field(default_factory=list)
    # Deprecated fields accepted when reading pre-v2 analysis fixtures.
    original_zone_by_item: dict[str, str] = field(default_factory=dict)
    staged_at_plate: dict[str, bool] = field(default_factory=dict)
    memory_barrier_latched: bool = False
    item_removed_from_zone: dict[str, bool] = field(default_factory=dict)
    returned_item_ids: set[str] = field(default_factory=set)

    def __post_init__(self) -> None:
        if not self.source_slot_by_item and self.original_zone_by_item:
            self.source_slot_by_item = {
                item: str(zone).replace("counter_zone_", "source_")
                for item, zone in self.original_zone_by_item.items()
            }
        if self.staged_at_plate:
            for item, value in self.staged_at_plate.items():
                if item in self.item_placed_by_item:
                    self.item_placed_by_item[item] = bool(value)
                    self.target_color_matches_by_item[item] = bool(value)
        if self.memory_barrier_latched:
            for item in ITEM_IDS:
                self.item_picked_by_item[item] = True
                self.source_color_memory_latched_by_item[item] = True


@dataclass(frozen=True)
class PhysicalState:
    item_in_source: Mapping[str, bool] = field(default_factory=dict)
    item_in_target: Mapping[str, bool] = field(default_factory=dict)
    released: Mapping[str, bool] = field(default_factory=dict)
    item_at_plate: Mapping[str, bool] = field(default_factory=dict)
    item_in_zone: Mapping[str, Mapping[str, bool]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        source = dict(self.item_in_source)
        target = dict(self.item_in_target)
        released = dict(self.released)
        if not target and self.item_at_plate:
            target = dict(self.item_at_plate)
        if not source and self.item_in_zone:
            source = {item: any(bool(value) for value in self.item_in_zone.get(item, {}).values()) for item in ITEM_IDS}
        object.__setattr__(self, "item_in_source", {item: bool(source.get(item, False)) for item in ITEM_IDS})
        object.__setattr__(self, "item_in_target", {item: bool(target.get(item, False)) for item in ITEM_IDS})
        object.__setattr__(self, "released", {item: bool(released.get(item, True)) for item in ITEM_IDS})


@dataclass(frozen=True)
class SuccessReport:
    initial_source_verified: bool
    items_picked: Mapping[str, bool]
    colors_latched: Mapping[str, bool]
    items_placed: Mapping[str, bool]
    target_color_matches: Mapping[str, bool]
    all_items_released: bool
    no_wrong_target_placement: bool
    no_protocol_violation: bool

    @property
    def success(self) -> bool:
        return all(self.checks().values())

    def checks(self) -> dict[str, bool]:
        checks = {
            "initial_source_physically_verified": bool(self.initial_source_verified),
            "both_items_picked": all(self.items_picked.values()),
            "both_colors_memory_latched": all(self.colors_latched.values()),
            "both_items_on_matching_target_plates": all(self.items_placed.values()) and all(self.target_color_matches.values()),
            "all_items_released": bool(self.all_items_released),
            "no_wrong_target_placement": bool(self.no_wrong_target_placement),
            "no_protocol_violation": bool(self.no_protocol_violation),
        }
        checks.update({f"{item}_picked": bool(self.items_picked.get(item, False)) for item in ITEM_IDS})
        checks.update({f"{item}_color_latched": bool(self.colors_latched.get(item, False)) for item in ITEM_IDS})
        checks.update({f"{item}_matched": bool(self.items_placed.get(item, False) and self.target_color_matches.get(item, False)) for item in ITEM_IDS})
        return checks


def record_initial_sources(
    state: SemanticState,
    source_slot_by_item: Mapping[str, str],
    source_color_by_item: Mapping[str, str],
) -> bool:
    slots = {str(item): str(slot) for item, slot in source_slot_by_item.items()}
    colors = {str(item): str(color) for item, color in source_color_by_item.items()}
    valid = (
        set(slots) == set(ITEM_IDS)
        and set(colors) == set(ITEM_IDS)
        and all(slot in SOURCE_SLOTS for slot in slots.values())
        and len(set(slots.values())) == len(ITEM_IDS)
        and all(bool(color) for color in colors.values())
        and not state.source_slot_by_item
    )
    if not valid:
        state.protocol_violation = True
        return False
    state.source_slot_by_item = slots
    state.source_color_by_item = colors
    return True


def record_item_picked(state: SemanticState, item: str, *, physically_picked: bool) -> bool:
    item = str(item)
    if item not in ITEM_IDS:
        raise KeyError(item)
    if physically_picked and item in state.source_slot_by_item and not state.protocol_violation:
        state.item_picked_by_item[item] = True
    return bool(state.item_picked_by_item.get(item, False))


def latch_source_color_memory(state: SemanticState, item: str) -> bool:
    item = str(item)
    if item not in ITEM_IDS:
        raise KeyError(item)
    if state.item_picked_by_item.get(item, False) and not state.protocol_violation:
        state.source_color_memory_latched_by_item[item] = True
    return bool(state.source_color_memory_latched_by_item.get(item, False))


def record_target_attempt(
    state: SemanticState,
    item: str,
    slot: str,
    *,
    matches: bool,
) -> bool:
    item, slot = str(item), str(slot)
    if item not in ITEM_IDS or slot not in TARGET_SLOTS:
        state.protocol_violation = True
        return False
    if not state.source_color_memory_latched_by_item.get(item, False):
        state.protocol_violation = True
        return False
    previous = state.target_slot_attempted_by_item.get(item)
    if previous is not None and previous != slot:
        state.protocol_violation = True
        return False
    state.target_slot_attempted_by_item.setdefault(item, slot)
    if not matches:
        state.protocol_violation = True
        return False
    return True


def record_target_placement(state: SemanticState, item: str, *, matches: bool) -> bool:
    item = str(item)
    if item not in ITEM_IDS or not matches or not state.source_color_memory_latched_by_item.get(item, False) or state.protocol_violation:
        if not matches:
            state.protocol_violation = True
        return False
    state.item_placed_by_item[item] = True
    state.target_color_matches_by_item[item] = True
    return True


def evaluate_success(
    state: SemanticState,
    physical: PhysicalState,
    *,
    target_slot_by_item: Mapping[str, str] | None = None,
    target_plate_color_by_slot: Mapping[str, str] | None = None,
) -> SuccessReport:
    del target_plate_color_by_slot
    in_target = {
        item: bool(physical.item_in_target.get(item, False) and physical.released.get(item, False))
        for item in ITEM_IDS
    }
    wrong_target = any(
        bool(physical.item_in_target.get(item, False)) and not in_target[item]
        for item in ITEM_IDS
    )
    # The environment records an attempt against the actual target slot. A
    # mismatch is already a protocol violation; this check protects callers
    # that construct PhysicalState directly.
    target_slot_by_item = dict(target_slot_by_item or {})
    attempted_slots = dict(state.target_slot_attempted_by_item)
    wrong_target = wrong_target or any(
        item in attempted_slots and target_slot_by_item and attempted_slots[item] != str(target_slot_by_item.get(item, ""))
        for item in ITEM_IDS
    )
    placed = {
        item: bool(state.item_placed_by_item.get(item, False) and in_target[item])
        for item in ITEM_IDS
    }
    matches = {
        item: bool(state.target_color_matches_by_item.get(item, False) and placed[item])
        for item in ITEM_IDS
    }
    return SuccessReport(
        initial_source_verified=(set(state.source_slot_by_item) == set(ITEM_IDS)),
        items_picked={item: bool(state.item_picked_by_item.get(item, False)) for item in ITEM_IDS},
        colors_latched={item: bool(state.source_color_memory_latched_by_item.get(item, False)) for item in ITEM_IDS},
        items_placed=placed,
        target_color_matches=matches,
        all_items_released=all(bool(physical.released.get(item, False)) for item in ITEM_IDS),
        no_wrong_target_placement=not wrong_target,
        no_protocol_violation=not state.protocol_violation,
    )


__all__ = [
    "PhysicalState", "SemanticState", "SuccessReport", "evaluate_success",
    "latch_source_color_memory", "record_initial_sources", "record_item_picked",
    "record_target_attempt", "record_target_placement", "record_initial_zones",
    "record_item_removed", "record_plate_stage", "latch_memory_barrier",
    "record_return_attempt", "record_return_completion", "record_stage_regression",
]


# Compatibility shims for pre-v2 callers. Active WR-08 code never uses these.
def record_initial_zones(state: SemanticState, zones: Mapping[str, str], *, physically_verified: bool = True) -> bool:
    if not physically_verified:
        state.protocol_violation = True
        return False
    mapped = {item: str(zone).replace("counter_zone_", "source_") for item, zone in zones.items()}
    return record_initial_sources(state, mapped, {item: state.source_color_by_item.get(item, "unknown") for item in mapped})


def record_item_removed(state: SemanticState, item: str, *, physically_removed: bool) -> bool:
    return record_item_picked(state, item, physically_picked=physically_removed)


def record_plate_stage(state: SemanticState, item: str, *args: Any, item_at_plate: bool | None = None, item_outside_zones: bool | None = None) -> bool:
    del args, item_outside_zones
    value = bool(True if item_at_plate is None else item_at_plate)
    state.item_placed_by_item[str(item)] = value
    state.target_color_matches_by_item[str(item)] = value
    return value


def latch_memory_barrier(state: SemanticState, *, item_at_plate: Mapping[str, bool] | None = None, all_outside_zones: bool = True) -> bool:
    del item_at_plate
    if all_outside_zones and not state.protocol_violation:
        state.memory_barrier_latched = True
        for item in ITEM_IDS:
            state.item_picked_by_item[item] = True
            state.source_color_memory_latched_by_item[item] = True
    return state.memory_barrier_latched


def record_return_attempt(state: SemanticState, item: str, zone: str) -> bool:
    del item, zone
    state.protocol_violation = True
    return False


def record_return_completion(state: SemanticState, item: str, *, item_in_original_zone: bool) -> bool:
    del item, item_in_original_zone
    state.protocol_violation = True
    return False


def record_stage_regression(state: SemanticState, *, kind: str) -> bool:
    if kind not in state.stage_regressions:
        state.stage_regressions.append(str(kind))
    state.protocol_violation = True
    return True
