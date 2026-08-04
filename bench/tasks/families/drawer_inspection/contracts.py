"""Pure fixture-order and exactly-once inspection state."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class DrawerInspectionSpec:
    fixture_ids: tuple[str, ...]
    required_inspections: int

    def __post_init__(self) -> None:
        if not self.fixture_ids or self.required_inspections != len(self.fixture_ids):
            raise ValueError("required_inspections must match fixture count")


@dataclass(slots=True)
class InspectionState:
    inspected: set[str] = field(default_factory=set)

    def record(self, fixture_id: str) -> bool:
        if fixture_id in self.inspected:
            return False
        self.inspected.add(fixture_id)
        return True

    def complete(self, spec: DrawerInspectionSpec) -> bool:
        return len(self.inspected) == spec.required_inspections
