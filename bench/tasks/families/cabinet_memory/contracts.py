"""Pure storage relation and memory barrier state."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class CabinetMemorySpec:
    item_ids: tuple[str, ...]
    level_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.item_ids or not self.level_ids:
            raise ValueError("cabinet memory requires items and levels")


@dataclass(frozen=True, slots=True)
class MemoryRelation:
    item_to_level: tuple[tuple[str, str], ...]

    def level_for(self, item_id: str) -> str:
        for item, level in self.item_to_level:
            if item == item_id:
                return level
        raise KeyError(item_id)
