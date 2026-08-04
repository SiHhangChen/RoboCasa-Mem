"""Pure shelf level representation and return relation."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ShelfReturnSpec:
    item_ids: tuple[str, ...]
    levels: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class OriginalLevels:
    values: tuple[tuple[str, str], ...]

    def level_for(self, item_id: str) -> str:
        for item, level in self.values:
            if item == item_id:
                return level
        raise KeyError(item_id)
