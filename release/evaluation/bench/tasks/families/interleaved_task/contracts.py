"""Pure precedence/event graph representation."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class EventGraph:
    events: tuple[str, ...]
    precedence: tuple[tuple[str, str], ...] = ()

    def legal_next(self, completed: set[str]) -> tuple[str, ...]:
        return tuple(
            event
            for event in self.events
            if event not in completed
            and all(before in completed for before, after in self.precedence if after == event)
        )
