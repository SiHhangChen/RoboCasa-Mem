"""Native recording consistency checks and final disposition."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class EpisodeDisposition(str, Enum):
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    ERROR = "error"


@dataclass(frozen=True, slots=True)
class RecordingValidation:
    disposition: EpisodeDisposition
    reasons: tuple[str, ...] = ()
    facts: Mapping[str, Any] = field(default_factory=dict)

    @property
    def accepted(self) -> bool:
        return self.disposition is EpisodeDisposition.ACCEPTED


def validate_aligned_lengths(fields: Mapping[str, Sequence[Any]], *, required: tuple[str, ...] = ("actions", "observations")) -> RecordingValidation:
    missing = tuple(name for name in required if name not in fields)
    if missing:
        return RecordingValidation(EpisodeDisposition.ERROR, tuple(f"missing:{name}" for name in missing))
    lengths = {name: len(values) for name, values in fields.items()}
    if not lengths or len(set(lengths.values())) != 1:
        return RecordingValidation(EpisodeDisposition.REJECTED, ("unaligned_lengths",), lengths)
    if next(iter(lengths.values())) <= 0:
        return RecordingValidation(EpisodeDisposition.REJECTED, ("empty_episode",), lengths)
    return RecordingValidation(EpisodeDisposition.ACCEPTED, facts=lengths)


__all__ = ["EpisodeDisposition", "RecordingValidation", "validate_aligned_lengths"]
