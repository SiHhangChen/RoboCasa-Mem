"""Deterministic random streams for reproducible data collection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import zlib

import numpy as np


def stable_seed(base_seed: int, *tokens: Any) -> int:
    """Derive a stable uint32 seed from an episode seed and semantic tokens."""
    seed = int(base_seed) & 0xFFFFFFFF
    for token in tokens:
        payload = str(token).encode("utf-8")
        seed = zlib.crc32(payload, seed) & 0xFFFFFFFF
    return int(seed)


@dataclass(frozen=True)
class SeedStream:
    """Small helper for named RNG streams.

    Use semantic tokens such as task id, phase name, station name, or attempt
    index. This keeps data generation reproducible while making each random
    decision auditable.
    """

    base_seed: int
    namespace: str = ""

    def seed(self, *tokens: Any) -> int:
        if self.namespace:
            return stable_seed(self.base_seed, self.namespace, *tokens)
        return stable_seed(self.base_seed, *tokens)

    def rng(self, *tokens: Any) -> np.random.Generator:
        return np.random.default_rng(self.seed(*tokens))

    def spawn(self, *tokens: Any) -> "SeedStream":
        namespace = ":".join(str(item) for item in ((self.namespace,) if self.namespace else ()) + tokens)
        return SeedStream(base_seed=self.base_seed, namespace=namespace)
