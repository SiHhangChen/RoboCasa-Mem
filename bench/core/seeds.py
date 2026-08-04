"""Stable, namespaced random seeds.

All episode randomness should be resolved before simulator creation.  These
helpers make it possible for recording and evaluation to derive the same
sub-streams without sharing mutable RNG state.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Iterator


def stable_seed(seed: int, namespace: str = "episode", *, index: int = 0) -> int:
    """Derive a reproducible 32-bit seed from an integer and namespace."""

    payload = f"membench-seed/v1|{int(seed)}|{namespace}|{int(index)}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**32)


@dataclass(frozen=True, slots=True)
class SeedStream:
    """Immutable namespace for deterministic child seed generation."""

    seed: int
    namespace: str = "episode"

    @property
    def base_seed(self) -> int:
        """Compatibility spelling for execution components during migration."""

        return int(self.seed)

    def rng(self, *tokens: object):
        """Return a deterministic NumPy generator for an optional substream."""

        import numpy as np

        namespace = ":".join((self.namespace, *(str(token) for token in tokens)))
        return np.random.default_rng(stable_seed(self.seed, namespace))

    def spawn(self, *tokens: object) -> "SeedStream":
        namespace = ":".join((self.namespace, *(str(token) for token in tokens)))
        return SeedStream(self.seed, namespace)

    def child(self, namespace: str, *, index: int = 0) -> int:
        return stable_seed(self.seed, f"{self.namespace}:{namespace}", index=index)

    def iter(self, namespace: str, *, start: int = 0) -> Iterator[int]:
        index = int(start)
        while True:
            yield self.child(namespace, index=index)
            index += 1


__all__ = ["SeedStream", "stable_seed"]
