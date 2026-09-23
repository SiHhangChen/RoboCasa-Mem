"""Robot-side boundary for WR07 observation and command adapters."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol

import numpy as np

from bench.policy.protocol import PolicyContract


class WR07RobotIO(Protocol):
    def observe(self) -> tuple[np.ndarray, Mapping[str, np.ndarray]]:
        """Return state17 and synchronized raw RGB front/left/right images."""

    def verify_execution_contract(self, contract: PolicyContract) -> bool:
        """Return True only after action17 units, mapping, limits, hold and stop are verified."""

    def execute_chunk(self, actions: np.ndarray, *, frequency_hz: float) -> None:
        """Execute exactly the provided action17 rows in order at the given rate."""

    def hold(self) -> None:
        """Stop motion safely when inference or transport fails."""

    def close(self) -> None: ...
