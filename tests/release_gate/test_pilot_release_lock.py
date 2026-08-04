from __future__ import annotations

import json
from pathlib import Path

import pytest

from bench.release import build_release_lock, resolve_release_entry, verify_release_lock


ROOT = Path(__file__).resolve().parents[2]
LOCK_FILE = (
    ROOT
    / "configs"
    / "benchmark"
    / "release_locks"
    / "pilot_wr01_wx01_v1.json"
)


@pytest.mark.release_gate
def test_pilot_release_lock_matches_current_resolved_inputs() -> None:
    locked = json.loads(LOCK_FILE.read_text(encoding="utf-8"))
    verify_release_lock(locked)
    current = build_release_lock(
        [
            resolve_release_entry("WR-01", seed=0),
            resolve_release_entry("WX-01", seed=0),
        ],
        release_id="pilot_wr01_wx01_v1",
    )

    assert current == locked
