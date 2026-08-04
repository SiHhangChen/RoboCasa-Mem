from __future__ import annotations

import json
from pathlib import Path

import pytest

from bench.release import build_release_lock, resolve_release_entry, verify_release_lock
from bench.tasks import list_builtin_task_plugins


ROOT = Path(__file__).resolve().parents[2]
LOCK_FILE = ROOT / "configs" / "benchmark" / "release_locks" / "all_tasks_v1.json"


@pytest.mark.release_gate
def test_all_task_release_lock_matches_current_resolved_inputs() -> None:
    locked = json.loads(LOCK_FILE.read_text(encoding="utf-8"))
    verify_release_lock(locked)
    current = build_release_lock(
        [
            resolve_release_entry(plugin.canonical_id, seed=0)
            for plugin in list_builtin_task_plugins()
        ],
        release_id="all_tasks_v1",
    )
    assert current == locked
    assert {item["task_id"] for item in locked["entries"]} == {
        plugin.canonical_id for plugin in list_builtin_task_plugins()
    }


@pytest.mark.release_gate
def test_full_release_lock_exposes_tm03_production_blocker() -> None:
    locked = json.loads(LOCK_FILE.read_text(encoding="utf-8"))
    by_task = {item["task_id"]: item for item in locked["entries"]}

    assert all(not item["allow_partial_calibration"] for item in by_task.values())
    assert by_task["TM03"]["release_facts"]["missing_physical_calibration"]
    assert all(
        not item["release_facts"].get("missing_physical_calibration")
        for task_id, item in by_task.items()
        if task_id != "TM03"
    )
