"""Recorded suite settings are authoritative, including family-specific budgets."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import pytest

from bench.evaluation import suite as suite_module
from bench.evaluation.suite import load_eval_suite
from bench.tasks import resolve_builtin_task_plugin

# Resources belong to the implementation under test, which may be an installed wheel.
ROOT = Path(suite_module.__file__).resolve().parents[2]
SOURCES = json.loads((ROOT / "configs/benchmark/evaluation_suite_sources.json").read_text())
ENTRIES = SOURCES["entries"]


def test_nine_tasks_have_three_recorded_model_families():
    expected = {"WX-01", "WA-01", "WA-05", "WR-01", "WR-08", "TS02", "TS03", "TM01", "TM02"}
    assert len(ENTRIES) == 27
    for family in ("pi05", "act", "diffusion"):
        entries = [e for e in ENTRIES if e["family"] == family]
        assert len(entries) == 9
        assert {e["task_id"] for e in entries} == expected


@pytest.mark.parametrize("entry", ENTRIES, ids=lambda e: f"{e['family']}-{e['task_id']}")
def test_copied_suite_matches_recorded_experiment(entry):
    path = ROOT / entry["suite_file"]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == entry["file_sha256"]
    suite = load_eval_suite(path)
    assert suite.content_hash == entry["normalized_suite_hash"]
    assert suite.task_id == entry["task_id"]
    assert asdict(suite.rollout) == entry["effective_rollout"]
    assert suite.seeds == tuple(range(1050, 1100))
    assert suite.env_split == "pretrain"
    plugin = resolve_builtin_task_plugin(suite.task_id)
    assert (
        ROOT / suite.task_options["task_config_file"]
    ).resolve() == plugin.default_task_config.resolve()
    assert suite.artifacts.video_cameras == ("agentview_left", "agentview_right", "eye_in_hand")
    assert suite.artifacts.video_fps == 20


def test_model_families_preserve_distinct_execution_settings():
    settings = {(e["family"], e["task_id"]): e["effective_rollout"] for e in ENTRIES}
    assert settings["pi05", "WA-01"]["max_steps"] == 2000
    assert settings["act", "WA-01"]["max_steps"] == 5000
    assert all(
        e["effective_rollout"]["replan_steps"] == 1 for e in ENTRIES if e["family"] == "diffusion"
    )
    assert all(
        e["effective_rollout"]["replan_steps"] == 40 for e in ENTRIES if e["family"] == "pi05"
    )
