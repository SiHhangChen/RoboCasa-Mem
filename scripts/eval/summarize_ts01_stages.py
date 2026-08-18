#!/usr/bin/env python3
"""Summarize TS01 evaluator stages overall and by deterministic K/N cell."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
EVAL_ROOT = ROOT / "outputs" / "eval"
STAGES = ("sugar_1", "sugar_2", "sugar_3", "button", "success")


def load_episodes(run: Path) -> list[dict[str, Any]]:
    episodes = run / "episodes"
    if episodes.is_dir():
        return [
            json.loads(path.read_text(encoding="utf-8"))
            for path in sorted(episodes.glob("*/result.json"))
        ]
    summary = run / "summary.json"
    if summary.is_file():
        payload = json.loads(summary.read_text(encoding="utf-8"))
        values = payload.get("results", [])
        if isinstance(values, list):
            return [dict(value) for value in values if isinstance(value, dict)]
    return []


def _diagnostics(episode: dict[str, Any]) -> dict[str, Any]:
    value = episode.get("diagnostics", {}) or {}
    return dict(value) if isinstance(value, dict) else {}


def cell_id(episode: dict[str, Any]) -> str | None:
    diagnostics = _diagnostics(episode)
    try:
        target = int(diagnostics["target_sugar_count"])
        total = int(diagnostics["total_sugar_count"])
    except (KeyError, TypeError, ValueError):
        return None
    if target not in (1, 2, 3) or total < target or total > 5:
        return None
    return f"k{target}_n{total}"


def stage_hits(
    episode: dict[str, Any],
    *,
    field: str,
) -> dict[str, bool]:
    stages = _diagnostics(episode).get("stages", {}) or {}
    if not isinstance(stages, dict):
        return {stage: False for stage in STAGES}
    return {
        stage: bool(stages.get(stage, {}).get(field))
        if isinstance(stages.get(stage), dict)
        else False
        for stage in STAGES
    }


def episode_success(episode: dict[str, Any]) -> bool:
    diagnostics = _diagnostics(episode)
    qualified = stage_hits(episode, field="qualified")["success"]
    return bool(
        episode.get("success") is True
        and diagnostics.get("success") is True
        and qualified
    )


def aggregate(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    rows = [
        episode
        for episode in episodes
        if episode.get("status") in {"success", "failure"}
    ]
    qualified = {stage: 0 for stage in STAGES}
    effective = {stage: 0 for stage in STAGES}
    failures: Counter[str] = Counter()
    successes = 0
    for episode in rows:
        successes += int(episode_success(episode))
        for stage, hit in stage_hits(episode, field="qualified").items():
            qualified[stage] += int(hit)
        for stage, hit in stage_hits(episode, field="effective_success").items():
            effective[stage] += int(hit)
        if not episode_success(episode):
            diagnostics = _diagnostics(episode)
            reason = (
                episode.get("failure_reason")
                or diagnostics.get("failure_code")
                or "unspecified"
            )
            failures[str(reason)] += 1

    count = len(rows)
    return {
        "episodes": count,
        "success_count": successes,
        "success_rate": successes / count if count else 0.0,
        "failure_reasons": dict(sorted(failures.items())),
        "stages": {
            stage: {
                "qualified_count": qualified[stage],
                "qualified_rate": qualified[stage] / count if count else 0.0,
                "effective_count": effective[stage],
                "effective_rate": effective[stage] / count if count else 0.0,
            }
            for stage in STAGES
        },
    }


def summarize_run(run: Path) -> dict[str, Any]:
    episodes = load_episodes(run)
    cells = sorted(
        {value for episode in episodes if (value := cell_id(episode)) is not None}
    )
    return {
        "schema_version": "membench.ts01_stage_summary/v1",
        "run_id": run.name,
        "qualification_source": "diagnostics.stages.<stage>.qualified",
        "effective_source": "diagnostics.stages.<stage>.effective_success",
        "cell_source": "diagnostics.target_sugar_count/total_sugar_count",
        "overall": aggregate(episodes),
        "cells": {
            cell: aggregate(
                [episode for episode in episodes if cell_id(episode) == cell]
            )
            for cell in cells
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    run = args.run.expanduser().resolve()
    output = (
        args.output.expanduser().resolve()
        if args.output is not None
        else run / "stage_summary.json"
    )
    summary = summarize_run(run)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"stage_summary={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
