#!/usr/bin/env python3
"""Summarize the ordered WR-01 eight-stage evaluation funnel.

Stages are read from ``diagnostics.stages.*.qualified`` so an observed stage
that skipped a prerequisite is not counted as completed. The loader accepts
both standard rollout directories (``episodes/*/result.json``) and the native
WR-01 replay format (``summary.json`` containing a ``results`` list).
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
EVAL_ROOT = ROOT / "outputs" / "eval"
STAGES = (
    "first_can_pick",
    "first_can_store",
    "second_can_pick",
    "second_can_store",
    "close_upper_cabinet",
    "close_lower_cabinet",
    "open_cabinet",
    "retrieve_item",
)
STAGE_LABELS = {
    "first_can_pick": "第一罐头取",
    "first_can_store": "第一罐头存",
    "second_can_pick": "第二罐头取",
    "second_can_store": "第二罐头存",
    "close_upper_cabinet": "关上柜",
    "close_lower_cabinet": "关下柜",
    "open_cabinet": "开柜",
    "retrieve_item": "取物",
}


def _manifest(run: Path) -> dict[str, Any]:
    for name in ("run_manifest.json", "manifest.json"):
        path = run / name
        if path.is_file():
            return json.loads(path.read_text(encoding="utf-8"))
    return {}


def load_episodes(run: Path) -> list[dict[str, Any]]:
    episodes_dir = run / "episodes"
    if episodes_dir.is_dir():
        rows: list[dict[str, Any]] = []
        for path in sorted(episodes_dir.glob("*/result.json")):
            rows.append(json.loads(path.read_text(encoding="utf-8")))
        return rows

    summary_path = run / "summary.json"
    if summary_path.is_file():
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        results = payload.get("results", [])
        if isinstance(results, list):
            return [dict(value) for value in results if isinstance(value, dict)]
    return []


def stage_hits(episode: dict[str, Any]) -> dict[str, bool]:
    diagnostics = episode.get("diagnostics", {}) or {}
    stages = diagnostics.get("stages", {}) or {}
    if isinstance(stages, dict) and all(
        isinstance(stages.get(stage), dict) for stage in STAGES
    ):
        hits = {
            stage: bool(stages[stage].get("qualified")) for stage in STAGES
        }
    else:
        first_steps = (
            diagnostics.get("physical_events", {}) or {}
        ).get("stage_first_steps", {})
        hits = {stage: stage in first_steps for stage in STAGES}

    hits["retrieve_item"] = bool(
        hits["retrieve_item"]
        and episode.get("success") is True
        and diagnostics.get("success") is True
    )
    return hits


def episode_row(run: Path, episode: dict[str, Any]) -> dict[str, Any]:
    diagnostics = episode.get("diagnostics", {}) or {}
    physical = diagnostics.get("physical_events", {}) or {}
    history = physical.get("phase_history", []) or []
    hits = stage_hits(episode)
    return {
        "run_id": run.name,
        "seed": episode.get("seed", ""),
        "status": episode.get("status", "success" if episode.get("success") else "failure"),
        "success": bool(episode.get("success") is True and hits["retrieve_item"]),
        "failure_reason": episode.get("failure_reason")
        or diagnostics.get("failure_code")
        or "",
        "replayed_steps": episode.get("replayed_steps", episode.get("env_steps", "")),
        "final_phase": physical.get("phase", ""),
        "phase_history": " -> ".join(
            f"{entry.get('phase', '?')}@{entry.get('step', '?')}"
            for entry in history
            if isinstance(entry, dict)
        ),
        **{f"stage_{stage}": hits[stage] for stage in STAGES},
    }


def summarize_run(run: Path) -> dict[str, Any]:
    episodes = load_episodes(run)
    counts = {"success": 0, "failure": 0, "error": 0, "cancelled": 0}
    hits = {stage: 0 for stage in STAGES}
    valid = 0
    for episode in episodes:
        status = str(episode.get("status", "success" if episode.get("success") else "failure"))
        counts[status] = counts.get(status, 0) + 1
        if status not in ("success", "failure"):
            continue
        valid += 1
        for stage, hit in stage_hits(episode).items():
            hits[stage] += int(hit)

    manifest = _manifest(run)
    planned = manifest.get("seeds", []) or []
    return {
        "run_id": run.name,
        "task_id": manifest.get("task_id", "WR-01"),
        "evaluator_profile": (
            manifest.get("replay", {}) or {}
        ).get("evaluator_profile", "wr01-eight-stage-v1"),
        "n_planned": len(planned),
        "n_episodes": len(episodes),
        "n_ok": valid,
        "n_success": counts.get("success", 0),
        "n_failure": counts.get("failure", 0),
        "n_error": counts.get("error", 0),
        "n_cancelled": counts.get("cancelled", 0),
        **{f"{stage}_n": hits[stage] for stage in STAGES},
        **{
            f"{stage}_pct": round(100.0 * hits[stage] / valid, 1) if valid else 0.0
            for stage in STAGES
        },
        "created_at": manifest.get("created_at", ""),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="*", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=EVAL_ROOT / "wr01_stage_summary.csv",
    )
    parser.add_argument(
        "--episodes-output",
        type=Path,
        default=EVAL_ROOT / "wr01_episode_stage_results.csv",
    )
    args = parser.parse_args()
    runs = [
        path.expanduser().resolve()
        for path in (args.runs or sorted(EVAL_ROOT.glob("wr01-*-final")))
        if path.is_dir()
    ]

    summaries = [summarize_run(run) for run in runs]
    fields = [
        "run_id",
        "task_id",
        "evaluator_profile",
        "n_planned",
        "n_episodes",
        "n_ok",
        "n_success",
        "n_failure",
        "n_error",
        "n_cancelled",
    ]
    for stage in STAGES:
        fields.extend((f"{stage}_n", f"{stage}_pct"))
    fields.append("created_at")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(summaries)

    episode_rows = [
        episode_row(run, episode)
        for run in runs
        for episode in load_episodes(run)
    ]
    episode_fields = [
        "run_id",
        "seed",
        "status",
        "success",
        "failure_reason",
        "replayed_steps",
        "final_phase",
        "phase_history",
        *[f"stage_{stage}" for stage in STAGES],
    ]
    args.episodes_output.parent.mkdir(parents=True, exist_ok=True)
    with args.episodes_output.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=episode_fields,
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(episode_rows)

    print("WR-01 分阶段达成率（error/cancelled 不进分母）")
    print("run", "n_ok", *[STAGE_LABELS[stage] for stage in STAGES], sep="\t")
    for row in summaries:
        print(
            row["run_id"],
            row["n_ok"],
            *[f"{row[f'{stage}_pct']:.1f}" for stage in STAGES],
            sep="\t",
        )
    print(f"已写入 {args.output}")
    print(f"已写入 {args.episodes_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
