#!/usr/bin/env python
"""Recompute and summarize five comparable 50-seed runs of one task/policy."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import statistics
import sys
import tempfile
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.eval.compute_scr import METRIC_ID, compute_run_scr  # noqa: E402


def _statistics(values: list[float]) -> dict[str, Any]:
    return {"round_values": values, "mean": statistics.mean(values),
            "sample_standard_deviation": statistics.stdev(values)}


def summarize_rounds(run_dirs: list[Path]) -> dict[str, Any]:
    """Read original episodes, rejecting missing, duplicate, or mixed runs."""
    if len(run_dirs) != 5:
        raise ValueError("provide exactly five complete 50-seed run directories")
    paths = [path.expanduser().resolve() for path in run_dirs]
    if len(set(paths)) != 5:
        raise ValueError("the five run directories must be distinct")
    reports = [compute_run_scr(path) for path in paths]
    if len({report["run_id"] for report in reports}) != 5:
        raise ValueError("the five runs must have distinct run IDs")
    first = reports[0]
    # Ignore server IDs, timestamps and directory names, which naturally vary
    # between rounds. These fields define the experiment being repeated.
    fields = ("metric_id", "task_id", "profile_id", "stage_order", "policy",
              "checkpoint", "source_revision", "rollout", "seeds")
    for number, report in enumerate(reports[1:], 2):
        for field in fields:
            if report[field] != first[field]:
                raise ValueError(f"round {number}: {field} differs from round 1")
        if report["suite"] != first["suite"]:
            raise ValueError(f"round {number}: suite differs from round 1")
        for reference, episode in zip(first["episodes"], report["episodes"]):
            for stage in first["stage_order"]:
                if reference["stages"][stage]["applicable"] != episode["stages"][stage]["applicable"]:
                    raise ValueError(f"round {number}: stage applicability differs for seed {episode['seed']}")
    stages = {name: {"paper_stage": first["stages"][name]["paper_stage"],
        **_statistics([report["stages"][name]["rate"] for report in reports])}
        for name in first["stage_order"]}
    return {"schema_version": "robocasa-mem.five_round_scr/v1", "metric_id": METRIC_ID,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        **{field: first[field] for field in (
            "task_id", "task_name", "profile_id", "stage_order", "policy", "checkpoint",
            "source_revision", "rollout", "suite", "seeds")},
        "round_count": 5, "episode_runs": 250, "unique_environment_seeds": 50,
        "scr": _statistics([report["scr"] for report in reports]),
        "task_success_rate": _statistics([report["task_success_rate"] for report in reports]),
        "stages": stages,
        "rounds": [{"round": number, **{field: report[field] for field in (
            "run_id", "run_dir", "manifest_sha256", "scr", "scr_exact_fraction",
            "task_success_rate", "counts", "stages")},
            "episode_result_hashes": {str(ep["seed"]): ep["result_sha256"] for ep in report["episodes"]}}
            for number, report in enumerate(reports, 1)],
        "aggregation": "Compute paper SCR separately for each 50-seed run, then "
            "take the mean and sample standard deviation of the five scores (ddof=1). "
            "SR and per-stage rates use the same round-level aggregation. "
            "The 250 episode runs repeat 50 environment seeds; they are not 250 unique seeds."}


def write_summary(report: dict[str, Any], output_dir: Path) -> None:
    output = output_dir.expanduser().resolve()
    for rd in report["rounds"]:
        run = Path(rd["run_dir"])
        if output == run or run in output.parents:
            raise ValueError("summary output directory must be outside all five source run directories")

    def metric(value: dict[str, Any]) -> str:
        return f"{value['mean']:.2%} ± {value['sample_standard_deviation']*100:.2f} pp"

    lines = [f"# {report['task_id']} / {report['task_name']} — five rounds", "",
        "Five runs of the same 50 seeds: 250 episode runs, 50 unique environment seeds.", "",
        "Mean ± sample standard deviation (ddof=1); standard deviations in percentage points.", "",
        f"**SCR: {metric(report['scr'])}; SR: {metric(report['task_success_rate'])}.**", "",
        "| Round | Run ID | SCR | SR |", "|---|---|---:|---:|"]
    for rd in report["rounds"]:
        lines.append(f"| {rd['round']} | `{rd['run_id']}` | {rd['scr']:.2%} | {rd['task_success_rate']:.2%} |")
    lines += ["", "| Stage | R1 | R2 | R3 | R4 | R5 | Mean ± SD |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for name in report["stage_order"]:
        stage = report["stages"][name]
        rates = " | ".join(f"{value:.2%}" for value in stage["round_values"])
        lines.append(f"| {stage['paper_stage']} (`{name}`) | {rates} | {metric(stage)} |")
    lines += ["", report["aggregation"], "",
        "Metrics are recomputed from original episode results, not cached scr.json files.", ""]
    files = {"SUMMARY.md": "\n".join(lines),
             "summary.json": json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n"}
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".rounds-", dir=output) as temporary:
        for name, content in files.items():
            (Path(temporary) / name).write_text(content, encoding="utf-8")
        for name in files:
            os.replace(Path(temporary) / name, output / name)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dirs", type=Path, nargs=5, help="Round 1 through round 5 run directories")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for the five-round summary")
    args = parser.parse_args(argv)
    try:
        report = summarize_rounds(args.run_dirs)
        write_summary(report, args.output_dir)
    except (OSError, ValueError, TypeError, KeyError) as exc:
        print(f"Five-round summary error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"task_id": report["task_id"], "round_count": 5,
        "scr": report["scr"], "task_success_rate": report["task_success_rate"],
        "output_dir": str(args.output_dir.expanduser().resolve())}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
