#!/usr/bin/env python
"""Rank pi0.5 TS02 checkpoints using offline flow-loss validation."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_INPUT = ROOT / "outputs" / "eval" / "offline"
EXPECTED_SEEDS = (500, 501, 502, 503, 504)
EXPECTED_FRAMES = 6391
EXPECTED_HORIZON = 50


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=DEFAULT_INPUT / "ts02-pi05-val5-checkpoint-summary.json",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=DEFAULT_INPUT / "ts02-pi05-val5-checkpoint-summary.csv",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = sorted(
        args.input_dir.glob("ts02-pi05-lora-*-val5.json"),
        key=lambda path: int(path.stem.split("-")[-2]),
    )
    if not paths:
        raise FileNotFoundError(f"No TS02 offline results found in {args.input_dir}")
    records = [_record(path) for path in paths]
    ranking = sorted(records, key=lambda record: record["mean"])
    rank_by_step = {record["step"]: rank for rank, record in enumerate(ranking, start=1)}
    for record in records:
        record["rank_by_mean"] = rank_by_step[record["step"]]

    payload = {
        "schema_version": "membench.ts02_checkpoint_summary/v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "metric": "teacher-forced flow-matching MSE in normalized 32D model action space",
        "distribution_unit": "one action-horizon position, averaged over 32 action dimensions",
        "expected_frames": EXPECTED_FRAMES,
        "expected_horizon": EXPECTED_HORIZON,
        "checkpoints": records,
        "best_by_mean": ranking[0],
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_csv(args.output_csv, records)

    print(f"checkpoints={len(records)}")
    print(f"best_step={ranking[0]['step']} best_mean={ranking[0]['mean']:.10f}")
    print(f"json={args.output_json.resolve()}")
    print(f"csv={args.output_csv.resolve()}")


def _record(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "complete":
        raise ValueError(f"Incomplete result: {path}")
    flow = payload["flow_matching_loss"]
    overall = flow["overall"]
    required_stats = ("mean", "median", "p90", "p95", "p99", "variance", "std", "min", "max")
    missing = [key for key in required_stats if key not in overall]
    if missing:
        raise ValueError(f"Result lacks {missing}: {path}")
    if flow["frames"] != EXPECTED_FRAMES or flow["horizon"] != EXPECTED_HORIZON:
        raise ValueError(f"Unexpected coverage in {path}: {flow['frames']}x{flow['horizon']}")
    if overall["count"] != EXPECTED_FRAMES * EXPECTED_HORIZON:
        raise ValueError(f"Unexpected loss count in {path}: {overall['count']}")

    episode_rows = {int(row["seed"]): row for row in flow["by_episode"]}
    if tuple(sorted(episode_rows)) != EXPECTED_SEEDS:
        raise ValueError(f"Unexpected seeds in {path}: {sorted(episode_rows)}")
    numeric_values = [float(overall[key]) for key in required_stats]
    numeric_values.extend(float(episode_rows[seed]["mean"]) for seed in EXPECTED_SEEDS)
    if not all(math.isfinite(value) for value in numeric_values):
        raise ValueError(f"Non-finite loss statistic in {path}")
    if not math.isclose(float(overall["variance"]), float(overall["std"]) ** 2, rel_tol=1e-9, abs_tol=1e-15):
        raise ValueError(f"Variance/std mismatch in {path}")

    step = int(payload["checkpoint"]["step"])
    return {
        "step": step,
        "rank_by_mean": None,
        "count": int(overall["count"]),
        "mean": float(overall["mean"]),
        "median": float(overall["median"]),
        "p90": float(overall["p90"]),
        "p95": float(overall["p95"]),
        "p99": float(overall["p99"]),
        "variance": float(overall["variance"]),
        "std": float(overall["std"]),
        "min": float(overall["min"]),
        "max": float(overall["max"]),
        "seed_500_mean": float(episode_rows[500]["mean"]),
        "seed_501_mean": float(episode_rows[501]["mean"]),
        "seed_502_mean": float(episode_rows[502]["mean"]),
        "seed_503_mean": float(episode_rows[503]["mean"]),
        "seed_504_mean": float(episode_rows[504]["mean"]),
        "runtime_seconds": float(payload["runtime"]["seconds"]),
        "loss_values_path": payload["settings"]["loss_values_output"],
        "result_path": str(path.resolve()),
    }


def _write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


if __name__ == "__main__":
    main()
