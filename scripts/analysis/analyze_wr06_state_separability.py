#!/usr/bin/env python3
"""Audit WR06 label separability from raw LeRobot state.

The semantic boundary records are used only for labels and frame locations.
State values are loaded from the raw WR06 parquet files.  The actual task
label is the original shelf level (lower/upper); the plate side is retained
only as a separate diagnostic because it is not the requested answer.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.analysis.analyze_wa02_state_separability import (
    STATE_NAMES,
    analyze_point,
    dimension_metrics,
    perfect_linear_separation,
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_seed_mapping(raw_root: Path) -> dict[int, int]:
    rows = read_jsonl(raw_root / "meta" / "seeds.jsonl")
    mapping = {int(row["seed"]): int(row["episode_index"]) for row in rows}
    if len(mapping) != len(rows) or sorted(mapping.values()) != list(range(len(rows))):
        raise ValueError("raw seed mapping is not one-to-one over episode indices")
    return mapping


def load_raw_state(raw_root: Path, seed: int, expected_length: int, seed_to_episode: dict[int, int]) -> np.ndarray:
    episode_index = seed_to_episode[seed]
    path = raw_root / "data" / "chunk-000" / f"file-{episode_index:03d}.parquet"
    table = pq.read_table(path, columns=["observation.state"])
    values = table["observation.state"].combine_chunks().values.to_numpy()
    state = np.asarray(values, dtype=np.float64).reshape(-1, 37)
    if len(state) != expected_length:
        raise ValueError(f"seed {seed}: raw rows {len(state)} != boundary length {expected_length}")
    return state


def stage_frame(row: dict[str, Any], index: int, edge: str = "start") -> int:
    return int(row["stages"][index][f"{edge}_frame"])


def analyze_label(
    records: list[dict[str, Any]],
    states: dict[int, np.ndarray],
    labels: np.ndarray,
    point_frames: dict[str, list[int]],
    n_perm: int,
) -> dict[str, Any]:
    ordered_states = [states[int(row["seed"])] for row in records]
    point_states = {
        name: np.vstack([states[int(row["seed"])][frame] for row, frame in zip(records, frames)])
        for name, frames in point_frames.items()
    }
    output: dict[str, Any] = {
        "n": int(len(labels)),
        "label_counts": {"lower": int(np.sum(labels == 0)), "upper": int(np.sum(labels == 1))},
        "boundary_frames": {
            name: {"min": min(frames), "max": max(frames), "mean": float(np.mean(frames))}
            for name, frames in point_frames.items()
        },
        "points": {},
        "pre_handoff_scan": {},
        "causal_offsets": {},
        "causal_scan": {},
    }
    for name, X in point_states.items():
        output["points"][name] = {
            "first_30": analyze_point(X[:, :30], labels, n_perm),
            "full_37": analyze_point(X, labels, n_perm),
        }

    handoff_frames = point_frames["return_handoff"]
    # Scan forward from the fixed start of original-level navigation.  Unlike
    # the post-handoff scan, this locates the first appearance of the answer
    # before the final return decision.  No permutation tests are spent here.
    observe_frames = point_frames["observe_shelf_handoff"]
    for offset in range(0, 101):
        valid = [i for i, (state, frame) in enumerate(zip(ordered_states, observe_frames)) if frame + offset < len(state)]
        X = np.vstack([ordered_states[i][observe_frames[i] + offset] for i in valid])
        yy = labels[valid]
        metrics = dimension_metrics(X[:, :30], yy)
        top = max(metrics["dimensions"], key=lambda row: row["accuracy"])
        output["pre_handoff_scan"][str(offset)] = {
            "n": len(valid),
            "max_single_dimension_accuracy": metrics["max_accuracy"],
            "top_dimension": {"index": top["index"], "name": top["name"], "accuracy": top["accuracy"]},
            "strictly_separable_dimensions": metrics["strictly_separable_indices"],
            "perfect_affine_linear_separation": perfect_linear_separation(X[:, :30], yy),
        }

    for offset in (0, 1, 5, 10, 20, 40, 80, 120, 160, 200):
        valid = [i for i, (state, frame) in enumerate(zip(ordered_states, handoff_frames)) if frame + offset < len(state)]
        X = np.vstack([ordered_states[i][handoff_frames[i] + offset] for i in valid])
        yy = labels[valid]
        output["causal_offsets"][str(offset)] = {
            "n": len(valid),
            "frame_offset": offset,
            "first_30": analyze_point(X[:, :30], yy, n_perm),
            "full_37": analyze_point(X, yy, n_perm),
        }

    # A dense post-handoff scan is diagnostic only.  It identifies the first
    # offset where target-specific navigation makes the label visible.
    for offset in range(0, 201):
        valid = [i for i, (state, frame) in enumerate(zip(ordered_states, handoff_frames)) if frame + offset < len(state)]
        X = np.vstack([ordered_states[i][handoff_frames[i] + offset] for i in valid])
        yy = labels[valid]
        output["causal_scan"][str(offset)] = {}
        for dims, XX in (("first_30", X[:, :30]), ("full_37", X)):
            metrics = dimension_metrics(XX, yy)
            top = max(metrics["dimensions"], key=lambda row: row["accuracy"])
            output["causal_scan"][str(offset)][dims] = {
                "n": len(valid),
                "max_single_dimension_accuracy": metrics["max_accuracy"],
                "top_dimension": {"index": top["index"], "name": top["name"], "accuracy": top["accuracy"]},
                "strictly_separable_dimensions": metrics["strictly_separable_indices"],
                "perfect_affine_linear_separation": perfect_linear_separation(XX, yy),
            }
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--boundaries", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--permutations", type=int, default=2000)
    args = parser.parse_args()

    records = sorted(read_jsonl(args.boundaries), key=lambda row: int(row["seed"]))
    if len(records) != 200 or len({int(row["seed"]) for row in records}) != len(records):
        raise ValueError(f"expected 200 unique WR06 boundary records, found {len(records)}")
    seed_to_episode = load_seed_mapping(args.raw_root)
    states = {
        int(row["seed"]): load_raw_state(args.raw_root, int(row["seed"]), int(row["recorded_total_frames"]), seed_to_episode)
        for row in records
    }

    labels = np.array([0 if row["original_level"] == "lower" else 1 for row in records], dtype=np.int8)
    plate_labels = np.array([0 if row["plate_side"] == "left" else 1 for row in records], dtype=np.int8)

    # Stage 12 start is the pre-return navigation handoff.  Earlier points
    # locate when the original-level branch first becomes visible in state.
    frame_functions = {
        "initial": lambda row: 0,
        "observe_shelf_handoff": lambda row: stage_frame(row, 1, "start"),
        "after_original_navigation": lambda row: stage_frame(row, 2, "start"),
        "after_shelf_pick": lambda row: stage_frame(row, 3, "start"),
        "after_shelf_branch": lambda row: stage_frame(row, 4, "start"),
        "after_transport_posture": lambda row: stage_frame(row, 5, "start"),
        "after_plate_return": lambda row: stage_frame(row, 8, "start"),
        "return_handoff": lambda row: stage_frame(row, 12, "start"),
    }
    point_frames = {name: [int(fn(row)) for row in records] for name, fn in frame_functions.items()}

    output: dict[str, Any] = {
        "dataset": str(args.raw_root),
        "n_episodes": len(records),
        "state_dim": 37,
        "model_state_dim": 30,
        "state_names": STATE_NAMES,
        "label_definition": "original_level: lower=0, upper=1; plate_side is diagnostic only",
        "label_counts": {"lower": int(np.sum(labels == 0)), "upper": int(np.sum(labels == 1))},
        "plate_side_counts": {"left": int(np.sum(plate_labels == 0)), "right": int(np.sum(plate_labels == 1))},
        "level_plate_crosstab": {
            level: {side: int(np.sum((labels == li) & (plate_labels == si))) for side, si in (("left", 0), ("right", 1))}
            for level, li in (("lower", 0), ("upper", 1))
        },
        "raw_seed_mapping": {
            "n": len(seed_to_episode),
            "min_seed": min(seed_to_episode),
            "max_seed": max(seed_to_episode),
            "missing_seed": sorted(set(range(max(seed_to_episode) + 1)) - set(seed_to_episode)),
        },
        "boundary_frames": {
            name: {"min": min(frames), "max": max(frames), "mean": float(np.mean(frames))}
            for name, frames in point_frames.items()
        },
        "labels": {},
    }
    output["labels"]["original_level"] = analyze_label(records, states, labels, point_frames, args.permutations)
    output["labels"]["plate_side_diagnostic"] = analyze_label(records, states, plate_labels, point_frames, args.permutations)

    # Make the finite-sample permutation p-value explicit in addition to the
    # shared helper's quantiles and count of maxima at least as large.
    for result in output["labels"].values():
        for point in result["points"].values():
            for metrics in point.values():
                p = metrics["permutation"]
                p["empirical_p_ge_actual_plus1"] = (p["n_ge_actual"] + 1) / (p["n"] + 1)
        for point in result["causal_offsets"].values():
            for metrics in (point["first_30"], point["full_37"]):
                p = metrics["permutation"]
                p["empirical_p_ge_actual_plus1"] = (p["n_ge_actual"] + 1) / (p["n"] + 1)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, ensure_ascii=True), encoding="utf-8")
    print(f"wrote {args.output}")
    for label_name, result in output["labels"].items():
        print(label_name, result["label_counts"])
        for point_name in ("initial", "observe_shelf_handoff", "after_original_navigation", "after_shelf_pick", "after_shelf_branch", "after_plate_return", "return_handoff"):
            m = result["points"][point_name]["first_30"]
            print(point_name, "max", m["max_single_dimension_accuracy"],
                  "strict", m["strictly_separable_dimensions"],
                  "perm_median", m["permutation"]["median"],
                  "perm_q975", m["permutation"]["q975"],
                  "p_ge", m["permutation"]["empirical_p_ge_actual_plus1"],
                  "linear", m["perfect_affine_linear_separation"]["separable"],
                  "cv", m["ten_fold_standardized_logistic_probe"]["mean_accuracy"])
        print("causal offsets", {k: result["causal_offsets"][k]["first_30"]["max_single_dimension_accuracy"] for k in ("0", "1", "5", "10", "20", "40", "80", "120", "160", "200")})


if __name__ == "__main__":
    main()
