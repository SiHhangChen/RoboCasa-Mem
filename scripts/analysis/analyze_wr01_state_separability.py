#!/usr/bin/env python3
"""State-only separability audit for WR01's remembered cabinet level.

The raw WR01 dataset uses episode_index for parquet file names while the
seed values skip failed generation attempts.  This script therefore resolves
seed -> episode_index through raw ``meta/seeds.jsonl`` before loading state.
Boundary records provide timing and semantic labels only; state is read from
the raw parquet files.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq

# Allow direct execution from the repository root or from any working dir.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.analysis.analyze_wa02_state_separability import (
    STATE_NAMES,
    analyze_point,
    dimension_metrics,
    perfect_linear_separation,
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_raw_state(raw_root: Path, seed: int, expected_length: int, seed_to_episode: dict[int, int]) -> np.ndarray:
    if seed not in seed_to_episode:
        raise ValueError(f"seed {seed} is absent from raw meta/seeds.jsonl")
    episode_index = seed_to_episode[seed]
    path = raw_root / "data" / "chunk-000" / f"file-{episode_index:03d}.parquet"
    table = pq.read_table(path, columns=["observation.state"])
    values = table["observation.state"].combine_chunks().values.to_numpy()
    state = np.asarray(values, dtype=np.float64).reshape(-1, 37)
    if len(state) != expected_length:
        raise ValueError(f"seed {seed}: raw rows {len(state)} != boundary length {expected_length}")
    return state


def load_seed_mapping(raw_root: Path) -> dict[int, int]:
    rows = read_jsonl(raw_root / "meta" / "seeds.jsonl")
    mapping = {int(row["seed"]): int(row["episode_index"]) for row in rows}
    if len(mapping) != len(rows) or sorted(mapping.values()) != list(range(len(rows))):
        raise ValueError("raw seed mapping is not one-to-one over episode indices")
    return mapping


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--boundaries", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--permutations", type=int, default=2000)
    args = parser.parse_args()

    records = sorted(read_jsonl(args.boundaries), key=lambda row: int(row["seed"]))
    if len(records) != 200 or len({int(row["seed"]) for row in records}) != len(records):
        raise ValueError(f"expected 200 unique WR01 boundary records, found {len(records)}")
    seed_to_episode = load_seed_mapping(args.raw_root)
    states = {
        int(row["seed"]): load_raw_state(args.raw_root, int(row["seed"]), int(row["total_frames"]), seed_to_episode)
        for row in records
    }

    # lower=0 and upper=1.  This is the level of the requested can, not its
    # color, and is the actual final cabinet-door choice.
    labels = np.array([0 if row["target_level"] == "lower" else 1 for row in records], dtype=np.int8)
    point_frames: dict[str, list[int]] = {}
    point_states: dict[str, np.ndarray] = {}
    frame_functions = {
        "initial": lambda row: 0,
        "after_first_can_stored": lambda row: int(row["stages"][4]["end_frame"]) - 1,
        "after_second_can_stored": lambda row: int(row["stages"][11]["end_frame"]) - 1,
        "after_upper_door_closed": lambda row: int(row["stages"][15]["end_frame"]) - 1,
        "after_lower_door_closed": lambda row: int(row["stages"][18]["end_frame"]) - 1,
        "after_lower_close_return": lambda row: int(row["stages"][19]["end_frame"]) - 1,
        "final_selection_handoff": lambda row: int(row["stages"][21]["start_frame"]),
        "retrieval_open_start": lambda row: int(row["stages"][22]["start_frame"]),
        "physical_retrieval_start": lambda row: int(row["stages"][26]["start_frame"]),
    }
    for name, frame_fn in frame_functions.items():
        frames = [int(frame_fn(row)) for row in records]
        point_frames[name] = frames
        point_states[name] = np.vstack([states[int(row["seed"])][frame] for row, frame in zip(records, frames)])

    output: dict[str, Any] = {
        "dataset": str(args.raw_root),
        "n_episodes": len(records),
        "state_dim": 37,
        "state_names": STATE_NAMES,
        "label_definition": "target_level: lower=0, upper=1",
        "label_counts": {"lower": int(np.sum(labels == 0)), "upper": int(np.sum(labels == 1))},
        "raw_seed_mapping": {"n": len(seed_to_episode), "min_seed": min(seed_to_episode), "max_seed": max(seed_to_episode)},
        "boundary_frames": {
            name: {"min": min(frames), "max": max(frames), "mean": float(np.mean(frames))}
            for name, frames in point_frames.items()
        },
        "points": {},
        "causal_offsets": {},
        "causal_scan": {},
    }
    for name, X in point_states.items():
        output["points"][name] = {
            "first_30": analyze_point(X[:, :30], labels, args.permutations),
            "full_37": analyze_point(X, labels, args.permutations),
        }

    handoff_frames = point_frames["final_selection_handoff"]
    ordered_states = [states[int(row["seed"])] for row in records]
    for offset in (0, 1, 5, 10, 20, 40, 80, 120, 160, 200):
        valid = [i for i, (state, frame) in enumerate(zip(ordered_states, handoff_frames)) if frame + offset < len(state)]
        X = np.vstack([ordered_states[i][handoff_frames[i] + offset] for i in valid])
        yy = labels[valid]
        output["causal_offsets"][str(offset)] = {
            "n": len(valid),
            "frame_offset": offset,
            "first_30": analyze_point(X[:, :30], yy, args.permutations),
            "full_37": analyze_point(X, yy, args.permutations),
        }

    # Dense post-handoff scan identifies the first frame at which the
    # target-specific navigation becomes visible in state.
    for offset in range(0, 201):
        valid = [i for i, (state, frame) in enumerate(zip(ordered_states, handoff_frames)) if frame + offset < len(state)]
        X = np.vstack([ordered_states[i][handoff_frames[i] + offset] for i in valid])
        yy = labels[valid]
        output["causal_scan"][str(offset)] = {}
        for dims, XX in (("first_30", X[:, :30]), ("full_37", X)):
            metrics = dimension_metrics(XX, yy)
            top = max(metrics["dimensions"], key=lambda row: row["accuracy"])
            output["causal_scan"][str(offset)][dims] = {
                "max_single_dimension_accuracy": metrics["max_accuracy"],
                "top_dimension": {"index": top["index"], "name": top["name"], "accuracy": top["accuracy"]},
                "strictly_separable_dimensions": metrics["strictly_separable_indices"],
                "perfect_affine_linear_separation": perfect_linear_separation(XX, yy),
            }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, ensure_ascii=True), encoding="utf-8")
    print(f"wrote {args.output}")
    for name, metrics in output["points"].items():
        result = metrics["full_37"]
        print(name, "full_37", "max", result["max_single_dimension_accuracy"],
              "strict", result["strictly_separable_dimensions"],
              "perm_median", result["permutation"]["median"],
              "linear", result["perfect_affine_linear_separation"]["separable"],
              "cv", result["ten_fold_standardized_logistic_probe"]["mean_accuracy"])


if __name__ == "__main__":
    main()
