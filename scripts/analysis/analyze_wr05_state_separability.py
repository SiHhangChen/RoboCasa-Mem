#!/usr/bin/env python3
"""State-only separability audit for WR05's two remembered shelf levels.

Boundary records provide timing and semantic labels only.  All state values
are loaded from the raw LeRobot parquet files, resolving the recorded seed
through ``meta/seeds.jsonl`` because parquet filenames use episode_index.
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


def analyze_lower_upper(X: np.ndarray, labels: np.ndarray, n_perm: int) -> dict[str, Any]:
    result = analyze_point(X, labels, n_perm)
    # The shared helper uses WA02's left/right field names; WR05 labels are
    # remembered shelf levels instead.
    result["class_counts"] = {
        "lower": int(np.sum(labels == 0)),
        "upper": int(np.sum(labels == 1)),
    }
    return result


def analyze_label(
    records: list[dict[str, Any]],
    states: dict[int, np.ndarray],
    labels: np.ndarray,
    point_frames: dict[str, list[int]],
    n_perm: int,
) -> dict[str, Any]:
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
        "causal_offsets": {},
        "causal_scan": {},
    }
    for name, X in point_states.items():
        output["points"][name] = {
            "first_30": analyze_lower_upper(X[:, :30], labels, n_perm),
            "full_37": analyze_lower_upper(X, labels, n_perm),
        }

    handoff_frames = point_frames["return_handoff"]
    ordered_states = [states[int(row["seed"])] for row in records]
    for offset in (0, 1, 5, 10, 20, 40, 80, 120, 160, 200):
        valid = [i for i, (state, frame) in enumerate(zip(ordered_states, handoff_frames)) if frame + offset < len(state)]
        X = np.vstack([ordered_states[i][handoff_frames[i] + offset] for i in valid])
        yy = labels[valid]
        output["causal_offsets"][str(offset)] = {
            "n": len(valid),
            "frame_offset": offset,
            "first_30": analyze_lower_upper(X[:, :30], yy, n_perm),
            "full_37": analyze_lower_upper(X, yy, n_perm),
        }

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
    return output


def stage_frame(row: dict[str, Any], name: str, edge: str = "start") -> int:
    matches = [stage for stage in row["stages"] if stage["name"] == name]
    if len(matches) != 1:
        raise ValueError(f"seed {row['seed']}: expected one stage named {name!r}, found {len(matches)}")
    return int(matches[0][f"{edge}_frame"])


def stage_frame_prefix(row: dict[str, Any], prefix: str, edge: str = "start") -> int:
    matches = [stage for stage in row["stages"] if stage["name"].startswith(prefix)]
    if len(matches) != 1:
        raise ValueError(f"seed {row['seed']}: expected one stage with prefix {prefix!r}, found {len(matches)}")
    return int(matches[0][f"{edge}_frame"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--boundaries", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--permutations", type=int, default=2000)
    args = parser.parse_args()

    records = sorted(read_jsonl(args.boundaries), key=lambda row: int(row["seed"]))
    if len(records) != 200 or len({int(row["seed"]) for row in records}) != len(records):
        raise ValueError(f"expected 200 unique WR05 boundary records, found {len(records)}")
    seed_to_episode = load_seed_mapping(args.raw_root)
    states = {
        int(row["seed"]): load_raw_state(args.raw_root, int(row["seed"]), int(row["recorded_total_frames"]), seed_to_episode)
        for row in records
    }

    # Lower=0, upper=1 is the remembered shelf level for each condiment.
    labels = {
        "ketchup": np.array([0 if row["home_assignment"]["ketchup"] == "lower" else 1 for row in records], dtype=np.int8),
        "mustard": np.array([0 if row["home_assignment"]["mustard"] == "lower" else 1 for row in records], dtype=np.int8),
    }
    # All points are diagnostics; only return_handoff is the pre-decision point.
    frame_functions = {
        "initial": lambda row: 0,
        "after_observe": lambda row: stage_frame(row, "observe_condiment_shelf", "end") - 1,
        "after_ketchup_shelf_pick": lambda row: stage_frame(row, "pick_ketchup_from_original_level", "end") - 1,
        "after_ketchup_plate_staged": lambda row: stage_frame_prefix(row, "place_ketchup_at_", "end") - 1,
        "after_mustard_shelf_pick": lambda row: stage_frame(row, "pick_mustard_from_original_level", "end") - 1,
        "after_both_staged": lambda row: stage_frame_prefix(row, "return_arm_after_mustard_plate_stage", "end") - 1,
        "ketchup_return_handoff": lambda row: stage_frame(row, "navigate_ketchup_to_remembered_level", "start"),
        "mustard_return_handoff": lambda row: stage_frame(row, "navigate_mustard_to_remembered_level", "start"),
    }

    output: dict[str, Any] = {
        "dataset": str(args.raw_root),
        "n_episodes": len(records),
        "state_dim": 37,
        "state_names": STATE_NAMES,
        "label_definition": "home_assignment level: lower=0, upper=1; separate ketchup and mustard",
        "raw_seed_mapping": {"n": len(seed_to_episode), "min_seed": min(seed_to_episode), "max_seed": max(seed_to_episode), "missing_seed": sorted(set(range(max(seed_to_episode) + 1)) - set(seed_to_episode))},
        "labels": {},
    }

    # Analyze each condiment at its own return handoff and at shared historical points.
    for condiment, y in labels.items():
        handoff_name = f"{condiment}_return_handoff"
        shared_names = ["initial", "after_observe", "after_ketchup_shelf_pick", "after_ketchup_plate_staged", "after_mustard_shelf_pick", "after_both_staged", handoff_name]
        point_frames = {name: [int(frame_functions[name](row)) for row in records] for name in shared_names}
        result = analyze_label(records, states, y, {"return_handoff": point_frames[handoff_name], **point_frames}, args.permutations)
        output["labels"][condiment] = result

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, ensure_ascii=True), encoding="utf-8")
    print(f"wrote {args.output}")
    for condiment, result in output["labels"].items():
        print(condiment, result["label_counts"])
        for name in ("initial", "after_both_staged", f"{condiment}_return_handoff"):
            for dims in ("first_30", "full_37"):
                m = result["points"][name][dims]
                print(name, dims, "max", m["max_single_dimension_accuracy"],
                      "strict", m["strictly_separable_dimensions"],
                      "perm_median", m["permutation"]["median"],
                      "linear", m["perfect_affine_linear_separation"]["separable"],
                      "cv", m["ten_fold_standardized_logistic_probe"]["mean_accuracy"])


if __name__ == "__main__":
    main()
