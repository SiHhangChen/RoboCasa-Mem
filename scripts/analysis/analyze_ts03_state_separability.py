#!/usr/bin/env python3
"""State-only separability audit for TS03's remembered sugar count.

TS03 has K=1 or K=2 target sugar cubes.  Boundary metadata is used only to
locate semantic frames; state is loaded directly from the raw LeRobot parquet
dataset.  The policy-relevant handoff is after the first completed sugar
cycle, where the expert must continue for K=2 or stop for K=1.
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
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def load_seed_mapping(raw_root: Path) -> dict[int, int]:
    rows = read_jsonl(raw_root / "meta" / "seeds.jsonl")
    mapping = {int(row["seed"]): int(row["episode_index"]) for row in rows}
    if len(mapping) != len(rows):
        raise ValueError("raw seed mapping contains duplicate seeds")
    if sorted(mapping.values()) != list(range(len(mapping))):
        raise ValueError("raw seed mapping is not one-to-one over episode indices")
    return mapping


def load_raw_state(
    raw_root: Path,
    seed: int,
    expected_length: int,
    seed_to_episode: dict[int, int],
) -> np.ndarray:
    episode_index = seed_to_episode.get(seed)
    if episode_index is None:
        raise ValueError(f"seed {seed} is absent from raw meta/seeds.jsonl")
    path = raw_root / "data" / "chunk-000" / f"file-{episode_index:03d}.parquet"
    table = pq.read_table(path, columns=["observation.state"])
    values = table["observation.state"].combine_chunks().values.to_numpy()
    state = np.asarray(values, dtype=np.float64).reshape(-1, 37)
    if len(state) != expected_length:
        raise ValueError(
            f"seed {seed}: raw rows {len(state)} != boundary length {expected_length}"
        )
    return state


def first_stage(row: dict[str, Any], prefix: str, *, exact: bool = False) -> dict[str, Any]:
    matches = [stage for stage in row["stages"] if stage["name"].startswith(prefix)]
    if not matches or (exact and len(matches) != 1):
        raise ValueError(
            f"seed {row['seed']}: expected stage starting {prefix!r}, found {len(matches)}"
        )
    return matches[0]


def frame_points(records: list[dict[str, Any]]) -> dict[str, list[int]]:
    points: dict[str, list[int]] = {"initial": [0 for _ in records]}
    place_last: list[int] = []
    return_last: list[int] = []
    return_next: list[int] = []
    completion: list[int] = []
    for row in records:
        place = first_stage(row, "place_")
        ret = first_stage(row, "return_after_")
        nav = first_stage(row, "navigate_completion_button", exact=True)
        place_last.append(int(place["end_frame"]) - 1)
        return_last.append(int(ret["end_frame"]) - 1)
        return_next.append(int(ret["end_frame"]))
        completion.append(int(nav["start_frame"]))
    points.update(
        {
            "after_first_sugar_place": place_last,
            "after_first_sugar_return": return_last,
            "first_return_next_frame": return_next,
            "completion_handoff": completion,
        }
    )
    return points


def analyze_frame_set(
    X: np.ndarray, labels: np.ndarray, n_perm: int
) -> dict[str, Any]:
    return {
        "first_30": analyze_point(X[:, :30], labels, n_perm),
        "full_37": analyze_point(X, labels, n_perm),
    }


def scan_offsets(
    states: list[np.ndarray], handoff_frames: list[int], labels: np.ndarray
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for offset in (0, 1, 5, 10, 20, 40, 80, 120, 160):
        valid = [
            i
            for i, (state, frame) in enumerate(zip(states, handoff_frames))
            if frame + offset < len(state)
        ]
        X = np.vstack([states[i][handoff_frames[i] + offset] for i in valid])
        y = labels[valid]
        output[str(offset)] = {
            "n": len(valid),
            "frame_offset": offset,
            "first_30": analyze_frame_set(X, y, 2000)["first_30"],
            "full_37": analyze_frame_set(X, y, 2000)["full_37"],
        }
    return output


def diagnostic_summary(X: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    rows: dict[str, Any] = {}
    for name, XX in (("first_30", X[:, :30]), ("full_37", X)):
        metrics = dimension_metrics(XX, labels)
        top = max(metrics["dimensions"], key=lambda row: row["accuracy"])
        rows[name] = {
            "max_single_dimension_accuracy": metrics["max_accuracy"],
            "top_dimension": {
                "index": top["index"],
                "name": top["name"],
                "accuracy": top["accuracy"],
            },
            "strictly_separable_dimensions": metrics["strictly_separable_indices"],
            "perfect_affine_linear_separation": perfect_linear_separation(XX, labels),
        }
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--boundaries", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--permutations", type=int, default=2000)
    args = parser.parse_args()

    records = sorted(read_jsonl(args.boundaries), key=lambda row: int(row["seed"]))
    if len(records) != 200 or len({int(row["seed"]) for row in records}) != 200:
        raise ValueError(f"expected 200 unique TS03 records, found {len(records)}")
    seed_to_episode = load_seed_mapping(args.raw_root)
    states_by_seed = {
        int(row["seed"]): load_raw_state(
            args.raw_root,
            int(row["seed"]),
            int(row["total_frames"]),
            seed_to_episode,
        )
        for row in records
    }
    ordered_states = [states_by_seed[int(row["seed"])] for row in records]
    labels = np.asarray(
        [int(row["target_sugar_count"]) - 1 for row in records], dtype=np.int8
    )
    if set(labels) != {0, 1}:
        raise ValueError(f"expected K=1/2 labels, found {sorted(set(labels))}")
    points = frame_points(records)

    output: dict[str, Any] = {
        "dataset": str(args.raw_root),
        "n_episodes": len(records),
        "state_dim": 37,
        "state_names": STATE_NAMES,
        "label_definition": "target_sugar_count K: 0=K1, 1=K2",
        "label_counts": {"K=1": int(np.sum(labels == 0)), "K=2": int(np.sum(labels == 1))},
        "raw_seed_mapping": {
            "n": len(seed_to_episode),
            "min_seed": min(seed_to_episode),
            "max_seed": max(seed_to_episode),
            "missing_in_seed_range": sorted(
                set(range(max(seed_to_episode) + 1)) - set(seed_to_episode)
            ),
        },
        "boundary_frames": {
            name: {
                "min": min(frames),
                "max": max(frames),
                "mean": float(np.mean(frames)),
            }
            for name, frames in points.items()
        },
        "points": {},
        "causal_offsets": {},
    }

    for name, frames in points.items():
        X = np.vstack(
            [state[int(frame)] for state, frame in zip(ordered_states, frames)]
        )
        output["points"][name] = analyze_frame_set(X, labels, args.permutations)

    # The first return is the real continue/stop handoff.  Offsets diagnose
    # when post-handoff movement makes the label visible in state.
    output["causal_offsets"] = scan_offsets(
        ordered_states, points["after_first_sugar_return"], labels
    )
    output["causal_dense_scan"] = {}
    handoff = points["after_first_sugar_return"]
    for offset in range(0, 81):
        valid = [
            i
            for i, (state, frame) in enumerate(zip(ordered_states, handoff))
            if frame + offset < len(state)
        ]
        X = np.vstack([ordered_states[i][handoff[i] + offset] for i in valid])
        output["causal_dense_scan"][str(offset)] = diagnostic_summary(X, labels[valid])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, ensure_ascii=True), encoding="utf-8")
    print(f"wrote {args.output}")
    for name, point in output["points"].items():
        for dims in ("first_30", "full_37"):
            m = point[dims]
            print(
                name,
                dims,
                "max",
                m["max_single_dimension_accuracy"],
                "strict",
                m["strictly_separable_dimensions"],
                "perm_median",
                m["permutation"]["median"],
                "linear",
                m["perfect_affine_linear_separation"]["separable"],
                "cv",
                m["ten_fold_standardized_logistic_probe"]["mean_accuracy"],
            )
    print("causal dense scan (first strict/perfect frames):")
    for dims in ("first_30", "full_37"):
        first_strict = next(
            (int(offset) for offset, row in output["causal_dense_scan"].items()
             if row[dims]["strictly_separable_dimensions"]),
            None,
        )
        first_linear = next(
            (int(offset) for offset, row in output["causal_dense_scan"].items()
             if row[dims]["perfect_affine_linear_separation"]["separable"]),
            None,
        )
        print(dims, "first_strict", first_strict, "first_perfect_affine", first_linear)


if __name__ == "__main__":
    main()
