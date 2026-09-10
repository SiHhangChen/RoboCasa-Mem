#!/usr/bin/env python3
"""State-only separability audit for TS01's remembered sugar count.

TS01's policy-relevant label is the requested number of sugar cubes K (1, 2,
or 3), not the internally sampled sugar names.  Boundary metadata supplies
stage timing and semantic labels only.  State values are loaded directly from
the raw LeRobot parquet dataset.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path
from typing import Any, Iterable

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
    if seed not in seed_to_episode:
        raise ValueError(f"seed {seed} is absent from raw meta/seeds.jsonl")
    episode_index = seed_to_episode[seed]
    path = raw_root / "data" / "chunk-000" / f"file-{episode_index:03d}.parquet"
    table = pq.read_table(path, columns=["observation.state"])
    values = table["observation.state"].combine_chunks().values.to_numpy()
    state = np.asarray(values, dtype=np.float64).reshape(-1, 37)
    if len(state) != expected_length:
        raise ValueError(
            f"seed {seed}: raw rows {len(state)} != boundary length {expected_length}"
        )
    return state


def stage(row: dict[str, Any], name: str) -> dict[str, Any]:
    matches = [item for item in row["stages"] if item["name"] == name]
    if len(matches) != 1:
        raise ValueError(
            f"seed {row['seed']}: expected one stage {name!r}, found {len(matches)}"
        )
    return matches[0]


def return_stages(row: dict[str, Any]) -> list[dict[str, Any]]:
    values = [item for item in row["stages"] if item["name"].startswith("return_after_")]
    if len(values) != int(row["target_sugar_count"]):
        raise ValueError(
            f"seed {row['seed']}: found {len(values)} returns for K={row['target_sugar_count']}"
        )
    return values


def valid_cuts(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    cuts = np.r_[0, np.flatnonzero(sorted_values[1:] != sorted_values[:-1]) + 1, len(values)]
    return np.unique(cuts)


def multiclass_threshold_score(
    values: np.ndarray, labels: np.ndarray, classes: Iterable[int]
) -> tuple[int, int, int, tuple[int, ...]]:
    """Return best accuracy from two scalar cuts and arbitrary class order.

    The returned cut indices refer to sorted samples: intervals are [0, i),
    [i, j), and [j, n).  Dynamic programming keeps the permutation audit
    linear in the number of candidate cuts.
    """

    values = np.asarray(values, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    classes_tuple = tuple(int(c) for c in classes)
    if len(classes_tuple) == 2:
        raise ValueError("use binary analysis for two classes")
    order = np.argsort(values, kind="mergesort")
    sorted_labels = labels[order]
    cuts = valid_cuts(values)
    prefix = np.zeros((len(classes_tuple), len(values) + 1), dtype=np.int32)
    for index, cls in enumerate(classes_tuple):
        prefix[index, 1:] = np.cumsum(sorted_labels == cls)
    total = prefix[:, -1]
    best = (-1, 0, len(values), classes_tuple)
    # K=1/K=2/K=3 uses exactly three intervals.  Empty intervals are allowed
    # so that a constant predictor remains a valid baseline.
    for permutation in itertools.permutations(range(len(classes_tuple))):
        p0, p1, p2 = permutation
        for i in cuts:
            first = int(prefix[p0, i])
            for j in cuts:
                if j < i:
                    continue
                score = first + int(prefix[p1, j] - prefix[p1, i])
                score += int(total[p2] - prefix[p2, j])
                if score > best[0]:
                    best = (score, int(i), int(j), tuple(classes_tuple[k] for k in permutation))
    return best


def multiclass_permutation_maxima(
    X: np.ndarray, labels: np.ndarray, n_perm: int, seed: int = 20260902
) -> np.ndarray:
    """Max scalar three-bin accuracy after fixed-composition label shuffles."""

    X = np.asarray(X, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    classes = tuple(sorted(int(c) for c in np.unique(labels)))
    if len(classes) != 3:
        raise ValueError("multiclass permutation audit expects exactly three classes")
    orders = np.argsort(X, axis=0, kind="mergesort")
    cuts_by_dim = [valid_cuts(X[orders[:, j], j]) for j in range(X.shape[1])]
    rng = np.random.default_rng(seed)
    maxima = np.empty(int(n_perm), dtype=np.float64)
    for index in range(int(n_perm)):
        shuffled = rng.permutation(labels)
        best = 0
        for j in range(X.shape[1]):
            order = orders[:, j]
            sorted_labels = shuffled[order]
            cuts = cuts_by_dim[j]
            prefix = np.zeros((3, len(labels) + 1), dtype=np.int32)
            for cls_index, cls in enumerate(classes):
                prefix[cls_index, 1:] = np.cumsum(sorted_labels == cls)
            total = prefix[:, -1]
            for permutation in itertools.permutations(range(3)):
                p0, p1, p2 = permutation
                # Dynamic programming over the second cut.  The first cut's
                # contribution is folded into the best prefix for each j.
                best_prefix = -10**9
                for cut_j in cuts:
                    candidate_i = int(cut_j)
                    best_prefix = max(
                        best_prefix,
                        int(prefix[p0, candidate_i] - prefix[p1, candidate_i]),
                    )
                    score = int(prefix[p1, cut_j]) + best_prefix
                    score += int(total[p2] - prefix[p2, cut_j])
                    if score > best:
                        best = score
        maxima[index] = best / len(labels)
    return maxima


def multiclass_dimension_metrics(X: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    classes = tuple(sorted(int(c) for c in np.unique(labels)))
    rows: list[dict[str, Any]] = []
    for j, name in enumerate(STATE_NAMES[: X.shape[1]]):
        score, cut1, cut2, ordering = multiclass_threshold_score(X[:, j], labels, classes)
        one_vs_rest: dict[str, dict[str, Any]] = {}
        for cls in classes:
            binary = (labels == cls).astype(np.int8)
            metrics = dimension_metrics(X[:, [j]], binary)["dimensions"][0]
            one_vs_rest[str(cls)] = {
                "class_min": float(X[labels == cls, j].min()),
                "class_max": float(X[labels == cls, j].max()),
                "rest_min": float(X[labels != cls, j].min()),
                "rest_max": float(X[labels != cls, j].max()),
                "strictly_separable": bool(metrics["strictly_separable"]),
                "best_accuracy": float(metrics["accuracy"]),
            }
        rows.append(
            {
                "index": j,
                "name": name,
                "best_correct": int(score),
                "best_accuracy": float(score / len(labels)),
                "cut1_sorted_index": int(cut1),
                "cut2_sorted_index": int(cut2),
                "interval_class_order": list(ordering),
                "one_vs_rest": one_vs_rest,
            }
        )
    return {
        "dimensions": rows,
        "max_accuracy": float(max(row["best_accuracy"] for row in rows)),
        "best_dimension": max(rows, key=lambda row: row["best_accuracy"])["index"],
        "strictly_separable_one_vs_rest": {
            str(cls): [
                row["index"]
                for row in rows
                if row["one_vs_rest"][str(cls)]["strictly_separable"]
            ]
            for cls in classes
        },
    }


def analyze_multiclass(X: np.ndarray, labels: np.ndarray, n_perm: int) -> dict[str, Any]:
    classes = tuple(sorted(int(c) for c in np.unique(labels)))
    counts = {str(cls): int(np.sum(labels == cls)) for cls in classes}
    baseline = max(counts.values()) / len(labels)
    dimensions = multiclass_dimension_metrics(X, labels)
    maxima = multiclass_permutation_maxima(X, labels, n_perm)
    actual = dimensions["max_accuracy"]
    q025, median, q975 = np.percentile(maxima, [2.5, 50, 97.5])
    one_vs_rest_linear = {
        str(cls): perfect_linear_separation(X, (labels == cls).astype(np.int8))
        for cls in classes
    }
    return {
        "n": int(len(labels)),
        "classes": list(classes),
        "class_counts": counts,
        "majority_baseline": float(baseline),
        "max_single_dimension_threshold_accuracy": actual,
        "dimensions": dimensions["dimensions"],
        "strictly_separable_one_vs_rest": dimensions["strictly_separable_one_vs_rest"],
        "permutation": {
            "n": int(n_perm),
            "seed": 20260902,
            "median": float(median),
            "q025": float(q025),
            "q975": float(q975),
            "mean": float(np.mean(maxima)),
            "actual_percentile": float(np.mean(maxima <= actual)),
            "n_ge_actual": int(np.sum(maxima >= actual)),
        },
        "one_vs_rest_perfect_affine_linear_separation": one_vs_rest_linear,
    }


def analyze_binary(
    X: np.ndarray, labels: np.ndarray, negative: str, positive: str, n_perm: int
) -> dict[str, Any]:
    result = analyze_point(X, labels.astype(np.int8), n_perm)
    result["class_counts"] = {
        negative: int(np.sum(labels == 0)),
        positive: int(np.sum(labels == 1)),
    }
    result["label_names"] = {"0": negative, "1": positive}
    return result


def collect_point_frames(records: list[dict[str, Any]]) -> dict[str, list[int | None]]:
    points: dict[str, list[int | None]] = {"initial": [0 for _ in records]}
    for ordinal in (1, 2, 3):
        place_frames: list[int | None] = []
        frames: list[int | None] = []
        for row in records:
            returns = return_stages(row)
            places = [
                item
                for item in row["stages"]
                if item["name"].startswith("place_")
                and item["name"].endswith("_saucepan")
            ]
            if len(places) != int(row["target_sugar_count"]):
                raise ValueError(
                    f"seed {row['seed']}: found {len(places)} placements for K={row['target_sugar_count']}"
                )
            if len(places) < ordinal:
                place_frames.append(None)
            else:
                place_frames.append(int(places[ordinal - 1]["end_frame"]) - 1)
            if len(returns) < ordinal:
                frames.append(None)
            else:
                frames.append(int(returns[ordinal - 1]["end_frame"]) - 1)
        points[f"after_{ordinal}_sugar_place"] = place_frames
        points[f"after_{ordinal}_sugar_return"] = frames
    points["completion_handoff"] = [
        int(stage(row, "navigate_completion_button")["start_frame"])
        for row in records
    ]
    return points


def analyze_point_set(
    X: np.ndarray,
    labels_k: np.ndarray,
    n_perm: int,
    binary_labels: dict[str, tuple[np.ndarray, str, str]],
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "multiclass_k": analyze_multiclass(X, labels_k, n_perm)
        if len(np.unique(labels_k)) == 3
        else None,
        "binary": {},
    }
    for name, (labels, negative, positive) in binary_labels.items():
        result["binary"][name] = analyze_binary(X, labels, negative, positive, n_perm)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--boundaries", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--permutations", type=int, default=2000)
    args = parser.parse_args()
    records = sorted(read_jsonl(args.boundaries), key=lambda row: int(row["seed"]))
    if len(records) != 200 or len({int(row["seed"]) for row in records}) != 200:
        raise ValueError(f"expected 200 unique TS01 records, found {len(records)}")
    seed_to_episode = load_seed_mapping(args.raw_root)
    states = {
        int(row["seed"]): load_raw_state(
            args.raw_root,
            int(row["seed"]),
            int(row["total_frames"]),
            seed_to_episode,
        )
        for row in records
    }
    labels_k = np.asarray([int(row["target_sugar_count"]) for row in records], dtype=np.int8)
    if set(labels_k) != {1, 2, 3}:
        raise ValueError(f"expected K classes 1/2/3, found {sorted(set(labels_k))}")
    point_frames = collect_point_frames(records)
    ordered_states = [states[int(row["seed"])] for row in records]
    output: dict[str, Any] = {
        "dataset": str(args.raw_root),
        "n_episodes": len(records),
        "state_dim": 37,
        "state_names": STATE_NAMES,
        "label_definition": "target_sugar_count K from resolved episode facts; K=1/2/3",
        "label_counts": {str(k): int(np.sum(labels_k == k)) for k in (1, 2, 3)},
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
                "min": min(value for value in frames if value is not None),
                "max": max(value for value in frames if value is not None),
                "mean": float(np.mean([value for value in frames if value is not None])),
                "n": sum(value is not None for value in frames),
            }
            for name, frames in point_frames.items()
        },
        "points": {},
        "causal_offsets": {},
    }

    for name, frames in point_frames.items():
        indices = [i for i, frame in enumerate(frames) if frame is not None]
        if not indices:
            continue
        X = np.vstack([ordered_states[i][int(frames[i])] for i in indices])
        yk = labels_k[indices]
        binary: dict[str, tuple[np.ndarray, str, str]] = {
            "needs_second": ((yk >= 2).astype(np.int8), "K=1", "K>=2"),
            "needs_third": ((yk == 3).astype(np.int8), "K=1/2", "K=3"),
        }
        # Only retain non-degenerate binary labels at each point.
        binary = {
            key: value
            for key, value in binary.items()
            if len(np.unique(value[0])) == 2
        }
        output["points"][name] = {
            "all_episodes": analyze_point_set(X[:, :30], yk, args.permutations, binary)
            if name == "initial" or name == "completion_handoff"
            else None,
        }
        # After each return, analyze the decision-relevant subset as well as
        # the full K groups when all three classes are present.
        if name in {"after_1_sugar_place", "after_1_sugar_return"}:
            output["points"][name]["decision_subset"] = analyze_point_set(
                X[:, :30], yk, args.permutations,
                {"needs_second": ((yk >= 2).astype(np.int8), "K=1", "K>=2")},
            )
            output["points"][name]["full_37_decision_subset"] = analyze_point_set(
                X, yk, args.permutations,
                {"needs_second": ((yk >= 2).astype(np.int8), "K=1", "K>=2")},
            )
        elif name in {"after_2_sugar_place", "after_2_sugar_return"}:
            # Only K>=2 episodes reach the second return, so this point is
            # already the K=2 versus K=3 decision subset.
            XX = X
            yy = yk
            output["points"][name]["decision_subset"] = analyze_point_set(
                XX[:, :30], yy, args.permutations,
                {"needs_third": ((yy == 3).astype(np.int8), "K=2", "K=3")},
            )
            output["points"][name]["full_37_decision_subset"] = analyze_point_set(
                XX, yy, args.permutations,
                {"needs_third": ((yy == 3).astype(np.int8), "K=2", "K=3")},
            )
        elif name in {"after_3_sugar_place", "after_3_sugar_return"}:
            output["points"][name] = {"note": "only K=3 episodes reach this point"}
        else:
            output["points"][name]["full_37"] = analyze_point_set(
                X, yk, args.permutations, binary
            ) if output["points"][name]["all_episodes"] is not None else None

    # Causal offsets from each actual continue/stop handoff.  This is a
    # diagnostic for state changes caused by the next navigation action.
    for handoff_name, eligible in (
        ("after_1_sugar_return", np.ones(len(records), dtype=bool)),
        ("after_2_sugar_return", labels_k >= 2),
    ):
        frames = point_frames[handoff_name]
        indices = np.flatnonzero(eligible).tolist()
        output["causal_offsets"][handoff_name] = {}
        for offset in (0, 1, 5, 10, 20, 40, 80, 120, 160):
            valid = [
                i
                for i in indices
                if frames[i] is not None
                and int(frames[i]) + offset < len(ordered_states[i])
            ]
            X = np.vstack([ordered_states[i][int(frames[i]) + offset] for i in valid])
            yy = labels_k[valid]
            if handoff_name == "after_1_sugar_return":
                labels = (yy >= 2).astype(np.int8)
                negative, positive = "K=1", "K>=2"
            else:
                labels = (yy == 3).astype(np.int8)
                negative, positive = "K=2", "K=3"
            output["causal_offsets"][handoff_name][str(offset)] = {
                "n": len(valid),
                "frame_offset": offset,
                "first_30": analyze_binary(X[:, :30], labels, negative, positive, args.permutations),
                "full_37": analyze_binary(X, labels, negative, positive, args.permutations),
            }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, ensure_ascii=True), encoding="utf-8")
    print(f"wrote {args.output}")
    for name, point in output["points"].items():
        if "decision_subset" in point:
            for dims in ("decision_subset", "full_37_decision_subset"):
                m = point[dims]["binary"]
                key = "needs_second" if name.startswith("after_1_") else "needs_third"
                value = m[key]
                print(
                    name,
                    dims,
                    key,
                    "max",
                    value["max_single_dimension_accuracy"],
                    "strict",
                    value["strictly_separable_dimensions"],
                    "perm_median",
                    value["permutation"]["median"],
                    "linear",
                    value["perfect_affine_linear_separation"]["separable"],
                    "cv",
                    value["ten_fold_standardized_logistic_probe"]["mean_accuracy"],
                )
        if point.get("all_episodes") and point["all_episodes"].get("multiclass_k"):
            m = point["all_episodes"]["multiclass_k"]
            print(name, "K multiclass max", m["max_single_dimension_threshold_accuracy"], "baseline", m["majority_baseline"], "perm_median", m["permutation"]["median"])


if __name__ == "__main__":
    main()
