#!/usr/bin/env python3
"""State-only separability audit for the generated WA02 v062 dataset.

Boundary records contain stage timing and semantic facts recovered from each
recorded episode.  State values are always loaded from the raw LeRobot
parquet files; boundary metadata is never used as a state source.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
from scipy.optimize import linprog, minimize


STATE_NAMES = [
    "arm_joint_qpos_0", "arm_joint_qpos_1", "arm_joint_qpos_2",
    "arm_joint_qpos_3", "arm_joint_qpos_4", "arm_joint_qpos_5",
    "arm_joint_qpos_6", "arm_joint_qvel_0", "arm_joint_qvel_1",
    "arm_joint_qvel_2", "arm_joint_qvel_3", "arm_joint_qvel_4",
    "arm_joint_qvel_5", "arm_joint_qvel_6", "gripper_qpos_0",
    "gripper_qpos_1", "gripper_qvel_0", "gripper_qvel_1",
    "base_pos_world_0", "base_pos_world_1", "base_pos_world_2",
    "base_quat_world_x", "base_quat_world_y", "base_quat_world_z",
    "base_quat_world_w", "base_joint_qvel_0", "base_joint_qvel_1",
    "base_joint_qvel_2", "torso_qpos", "torso_qvel", "eef_pos_in_base_0",
    "eef_pos_in_base_1", "eef_pos_in_base_2", "eef_quat_in_base_x",
    "eef_quat_in_base_y", "eef_quat_in_base_z", "eef_quat_in_base_w",
]
SLOT_TO_SIDE = {"slot_1": "right", "slot_2": "left"}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_state(raw_root: Path, seed: int, expected_length: int) -> np.ndarray:
    path = raw_root / "data" / "chunk-000" / f"file-{seed:03d}.parquet"
    table = pq.read_table(path, columns=["observation.state"])
    values = table["observation.state"].combine_chunks().values.to_numpy()
    state = np.asarray(values, dtype=np.float64).reshape(-1, 37)
    if len(state) != expected_length:
        raise ValueError(f"seed {seed}: raw rows {len(state)} != boundary length {expected_length}")
    return state


def threshold_metrics(x: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    """Best one-dimensional threshold, allowing either orientation."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.int8)
    order = np.argsort(x, kind="mergesort")
    xs, ys = x[order], y[order]
    # k values at or below the threshold are predicted as class 0.  The
    # complementary orientation is also evaluated at every cut.
    n = len(y)
    prefix_one = np.concatenate(([0], np.cumsum(ys == 1)))
    prefix_zero = np.concatenate(([0], np.cumsum(ys == 0)))
    total_one = int(prefix_one[-1])
    total_zero = int(prefix_zero[-1])
    # A cut is only allowed between distinct values, plus the two extremes.
    cuts = [0]
    cuts.extend((np.flatnonzero(xs[1:] != xs[:-1]) + 1).tolist())
    cuts.append(n)
    best = (-1, None, None)
    for k in cuts:
        # x <= threshold => 0, x > threshold => 1
        acc_up = int(prefix_zero[k] + (total_one - prefix_one[k]))
        # x <= threshold => 1, x > threshold => 0
        acc_down = int(prefix_one[k] + (total_zero - prefix_zero[k]))
        if acc_up > best[0]:
            if k == 0:
                threshold = float(np.nextafter(xs[0], -np.inf))
            elif k == n:
                threshold = float(np.nextafter(xs[-1], np.inf))
            else:
                threshold = float((xs[k - 1] + xs[k]) / 2.0)
            best = (acc_up, threshold, "x_gt_threshold_is_1")
        if acc_down > best[0]:
            if k == 0:
                threshold = float(np.nextafter(xs[0], -np.inf))
            elif k == n:
                threshold = float(np.nextafter(xs[-1], np.inf))
            else:
                threshold = float((xs[k - 1] + xs[k]) / 2.0)
            best = (acc_down, threshold, "x_le_threshold_is_1")
    return {"correct": int(best[0]), "accuracy": float(best[0] / n), "threshold": best[1], "orientation": best[2]}


def dimension_metrics(X: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    left = X[y == 0]
    right = X[y == 1]
    rows: list[dict[str, Any]] = []
    for j, name in enumerate(STATE_NAMES[: X.shape[1]]):
        lo, hi = float(left[:, j].min()), float(left[:, j].max())
        ro, rh = float(right[:, j].min()), float(right[:, j].max())
        strict = (hi < ro) or (rh < lo)
        row = {"index": j, "name": name, "left_min": lo, "left_max": hi,
               "right_min": ro, "right_max": rh, "strictly_separable": bool(strict)}
        row.update(threshold_metrics(X[:, j], y))
        rows.append(row)
    return {"dimensions": rows, "max_accuracy": max(r["accuracy"] for r in rows),
            "strictly_separable_indices": [r["index"] for r in rows if r["strictly_separable"]]}


def standardize_fit(X_train: np.ndarray, X_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = X_train.mean(axis=0)
    scale = X_train.std(axis=0)
    scale[scale < 1e-12] = 1.0
    return (X_train - mean) / scale, (X_test - mean) / scale


def fit_logistic(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    z = 2.0 * y.astype(np.float64) - 1.0
    d = X.shape[1]

    def objective(theta: np.ndarray) -> tuple[float, np.ndarray]:
        margins = z * (X @ theta[:d] + theta[d])
        loss = np.logaddexp(0.0, -margins).sum() + 0.5 * np.dot(theta[:d], theta[:d])
        weight = -z / (1.0 + np.exp(np.clip(margins, -60.0, 60.0)))
        grad_w = X.T @ weight + theta[:d]
        grad_b = float(weight.sum())
        return float(loss), np.r_[grad_w, grad_b]

    result = minimize(lambda t: objective(t), np.zeros(d + 1), jac=True, method="L-BFGS-B",
                      options={"maxiter": 2000, "ftol": 1e-12, "gtol": 1e-8})
    if not result.success and result.message not in {"STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT"}:
        raise RuntimeError(f"logistic fit failed: {result.message}")
    return result.x


def stratified_folds(y: np.ndarray, n_folds: int = 10) -> list[np.ndarray]:
    parts = [np.array_split(np.flatnonzero(y == cls), n_folds) for cls in (0, 1)]
    return [np.concatenate((parts[0][k], parts[1][k])) for k in range(n_folds)]


def cross_validated_probe(X: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    folds = stratified_folds(y)
    scores: list[float] = []
    for test_idx in folds:
        train_idx = np.setdiff1d(np.arange(len(y)), test_idx, assume_unique=False)
        train, test = standardize_fit(X[train_idx], X[test_idx])
        theta = fit_logistic(train, y[train_idx])
        pred = ((test @ theta[:-1] + theta[-1]) >= 0).astype(np.int8)
        scores.append(float(np.mean(pred == y[test_idx])))
    return {"fold_accuracies": scores, "mean_accuracy": float(np.mean(scores)),
            "std_accuracy": float(np.std(scores, ddof=1)), "min_accuracy": float(np.min(scores)),
            "max_accuracy": float(np.max(scores))}


def perfect_linear_separation(X: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    mean = X.mean(axis=0)
    scale = X.std(axis=0)
    scale[scale < 1e-12] = 1.0
    Z = (X - mean) / scale
    signed = 2.0 * y.astype(np.float64) - 1.0
    augmented = np.c_[Z, np.ones(len(Z))]
    # signed*(w dot x + b) >= 1 for every sample.
    A_ub = -signed[:, None] * augmented
    result = linprog(np.zeros(augmented.shape[1]), A_ub=A_ub, b_ub=-np.ones(len(Z)),
                     bounds=[(None, None)] * augmented.shape[1], method="highs")
    return {"separable": bool(result.success), "status": int(result.status),
            "message": str(result.message)}


def permutation_maxima(X: np.ndarray, y: np.ndarray, n_perm: int, seed: int = 20260902) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n, d = X.shape
    orders = np.argsort(X, axis=0, kind="mergesort").T
    valid = np.zeros((d, n + 1), dtype=bool)
    valid[:, 0] = True
    valid[:, n] = True
    for j in range(d):
        valid[j, 1:n] = X[orders[j, 1:], j] != X[orders[j, :-1], j]
    maxima = np.empty(n_perm, dtype=np.float64)
    for i in range(n_perm):
        yp = rng.permutation(y)
        sorted_y = yp[orders]
        prefix_one = np.zeros((d, n + 1), dtype=np.int32)
        prefix_one[:, 1:] = np.cumsum(sorted_y == 1, axis=1)
        total_one = prefix_one[:, -1, None]
        prefix_zero = np.zeros((d, n + 1), dtype=np.int32)
        prefix_zero[:, 1:] = np.cumsum(sorted_y == 0, axis=1)
        total_zero = prefix_zero[:, -1, None]
        up = prefix_zero + total_one - prefix_one
        down = prefix_one + total_zero - prefix_zero
        scores = np.maximum(up, down).astype(np.int32)
        scores[~valid] = -1
        maxima[i] = float(scores.max() / n)
    return maxima


def analyze_point(X: np.ndarray, y: np.ndarray, n_perm: int) -> dict[str, Any]:
    baseline = float(max(np.sum(y == 0), np.sum(y == 1)) / len(y))
    one = dimension_metrics(X, y)
    maxima = permutation_maxima(X, y, n_perm)
    q = np.percentile(maxima, [2.5, 50, 97.5])
    result = {
        "n": int(len(y)), "class_counts": {"left": int(np.sum(y == 0)), "right": int(np.sum(y == 1))},
        "majority_baseline": baseline, "max_single_dimension_accuracy": one["max_accuracy"],
        "strictly_separable_dimensions": one["strictly_separable_indices"],
        "dimensions": one["dimensions"], "permutation": {
            "n": int(n_perm), "seed": 20260902, "median": float(q[1]), "q025": float(q[0]),
            "q975": float(q[2]), "mean": float(np.mean(maxima)),
            "actual_percentile": float(np.mean(maxima <= one["max_accuracy"])),
            "n_ge_actual": int(np.sum(maxima >= one["max_accuracy"])),
        },
        "perfect_affine_linear_separation": perfect_linear_separation(X, y),
        "ten_fold_standardized_logistic_probe": cross_validated_probe(X, y),
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--boundaries", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--permutations", type=int, default=2000)
    args = parser.parse_args()
    records = []
    for path in args.boundaries:
        records.extend(read_jsonl(path))
    records.sort(key=lambda r: int(r["seed"]))
    if [int(r["seed"]) for r in records] != list(range(len(records))):
        raise ValueError("boundary records must contain consecutive seeds starting at 0")

    states: dict[int, np.ndarray] = {}
    for rec in records:
        states[int(rec["seed"])] = load_state(args.raw_root, int(rec["seed"]), int(rec["total_frames"]))

    point_frames: dict[str, list[int]] = {}
    point_states: dict[str, np.ndarray] = {}
    for name, frame_fn in {
        "initial": lambda r: 0,
        "after_first_fill": lambda r: int(r["stages"][4]["end_frame"]) - 1,
        "after_first_cup_placed": lambda r: int(r["stages"][8]["end_frame"]) - 1,
        "after_first_service_return": lambda r: int(r["stages"][10]["end_frame"]) - 1,
        "final_selection_handoff": lambda r: int(r["stages"][23]["start_frame"]),
        "final_pick_navigation_start": lambda r: int(r["stages"][25]["start_frame"]),
    }.items():
        frames = [int(frame_fn(r)) for r in records]
        point_frames[name] = frames
        point_states[name] = np.vstack([states[int(r["seed"])][frame] for r, frame in zip(records, frames)])

    labels = np.array([0 if SLOT_TO_SIDE[str(r["service_slot_by_cup"][r["requested_cup_id"]])] == "left" else 1 for r in records], dtype=np.int8)
    output: dict[str, Any] = {
        "dataset": str(args.raw_root), "n_episodes": len(records), "state_dim": 37,
        "state_names": STATE_NAMES, "label_definition": "requested cup service slot mapped slot_2=left, slot_1=right",
        "label_counts": {"left": int(np.sum(labels == 0)), "right": int(np.sum(labels == 1))},
        "boundary_frames": {k: {"min": min(v), "max": max(v), "mean": float(np.mean(v))} for k, v in point_frames.items()},
        "points": {}, "causal_offsets": {}, "causal_scan": {},
    }
    for name, X in point_states.items():
        output["points"][name] = {
            "first_30": analyze_point(X[:, :30], labels, args.permutations),
            "full_37": analyze_point(X, labels, args.permutations),
        }

    handoff_frames = point_frames["final_selection_handoff"]
    for offset in (0, 1, 5, 10, 20, 40, 80, 120):
        valid = [i for i, (s, f) in enumerate(zip(states.values(), handoff_frames)) if f + offset < len(s)]
        X = np.vstack([list(states.values())[i][handoff_frames[i] + offset] for i in valid])
        yy = labels[valid]
        output["causal_offsets"][str(offset)] = {
            "n": len(valid), "frame_offset": offset,
            "first_30": analyze_point(X[:, :30], yy, args.permutations),
            "full_37": analyze_point(X, yy, args.permutations),
        }

    # Dense scan is diagnostic only: it reports when post-handoff movement
    # makes the label visible, without spending permutations on every frame.
    for offset in range(0, 81):
        valid = [i for i, (s, f) in enumerate(zip(states.values(), handoff_frames)) if f + offset < len(s)]
        X = np.vstack([list(states.values())[i][handoff_frames[i] + offset] for i in valid])
        yy = labels[valid]
        output["causal_scan"][str(offset)] = {}
        for dims, XX in (("first_30", X[:, :30]), ("full_37", X)):
            dm = dimension_metrics(XX, yy)
            top = max(dm["dimensions"], key=lambda row: row["accuracy"])
            output["causal_scan"][str(offset)][dims] = {
                "max_single_dimension_accuracy": dm["max_accuracy"],
                "top_dimension": {"index": top["index"], "name": top["name"], "accuracy": top["accuracy"]},
                "strictly_separable_dimensions": dm["strictly_separable_indices"],
                "perfect_affine_linear_separation": perfect_linear_separation(XX, yy),
            }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, ensure_ascii=True), encoding="utf-8")
    print(f"wrote {args.output}")
    for name in output["points"]:
        for dims in ("first_30", "full_37"):
            m = output["points"][name][dims]
            print(name, dims, "max", m["max_single_dimension_accuracy"],
                  "strict", m["strictly_separable_dimensions"],
                  "perm_median", m["permutation"]["median"],
                  "linear", m["perfect_affine_linear_separation"]["separable"],
                  "cv", m["ten_fold_standardized_logistic_probe"]["mean_accuracy"])


if __name__ == "__main__":
    main()
