#!/usr/bin/env python3
"""Audit WR-08 native validation batches without discarding failed attempts."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from bench.evaluation.replay import parse_seed_spec
from bench.tasks.wr08.domain.contracts import CARTON_ASSET_BY_ITEM, TASK_REVISION


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, action="append", required=True)
    parser.add_argument("--expected-seeds", required=True)
    parser.add_argument("--source-snapshot", type=Path, required=True)
    parser.add_argument("--replay-run", type=Path, action="append", default=[])
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    expected = parse_seed_spec(args.expected_seeds)
    frozen = read_json(args.source_snapshot)
    changed = [
        name
        for name, digest in frozen["files"].items()
        if not (ROOT / name).is_file()
        or hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != digest
    ]
    episodes = []
    items = []
    visited = set()
    for dataset_root in args.dataset_root:
        dataset_root = dataset_root.resolve()
        if not dataset_root.is_dir():
            raise FileNotFoundError(dataset_root)
        manifest_items = {}
        for manifest in dataset_root.rglob("manifest.json"):
            for item in read_json(manifest).get("items", []):
                if item.get("task_id") == "WR-08":
                    manifest_items[(int(item["seed"]), int(item.get("attempt", 0)))] = (
                        item
                    )
        for path in sorted(dataset_root.rglob("episode.json")):
            if "_staging" in path.parts or path.resolve() in visited:
                continue
            visited.add(path.resolve())
            episode = read_json(path)
            metadata = episode.get("metadata", {})
            if metadata.get("task_id") != "WR-08":
                continue
            seed = int(metadata["seed"])
            if seed not in expected:
                continue
            attempt = int(path.parent.name.rsplit("_attempt_", 1)[1])
            planner = read_json(path.parent / "planner.json")
            result = planner.get("episode_result", {})
            diagnostics = planner.get("final_diagnostics", {})
            stages = planner.get("rule_policy", {}).get("operations", [])
            failures = [stage for stage in stages if not stage.get("success", False)]
            failed = failures[0] if failures else {}
            failure_details = failed.get("details", {})
            resolved = metadata.get("resolved_episode", {})
            revision = (
                metadata.get("expert_plan", {}).get("metadata", {}).get("task_revision")
            )
            if revision != TASK_REVISION:
                raise ValueError(f"Not a WR-08 {TASK_REVISION} recording: {path}")
            carton_assets = metadata.get("expert_plan", {}).get("metadata", {}).get(
                "carton_asset_by_item"
            )
            if carton_assets != dict(CARTON_ASSET_BY_ITEM):
                raise ValueError(
                    f"WR-08 recording has the wrong carton assets: {path}"
                )
            success = bool(
                episode.get("accepted")
                and episode.get("planner_pass")
                and episode.get("termination", {}).get("success")
                and result.get("success")
                and diagnostics.get("success")
                and result.get("stage_count") == 17
                and len(stages) == 17
                and not failures
            )
            entry = {
                "seed": seed,
                "attempt": attempt,
                "success": success,
                "steps": episode.get("step_count"),
                "elapsed_s": episode.get("elapsed_s"),
                "stage_count": result.get("stage_count"),
                "failure_stage": failed.get("stage_index"),
                "failure_primitive": failure_details.get(
                    "primitive", failed.get("stage_kind")
                ),
                "failure_reason": (
                    None
                    if success
                    else (
                        failed.get("failure_code")
                        or failure_details.get("reason")
                        or diagnostics.get("failure_code")
                        or episode.get("termination", {}).get("reason")
                    )
                ),
                "episode_dir": str(path.parent),
                "episode_identity": metadata.get("episode_identity"),
                "config_hash": resolved.get("config_hash"),
                "resource_hash": resolved.get("resource_hash"),
                "transport_order": resolved.get("facts", {}).get("transport_order"),
                "carton_asset_by_item": dict(carton_assets),
            }
            episodes.append(entry)
            items.append(
                {
                    **manifest_items.get((seed, attempt), {}),
                    "task_id": "WR-08",
                    "seed": seed,
                    "attempt": attempt,
                    "variant": metadata.get("variant", "memory"),
                    "success": success,
                    "stage_count": entry["stage_count"],
                    "episode_identity": entry["episode_identity"],
                    "episode_dir": entry["episode_dir"],
                }
            )
    episodes.sort(key=lambda entry: (entry["seed"], entry["attempt"]))
    items.sort(key=lambda entry: (entry["seed"], entry["attempt"]))
    counts = Counter(entry["seed"] for entry in episodes)
    duplicates = sorted(seed for seed, count in counts.items() if count > 1)
    missing = sorted(expected - set(counts))
    successes = sum(entry["success"] for entry in episodes)
    replays = [
        {"run": str(path.resolve()), "summary": read_json(path / "summary.json")}
        for path in args.replay_run
    ]
    replay_pass = all(
        run["summary"].get("publishable")
        and run["summary"].get("counts", {}).get("success", 0) > 0
        and run["summary"].get("success_rate") == 1.0
        for run in replays
    )
    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "task_id": "WR-08",
        "task_revision": TASK_REVISION,
        "carton_asset_by_item": dict(CARTON_ASSET_BY_ITEM),
        "expected_seeds": sorted(expected),
        "recorded": len(episodes),
        "successes": successes,
        "failures": len(episodes) - successes,
        "missing_seeds": missing,
        "duplicate_seeds": duplicates,
        "success_rate": successes / len(episodes) if episodes else None,
        "failure_reasons": dict(
            Counter(
                entry["failure_reason"] for entry in episodes if not entry["success"]
            )
        ),
        "source_snapshot": str(args.source_snapshot.resolve()),
        "source_unchanged": not changed,
        "changed_source_files": changed,
        "config_hashes": sorted({entry["config_hash"] for entry in episodes}),
        "resource_hashes": sorted({entry["resource_hash"] for entry in episodes}),
        "recording_gate_pass": bool(
            not missing
            and not duplicates
            and not changed
            and successes == len(expected)
        ),
        "replay_runs": replays,
        "replay_gate_pass": bool(replays) and replay_pass,
        "episodes": episodes,
    }
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    for name, payload in (
        ("summary.json", report),
        (
            "manifest.json",
            {
                "schema_version": "membench.dataset/v2",
                "dataset_id": output.name,
                "items": items,
            },
        ),
    ):
        (output / name).write_text(
            json.dumps(payload, indent=2) + "\n", encoding="utf-8"
        )
    print(
        json.dumps(
            {
                key: value
                for key, value in report.items()
                if key not in {"episodes", "replay_runs"}
            },
            indent=2,
        )
    )
    return 0 if report["recording_gate_pass"] and replay_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
