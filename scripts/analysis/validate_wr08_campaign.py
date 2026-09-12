#!/usr/bin/env python3
"""Join WR-08 recording and production-replay evidence by seed and identity.

Publication requires exact expected-seed coverage, matching executed actions,
successful native trajectories, and all seven public report stages. A policy
replay may end at a verified task stop before the expert's remaining actions.
Interrupted replay attempts remain in the report but are not treated as final
results. This command does not start simulations or alter source recordings.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from bench.evaluation.artifacts import ArtifactWriter
from bench.evaluation.report_profiles import WR08_REPORT_PROFILE
from bench.evaluation.replay import discover_recorded_episodes, parse_seed_spec
from bench.evaluation.summary import aggregate_episode_results
from bench.recording.artifacts import atomic_json_commit
from bench.recording.plan import RecordingWorkItem
from scripts.analysis.verify_wr08_native_episodes import verify
from scripts.analysis.wr08_recording_provenance import validate_recording_provenance

PUBLIC_STAGES = WR08_REPORT_PROFILE.stage_ids


def replay_evidence_errors(result, native, actions) -> list[str]:
    errors = []
    seed = int(result["seed"])
    if result.get("task_id") != "WR-08":
        errors.append("wrong_task")
    if result.get("status") != "success" or not result.get("success"):
        errors.append("replay_not_successful")
    metadata = result.get("episode_metadata", {})
    replay = metadata.get("replay", {})
    if replay.get("episode_identity_matches") is not True:
        errors.append("episode_identity_not_verified")
    for name in ("expected_episode_identity", "current_episode_identity"):
        if replay.get(name) != native["episode_identity"]:
            errors.append(f"{name}_mismatch")
    executed = result.get("env_steps")
    valid_count = isinstance(executed, int) and not isinstance(executed, bool) and 0 < executed <= native["steps"]
    diagnostics = result.get("diagnostics") or {}
    evaluation = diagnostics.get("evaluation") or {}
    verified_stop = bool(
        diagnostics.get("stop_execution") is True
        and diagnostics.get("terminal") is True
        and diagnostics.get("success") is True
        and diagnostics.get("failure_code") is None
        and evaluation.get("mode") == "policy_physical"
        and evaluation.get("terminal_step") == executed
        and result.get("success_step") == executed
    )
    if (
        not valid_count
        or result.get("policy_actions_consumed") != executed
        or len(actions) != executed
        or (executed != native["steps"] and not verified_stop)
    ):
        errors.append("incomplete_action_stream")
    with h5py.File(Path(native["episode_dir"]) / "trajectory.h5", "r") as handle:
        recorded = np.asarray(handle["data/demo_1/actions"], dtype=np.float32)
    expected_actions = recorded[:executed] if valid_count else recorded
    if not np.array_equal(expected_actions, actions):
        errors.append("replayed_actions_differ_from_recording")
    report = result.get("report_stages") or {}
    if report.get("profile_id") != WR08_REPORT_PROFILE.profile_id or tuple(report.get("order", ())) != PUBLIC_STAGES:
        errors.append("wrong_public_report_profile")
    stages = report.get("stages", {})
    if any(stages.get(name, {}).get("passed") is not True
           or stages.get(name, {}).get("applicable") is not True for name in PUBLIC_STAGES):
        errors.append("public_report_stage_not_passed")
    return [f"seed {seed}: {error}" for error in errors]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, action="append", required=True)
    parser.add_argument("--replay-run", type=Path, action="append", default=[])
    parser.add_argument("--expected-seeds", default="0-199")
    parser.add_argument("--source-snapshot", type=Path, required=True)
    parser.add_argument("--recording-provenance", type=Path,
                        help="Source-batch manifest for mixed original/repaired expert recordings")
    parser.add_argument("--verify-videos", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    expected = parse_seed_spec(args.expected_seeds)
    if not expected:
        raise ValueError("expected seed set must not be empty")
    frozen = json.loads(args.source_snapshot.read_text())
    changed = [name for name, digest in frozen["files"].items()
               if not (ROOT / name).is_file()
               or hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != digest]
    items, native, native_errors, paths = [], {}, [], set()
    failed_attempts = []
    for root in args.dataset_root:
        for item in discover_recorded_episodes(root.resolve(), "WR-08", expected, source_status="success"):
            path = Path(item["episode_dir"]).resolve()
            if path in paths:
                continue
            paths.add(path)
            seed = int(item["seed"])
            try:
                row = verify(path, videos=args.verify_videos)
                if seed in native:
                    raise ValueError("duplicate successful recording for seed")
                if seed != row["seed"]:
                    raise ValueError("directory/metadata seed mismatch")
                native[seed] = row
                metadata = json.loads((path / "episode.json").read_text())["metadata"]
                work = RecordingWorkItem("WR-08", seed, metadata["variant"], item["attempt"])
                items.append({"work_id": work.work_id, "task_id": work.task_id, "seed": seed,
                              "variant": work.variant, "attempt": work.attempt, "success": True,
                              "stage_count": 17, "episode_identity": row["episode_identity"],
                              "episode_dir": row["episode_dir"]})
            except Exception as exc:
                native_errors.append({"seed": seed, "path": str(path), "error": str(exc)})
        for item in discover_recorded_episodes(root.resolve(), "WR-08", expected, source_status="failed"):
            failed_attempts.append({**item, "episode_dir": str(item["episode_dir"])})
    results, replay_errors, cancelled, result_paths = [], [], [], set()
    for run in args.replay_run:
        writer = ArtifactWriter(run.resolve())
        invalid = writer.invalid_episode_artifacts()
        replay_errors.extend(f"{run}/{path}: {error}" for path, error in invalid.items())
        for result in writer.load_episode_results():
            seed = int(result["seed"])
            path = (run / "episodes" / f"seed_{seed:06d}" / "result.json").resolve()
            if path in result_paths:
                continue
            result_paths.add(path)
            if result["status"] == "cancelled":
                cancelled.append({"seed": seed, "result": str(path)})
                continue
            results.append(result)
            if seed not in native:
                replay_errors.append(f"seed {seed}: no selected native recording")
                continue
            with np.load(path.parent / "actions.npz", allow_pickle=False) as payload:
                actions = np.asarray(payload["actions"], dtype=np.float32)
            replay_errors.extend(replay_evidence_errors(result, native[seed], actions))
    aggregate = aggregate_episode_results(results, expected_seeds=sorted(expected))
    actual_replays = Counter(int(row["seed"]) for row in results)
    missing_recordings = sorted(expected - set(native))
    missing_replays = sorted(expected - set(actual_replays))
    provenance = (validate_recording_provenance(native, args.recording_provenance, args.source_snapshot)
                  if args.recording_provenance else None)
    if provenance is not None and provenance["verified"]:
        sources = {seed: batch["source_snapshot"] for batch in provenance["batches"]
                   for seed in batch["seeds"]}
        for item in items:
            item["recording_source_snapshot"] = sources[item["seed"]]
    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "task_id": "WR-08",
        "expected_seeds": sorted(expected), "recorded_successes": len(native),
        "missing_recordings": missing_recordings, "native_errors": native_errors,
        "recording_failures_history": failed_attempts,
        "total_control_frames": sum(row["steps"] for row in native.values()),
        "all_mp4_frames_verified": args.verify_videos,
        "source_snapshot": str(args.source_snapshot.resolve()),
        "source_unchanged": not changed, "changed_source_files": changed,
        "recording_provenance": provenance,
        "missing_replays": missing_replays, "replay_errors": replay_errors,
        "interrupted_replay_attempts": cancelled, "replay_summary": aggregate,
        "publishable": bool(not missing_recordings and not native_errors and not changed
                            and not replay_errors and not missing_replays
                            and aggregate["publishable"] and aggregate["success_rate"] == 1.0
                            and aggregate.get("report_stages", {}).get("order") == list(PUBLIC_STAGES)
                            and (provenance is None or provenance["verified"])
                            and args.verify_videos),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    atomic_json_commit(args.output_dir / "summary.json", report)
    atomic_json_commit(args.output_dir / "manifest.json", {
        "schema_version": "membench.dataset/v2", "dataset_id": args.output_dir.name,
        "recording_provenance": str(args.recording_provenance.resolve()) if args.recording_provenance else None,
        "items": sorted(items, key=lambda item: item["seed"]),
    })
    atomic_json_commit(args.output_dir / "native_integrity.json", {
        "episodes": [native[seed] for seed in sorted(native)], "errors": native_errors,
    })
    print(json.dumps({"recorded": len(native), "replayed": len(results),
                      "missing_recordings": len(missing_recordings), "missing_replays": len(missing_replays),
                      "native_errors": len(native_errors), "replay_errors": len(replay_errors),
                      "source_changed": changed, "publishable": report["publishable"]}), flush=True)
    return 0 if report["publishable"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
