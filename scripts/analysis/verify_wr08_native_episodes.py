#!/usr/bin/env python3
"""Check finalized WR-08 recordings, including actual HDF5 and MP4 lengths.

This is a data-integrity audit, not a replacement for production action replay.
Staging directories are deliberately excluded by the canonical discovery API.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from fractions import Fraction
import json
from pathlib import Path
import subprocess
import sys

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from bench.evaluation.replay import discover_recorded_episodes, parse_seed_spec
from bench.tasks.wr08.domain.contracts import CARTON_ASSET_BY_ITEM, TASK_REVISION

CAMERAS = ("robot0_agentview_left", "robot0_agentview_right", "robot0_eye_in_hand")


def check(condition, message):
    if not condition:
        raise ValueError(message)


def verify(episode_dir: Path, *, videos: bool) -> dict:
    episode = json.loads((episode_dir / "episode.json").read_text())
    planner = json.loads((episode_dir / "planner.json").read_text())
    metadata = episode["metadata"]
    plan_metadata = metadata["expert_plan"]["metadata"]
    check(metadata["task_id"] == "WR-08", "wrong task")
    check(plan_metadata["task_revision"] == TASK_REVISION, "wrong task revision")
    check(plan_metadata["carton_asset_by_item"] == dict(CARTON_ASSET_BY_ITEM), "wrong carton assets")
    check(episode["accepted"] and episode["planner_pass"], "episode was not accepted")
    check(episode["termination"]["success"], "recorded termination was not successful")
    check(planner["episode_result"]["success"] and planner["final_diagnostics"]["success"], "planner/semantics failed")
    stages = planner["rule_policy"]["operations"]
    check(planner["episode_result"]["stage_count"] == len(stages) == 17, "expected 17 internal stages")
    check(all(s["success"] for s in stages), "failed internal stage")
    n = int(episode["step_count"])
    check(n > 0, "empty recording")
    check(episode["video"]["fps"] == 20 and episode["video"]["frame_stride"] == 1, "unexpected video rate")
    check(tuple(episode["video"]["cameras"]) == CAMERAS, "wrong camera set")
    with h5py.File(episode_dir / "trajectory.h5", "r") as handle:
        demo = handle["data/demo_1"]
        expected = {"actions": (n, 13), "obs/robot_state": (n, 37)}
        expected.update({name: (n,) for name in (
            "dones", "rewards", "success", "terminated", "truncated", "timestamps/sim_time",
            "diagnostics/frame_json", "oracle/frame_json",
        )})
        for camera in CAMERAS:
            expected[f"calibration/{camera}/intrinsic"] = (n, 3, 3)
            expected[f"calibration/{camera}/extrinsic_c2w"] = (n, 4, 4)
            expected[f"video_frame_indices/{camera}"] = (n,)
        for name, shape in expected.items():
            check(name in demo, f"missing HDF5 dataset: {name}")
            data = demo[name]
            check(data.shape == shape, f"unaligned {name}: {data.shape}, expected {shape}")
            if data.dtype.kind in "fi":
                check(np.all(np.isfinite(data[:])), f"nonfinite values: {name}")
        check(demo["actions"].dtype == np.dtype("float32"), "actions must be float32")
        check(bool(demo["success"][-1]) and bool(demo["terminated"][-1]), "missing successful terminal frame")
        check(not bool(demo["truncated"][-1]), "successful trajectory marked truncated")
        times = np.asarray(demo["timestamps/sim_time"])
        check(np.allclose(np.diff(times), 0.05, atol=1e-5, rtol=0), "simulation timestamps are not 20 Hz")
        # The writer creates an empty depth group even when capture is disabled.
        depth_datasets = []
        if "obs/depth" in demo:
            demo["obs/depth"].visititems(
                lambda name, obj: depth_datasets.append(name)
                if isinstance(obj, h5py.Dataset) and obj.size else None
            )
        check(not depth_datasets, "unexpected recorded depth datasets")
        for camera in CAMERAS:
            check(np.array_equal(demo[f"video_frame_indices/{camera}"][:], np.arange(n)), f"bad frame indices: {camera}")
            check(episode["video"]["frame_counts"][camera] == n, f"video metadata length mismatch: {camera}")
        env_args = json.loads(handle["data"].attrs.get("env_args", "{}"))
        depth_rendered = env_args.get("env_kwargs", {}).get("camera_depths")
    video_info = {}
    for camera in CAMERAS:
        path = episode_dir / "videos" / f"{camera}.mp4"
        check(path.is_file() and path.stat().st_size > 48, f"missing/empty video: {camera}")
        if videos:
            result = subprocess.run([
                "ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames", "-show_entries",
                "stream=nb_frames,nb_read_frames,width,height,r_frame_rate,codec_name", "-of", "json", str(path),
            ], capture_output=True, text=True, check=True, timeout=30)
            check(not getattr(result, "stderr", "").strip(), f"MP4 decode errors: {camera}")
            info = json.loads(result.stdout)["streams"][0]
            check(int(info["nb_frames"]) == n, f"actual MP4 frame count mismatch: {camera}")
            check(int(info["nb_read_frames"]) == n, f"decoded MP4 frame count mismatch: {camera}")
            check((info["width"], info["height"]) == (224, 224), f"wrong video resolution: {camera}")
            check(Fraction(info["r_frame_rate"]) == 20, f"wrong video FPS: {camera}")
            video_info[camera] = info
    return {
        "seed": int(metadata["seed"]), "episode_dir": str(episode_dir.resolve()),
        "episode_identity": metadata["episode_identity"], "steps": n,
        "elapsed_s": episode["elapsed_s"], "internal_stages": len(stages),
        "depth_recorded": False, "depth_rendered": depth_rendered,
        "mp4_verified": videos, "mp4_frames_decoded": videos, "videos": video_info,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, action="append", required=True)
    parser.add_argument("--seeds", default=None)
    parser.add_argument("--verify-videos", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    seen_paths, seen_seeds, verified, failures = set(), set(), [], []
    for root in args.dataset_root:
        for item in discover_recorded_episodes(root.resolve(), "WR-08", parse_seed_spec(args.seeds), source_status="success"):
            path = Path(item["episode_dir"]).resolve()
            if path in seen_paths:
                continue
            seen_paths.add(path)
            try:
                row = verify(path, videos=args.verify_videos)
                check(row["seed"] == item["seed"], "directory/metadata seed mismatch")
                check(row["seed"] not in seen_seeds, "duplicate successful seed")
                seen_seeds.add(row["seed"])
                verified.append(row)
            except Exception as exc:
                failures.append({"seed": item["seed"], "episode_dir": str(path), "error": f"{type(exc).__name__}: {exc}"})
    verified.sort(key=lambda row: row["seed"])
    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "native_data_integrity_only", "task_revision": TASK_REVISION,
        "episode_count": len(seen_paths), "verified_count": len(verified),
        "passed": bool(verified) and not failures,
        "total_control_frames": sum(row["steps"] for row in verified),
        "mp4_verified": args.verify_videos, "failures": failures, "episodes": verified,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({
        **{k: v for k, v in report.items() if k not in {"episodes", "failures"}},
        "failure_count": len(failures), "first_failures": failures[:5],
    }), flush=True)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
