#!/usr/bin/env python3
"""Smoke-test an installed wheel with pinned simulator sources in isolation.

Run with the acceptance environment's Python and -I, outside the checkout.
The acceptance root contains .venv/, sources/robocasa/, sources/robosuite/,
and reports/. Copy this script into that root before starting the check.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
import importlib.abc
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time
import traceback

TASKS = ("WX-01", "WA-01", "WA-05", "WR-01", "WR-08", "TS02", "TS03", "TM01", "TM02")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--acceptance-root", type=Path, required=True)
    parser.add_argument("--task", choices=TASKS, required=True)
    parser.add_argument("--seed", type=int, default=1050)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--render-device", default="1")
    parser.add_argument("--development-root", type=Path, required=True)
    parser.add_argument("--original-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.acceptance_root.resolve()
    if not sys.flags.isolated or not Path(sys.prefix).resolve().is_relative_to(root):
        parser.error("use the acceptance virtualenv's Python with -I")
    if args.steps < 1:
        parser.error("--steps must be positive")
    for path in ("cache", "reports"):
        (root / path).mkdir(exist_ok=True)
    os.environ.update({
        "MUJOCO_GL": "egl", "PYOPENGL_PLATFORM": "egl",
        "MUJOCO_EGL_DEVICE_ID": args.render_device,
        "NUMBA_CACHE_DIR": str(root / "cache/numba"),
        "MPLCONFIGDIR": str(root / "cache/matplotlib"),
        "XDG_CACHE_HOME": str(root / "cache"),
        "HF_HUB_OFFLINE": "1", "WANDB_MODE": "disabled",
        "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
    })
    forbidden = (args.development_root.resolve(), args.original_root.resolve())
    blocked = []

    def audit(event, values):
        if event != "open" or not values or not isinstance(values[0], (str, bytes)):
            return
        path = Path(os.fsdecode(values[0])).resolve()
        if not path.is_relative_to(root) and any(path.is_relative_to(p) for p in forbidden):
            blocked.append(str(path))
            raise PermissionError(f"acceptance forbids development-file access: {path}")

    class NoExperts(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, *unused):
            if ".expert" in fullname or fullname.startswith("bench.recording") or fullname == "mplib":
                raise ImportError(f"acceptance forbids optional expert/planner import: {fullname}")

    sys.addaudithook(audit)
    sys.meta_path.insert(0, NoExperts())
    output = root / "reports" / f"{args.task}-{args.seed}"
    output.mkdir(exist_ok=False)
    started = time.monotonic()
    report = {
        "schema_version": "membench.installed_candidate_smoke/v1",
        "task_id": args.task, "seed": args.seed, "steps": args.steps,
        "python": sys.version, "executable": sys.executable,
        "isolated_python": True, "cwd": str(Path.cwd()),
        "development_paths_blocked": [str(p) for p in forbidden],
        "policy_model_loaded": False,
    }
    try:
        import cv2
        import numpy as np
        import bench
        import robocasa
        import robosuite
        from bench.evaluation.adapter import get_adapter
        from bench.evaluation.artifacts import ArtifactWriter
        from bench.evaluation.release_cli import resolve_suite
        from bench.evaluation.report_stages import validate_report_stages
        from bench.evaluation.rollout import RolloutRunner
        from bench.evaluation.suite import load_eval_suite
        from bench.policy.backends.fake import FakePolicyBackend
        from bench.policy.protocol import InferenceResponse
        from bench.tasks import TaskCatalog

        origins = {name: str(Path(module.__file__).resolve()) for name, module in (
            ("bench", bench), ("robocasa", robocasa), ("robosuite", robosuite)
        )}
        for path in origins.values():
            if not Path(path).is_relative_to(root):
                raise RuntimeError(f"module was loaded outside the acceptance root: {path}")
        if list(TaskCatalog.ids()) != list(TASKS):
            raise RuntimeError("installed wheel does not expose exactly the nine release tasks")
        report["module_origins"] = origins
        report["versions"] = {
            name: importlib.metadata.version(name)
            for name in ("MobileMemBench", "robocasa", "robosuite", "mujoco", "numpy", "numba", "scipy", "Pillow")
        }
        suite = load_eval_suite(resolve_suite("pi05:" + args.task))
        suite = replace(
            suite, seeds=(args.seed,),
            rollout=replace(suite.rollout, max_steps=args.steps, replan_steps=1, stop_on_success=False),
            artifacts=replace(suite.artifacts, record_video=False),
        )
        contract = FakePolicyBackend(horizon=1).contract
        frames = {}

        class HoldPolicy:
            def __init__(self):
                self.contract = contract

            def reset(self, episode_id):
                self.episode_id = episode_id

            def infer(self, observation, *, episode_id, step):
                if observation.state.shape != (37,) or not np.isfinite(observation.state).all():
                    raise RuntimeError("invalid installed runtime state37")
                for camera in contract.observation.required_cameras:
                    frame = np.asarray(observation.images[camera])
                    if frame.shape != (224, 224, 3) or frame.dtype != np.uint8 or float(frame.std()) < 1:
                        raise RuntimeError(f"invalid installed runtime camera: {camera}")
                    frames[camera] = frame.copy()
                action = np.zeros((1, 13), dtype=np.float32)
                action[0, :7] = observation.state[:7]
                action[0, 7] = 1
                return InferenceResponse(
                    request_id=f"hold-{step}", episode_id=episode_id,
                    actions=action, timing={"model_ms": 0.0},
                )

        writer = ArtifactWriter(output / "rollout", expected_seeds=(args.seed,))
        result = RolloutRunner(client=HoldPolicy(), suite=suite, artifacts=writer).run_episode(
            get_adapter(args.task), args.seed
        )
        if result.status not in ("success", "failure") or result.env_steps != args.steps:
            raise RuntimeError(f"installed rollout did not execute {args.steps} steps: {result.error}")
        validate_report_stages(result.report_stages, expected_task_id=args.task, expected_success=result.success)
        if writer.invalid_episode_artifacts():
            raise RuntimeError("installed rollout wrote invalid artifacts")
        image_evidence = {}
        for camera, frame in frames.items():
            image_path = output / f"{camera}.png"
            if not cv2.imwrite(str(image_path), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)):
                raise RuntimeError(f"failed to write image: {image_path}")
            image_evidence[camera] = {
                "file": image_path.name, "shape": list(frame.shape),
                "sha256": hashlib.sha256(image_path.read_bytes()).hexdigest(),
            }
        report.update({
            "passed": True, "control_frequency_hz": contract.control_frequency_hz,
            "state_dim": 37, "action_dim": 13, "images": image_evidence,
            "rollout_status": result.status, "env_steps": result.env_steps,
            "report_stages": result.report_stages,
            "resolved_episode": dict(result.episode_metadata).get("resolved_episode"),
            "note": "Short hold-action acceptance; task success is not the pass criterion.",
        })
    except Exception as exc:
        report.update({"passed": False, "error": str(exc), "traceback": traceback.format_exc()})
    report["blocked_development_accesses"] = blocked
    report["duration_s"] = time.monotonic() - started
    (output / "report.json").write_text(json.dumps(report, indent=2, default=str) + "\n")
    print(json.dumps({key: report.get(key) for key in ("task_id", "seed", "passed", "error", "duration_s")}))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
