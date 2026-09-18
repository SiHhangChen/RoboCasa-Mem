#!/usr/bin/env python3
"""Profile saved WR08 actions, or compare old/new evidence at identical states.

No policy server is required. Setup and inference are excluded; cameras stay
enabled unless --no-cameras is supplied. Outputs are profiling data, not scores.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import tempfile
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bench.evaluation.adapter import get_adapter
from bench.evaluation.artifacts import ArtifactWriter
from bench.evaluation.suite import load_eval_suite
from bench.policy.protocol import PolicyContract
from bench.tasks.wr08.contact_evidence import ContactIndex


class Timings:
    def __init__(self):
        self.values = {}
        self.originals = []

    def wrap(self, owner, name, label):
        original = getattr(owner, name)
        had_own = name in vars(owner)

        def measured(*args, **kwargs):
            start = time.perf_counter()
            try:
                return original(*args, **kwargs)
            finally:
                item = self.values.setdefault(label, {"calls": 0, "seconds": 0.0})
                item["calls"] += 1
                item["seconds"] += time.perf_counter() - start

        self.originals.append((owner, name, original, had_own))
        setattr(owner, name, measured)

    def restore(self):
        for owner, name, original, had_own in reversed(self.originals):
            if had_own:
                setattr(owner, name, original)
            else:
                delattr(owner, name)


def replay(adapter, suite, contract, action_file, args):
    seed = int(action_file.parent.name.removeprefix("seed_"))
    runtime = adapter.create_episode(seed=seed, policy_contract=contract, suite=suite)
    shared, env = runtime.metadata["shared_runtime"], runtime.env
    timings = Timings()
    consumed = verified = 0
    ncon_sum = ncon_max = 0
    try:
        if args.no_cameras:
            for name, observable in env._observables.items():
                if "_image" in name or "_depth" in name:
                    observable.set_enabled(False)
        old_evaluator = copy.deepcopy(env._wr08_evaluator)
        new_evaluator = copy.deepcopy(env._wr08_evaluator)
        if args.mode == "legacy":
            # Restore pre-optimization uncached snapshots and the eager default
            # success check, without changing the evaluator or physical rules.
            env.evaluation_snapshot = lambda: env._policy_evaluation_snapshot(
                use_contact_index=False
            )
            original_diagnostics = shared.diagnostics

            def legacy_diagnostics():
                result = original_diagnostics()
                shared._runtime_api.check_success(shared.initialized, {})
                return result

            shared.diagnostics = legacy_diagnostics

        for name in ("step", "step1", "step2", "forward"):
            timings.wrap(env.sim, name, "physics")
        timings.wrap(env.sim, "render", "render")
        timings.wrap(env, "_pre_action", "control")
        timings.wrap(env, "evaluation_snapshot", "snapshot_access")
        timings.wrap(env, "_policy_evaluation_snapshot", "snapshot_build")
        timings.wrap(env, "check_contact", "legacy_contact_query")
        timings.wrap(ContactIndex, "_scan", "contact_index_scan")
        timings.wrap(env._wr08_evaluator, "observe", "evaluator")
        with np.load(action_file) as archive:
            actions = archive["actions"]
        if actions.ndim != 2 or actions.shape[1] != 13:
            raise ValueError(f"invalid action shape: {actions.shape}")
        if args.max_steps:
            actions = actions[: args.max_steps]
        with tempfile.TemporaryDirectory(prefix="wr08-profile-") as temp:
            session = ArtifactWriter(Path(temp)).start_episode(
                seed=seed,
                record_video=args.record_video,
                video_cameras=suite.artifacts.video_cameras,
                video_fps=suite.artifacts.video_fps,
            )
            timings.wrap(session, "append_frames", "video_append")
            timings.wrap(session, "_close_streams", "video_flush")
            start = time.perf_counter()
            try:
                for action in actions:
                    transition = shared.step(action)
                    runtime.observation = transition.observation
                    session.append_action(action)
                    if args.record_video:
                        session.append_frames(
                            adapter.artifact_frames(runtime, suite.artifacts.video_cameras)
                        )
                    success = adapter.check_success(runtime, transition.info)
                    consumed += 1
                    ncon = int(env.sim.data.ncon)
                    ncon_sum += ncon
                    ncon_max = max(ncon_max, ncon)
                    if args.mode == "verify":
                        old = env._policy_evaluation_snapshot(use_contact_index=False)
                        new = env.evaluation_snapshot()
                        if asdict(old) != asdict(new):
                            raise AssertionError(
                                f"physical snapshot mismatch at seed={seed}, step={consumed}"
                            )
                        old_result = old_evaluator.observe(old)
                        new_result = new_evaluator.observe(new)
                        if asdict(old_result) != asdict(new_result):
                            raise AssertionError(
                                f"evaluator mismatch at seed={seed}, step={consumed}"
                            )
                        if asdict(new_result) != asdict(env._wr08_evaluator.result()):
                            raise AssertionError(
                                f"runtime evaluator mismatch at seed={seed}, step={consumed}"
                            )
                        verified += 1
                    if consumed % 250 == 0:
                        print(f"seed={seed} progress={consumed}/{len(actions)}", flush=True)
                    if transition.diagnostics.stop_execution:
                        break
                session._close_streams()
                elapsed = time.perf_counter() - start
            finally:
                session.abort()
        return {
            "seed": seed,
            "mode": args.mode,
            "steps": consumed,
            "verified_steps": verified,
            "duration_s": elapsed,
            "ms_per_step": elapsed * 1000 / consumed if consumed else None,
            "ncon_mean": ncon_sum / consumed if consumed else None,
            "ncon_max": ncon_max,
            "success": success if consumed else False,
            "failure_code": env._wr08_evaluator.result().failure_code,
            "timings_inclusive": timings.values,
            "report_stages": env._wr08_evaluator.result().to_report_stages(),
        }
    finally:
        timings.restore()
        adapter.close(runtime)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_run", type=Path)
    parser.add_argument("--seeds", required=True, help="comma-separated completed seeds")
    parser.add_argument("--mode", choices=("legacy", "optimized", "verify"), default="verify")
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--no-cameras", action="store_true")
    parser.add_argument(
        "--record-video",
        action="store_true",
        help="encode temporary videos (discarded after measurement)",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.max_steps is not None and args.max_steps < 1:
        parser.error("--max-steps must be positive")
    if args.no_cameras and args.record_video:
        parser.error("--record-video requires cameras")
    if args.output.exists():
        raise FileExistsError(args.output)
    source = args.source_run.resolve()
    manifest = json.loads((source / "run_manifest.json").read_text())
    if manifest["task"]["id"] != "WR-08":
        raise ValueError("only WR-08 source runs are supported")
    seeds = [int(value) for value in args.seeds.split(",")]
    files = [source / "episodes" / f"seed_{seed:06d}" / "actions.npz" for seed in seeds]
    for file in files:
        if not file.is_file():
            raise FileNotFoundError(file)
    suite = load_eval_suite(Path(manifest["suite"]["path"]))
    contract = PolicyContract.from_hello(manifest["policy_contract"])
    adapter = get_adapter("WR-08")
    rows = []
    for file in files:
        row = replay(adapter, suite, contract, file, args)
        rows.append(row)
        print(
            f"seed={row['seed']} steps={row['steps']} ms/step={row['ms_per_step']:.2f} verified={row['verified_steps']}",
            flush=True,
        )
    report = {
        "source_run": str(source),
        "mode": args.mode,
        "cameras_enabled": not args.no_cameras,
        "video_encoded": args.record_video,
        "inference_included": False,
        "setup_included": False,
        "note": "timings are inclusive and overlap; verify mode includes legacy comparison overhead",
        "episodes": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
