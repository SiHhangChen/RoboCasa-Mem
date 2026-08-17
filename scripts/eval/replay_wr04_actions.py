#!/usr/bin/env python3
"""Replay saved WR-04 actions with the current environment-side evaluator."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bench.evaluation.adapter import get_adapter
from bench.evaluation.artifacts import ArtifactWriter
from bench.evaluation.rollout import EpisodeResult
from bench.evaluation.suite import load_eval_suite
from bench.policy.protocol import PolicyContract


WR04_EVALUATOR_PROFILE = "wr04-full-horizon-v5"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_run", type=Path)
    parser.add_argument("output_run", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source = args.source_run.expanduser().resolve()
    output = args.output_run.expanduser().resolve()
    manifest = json.loads((source / "run_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("task", {}).get("id") != "WR-04":
        raise ValueError("replay_wr04_actions.py only accepts WR-04 runs")
    action_files = sorted((source / "episodes").glob("seed_*/actions.npz"))
    seeds = tuple(int(path.parent.name.removeprefix("seed_")) for path in action_files)
    if not seeds:
        raise FileNotFoundError(f"no saved actions found under {source / 'episodes'}")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output exists and is non-empty: {output}")

    suite = load_eval_suite(Path(manifest["suite"]["path"]))
    contract = PolicyContract.from_hello(manifest["policy_contract"])
    adapter = get_adapter("WR-04")
    writer = ArtifactWriter(output, expected_seeds=seeds)
    replay_manifest = dict(manifest)
    replay_rollout = dict(manifest.get("rollout", {}))
    replay_rollout.update(
        {
            "stop_on_success": False,
            "full_horizon_terminal_validation": True,
        }
    )
    replay_manifest.update(
        {
            "run_id": output.name,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "seeds": list(seeds),
            "replay": {
                "source_run": str(source),
                "source_run_id": manifest.get("run_id"),
                "mode": "saved_actions_environment_rescore",
                "evaluator_profile": WR04_EVALUATOR_PROFILE,
            },
            "rollout": replay_rollout,
            "artifacts": {"record_video": False, "video_cameras": [], "video_fps": 20},
        }
    )
    writer.write_run_manifest(replay_manifest)

    for index, (seed, action_path) in enumerate(zip(seeds, action_files), start=1):
        print(f"[replay] [{index}/{len(seeds)}] seed={seed:06d}", flush=True)
        runtime = adapter.create_episode(seed=seed, policy_contract=contract, suite=suite)
        session = writer.start_episode(
            seed=seed,
            record_video=False,
            video_cameras=(),
            video_fps=20,
        )
        started = time.monotonic()
        actions = np.load(action_path)["actions"]
        success_step = None
        violation_reason = None
        diagnostics: dict = {}
        try:
            shared = runtime.metadata["shared_runtime"]
            consumed = 0
            for step, action in enumerate(actions, start=1):
                transition = shared.step(action)
                session.append_action(action)
                consumed = step
                diagnostics = dict(transition.diagnostics.details)
                if transition.diagnostics.success and success_step is None:
                    success_step = step
                physical_events = diagnostics.get("physical_events", {})
                if physical_events.get("phase") == "invalid":
                    violation_reason = physical_events.get("violation_reason")
                    break
            # Reaching COMPLETE is only a candidate. Consume the entire saved
            # horizon so post-success object removal/repeated stages can revoke
            # it. The robot may move freely if the object remains valid.
            success = bool(
                success_step is not None
                and violation_reason is None
                and diagnostics.get("success") is True
            )
            used = np.asarray(actions[:consumed], dtype=np.float32)
            source_result_path = action_path.parent / "result.json"
            source_result = (
                json.loads(source_result_path.read_text(encoding="utf-8"))
                if source_result_path.is_file()
                else {}
            )
            result = EpisodeResult(
                seed=seed,
                episode_id=f"wr-04-seed-{seed:06d}",
                task_id="WR-04",
                suite_version=suite.suite_version,
                status="success" if success else "failure",
                success=success,
                success_step=success_step,
                env_steps=consumed,
                policy_calls=0,
                reward_sum=0.0,
                failure_reason=(
                    None
                    if success
                    else str(
                        violation_reason
                        or (
                            "terminal_state_not_preserved"
                            if success_step is not None
                            else "saved_actions_exhausted"
                        )
                    )
                ),
                diagnostics=diagnostics,
                action_mean=used.mean(axis=0).tolist(),
                action_std=used.std(axis=0).tolist(),
                action_min=used.min(axis=0).tolist(),
                action_max=used.max(axis=0).tolist(),
                policy_actions_returned=0,
                policy_actions_consumed=consumed,
                duration_s=time.monotonic() - started,
                episode_metadata=source_result.get("episode_metadata", runtime.result_metadata),
            )
            session.finish(asdict(result))
            writer.write_summary()
            print(
                f"[replay] seed={seed:06d} "
                f"{'SUCCESS' if success else 'FAILURE'} steps={consumed}",
                flush=True,
            )
        except BaseException:
            session.abort()
            raise
        finally:
            adapter.close(runtime)

    summary = writer.write_summary()
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"run_dir={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
