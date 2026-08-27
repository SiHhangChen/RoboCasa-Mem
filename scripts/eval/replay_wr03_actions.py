#!/usr/bin/env python3
"""Replay saved WR-03 policy actions with the current environment evaluator."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import argparse
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


WR03_REPLAY_PROFILE = "wr03-six-stage-saved-actions-replay-v1"


def _disable_camera_observables(runtime) -> int:
    """Skip offscreen camera rendering during action-only physics replay."""
    env = runtime.env
    observables = getattr(env, "_observables", {})
    disabled = 0
    for name, observable in observables.items():
        if "_image" in str(name) or "_depth" in str(name):
            setter = getattr(observable, "set_enabled", None)
            if callable(setter) and observable.is_enabled():
                setter(False)
                disabled += 1
    return disabled


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_run", type=Path)
    parser.add_argument("output_run", type=Path)
    parser.add_argument(
        "--seeds",
        default=None,
        help="optional comma-separated seed list or inclusive range, e.g. 1050-1052",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="optional cap on replayed actions per episode (defaults to all saved actions)",
    )
    return parser.parse_args()


def _parse_seeds(value: str | None) -> set[int] | None:
    if value is None:
        return None
    selected: set[int] = set()
    for token in str(value).replace(";", ",").split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            start, end = (int(part) for part in token.split("-", 1))
            selected.update(range(min(start, end), max(start, end) + 1))
        else:
            selected.add(int(token))
    return selected


def main() -> int:
    args = parse_args()
    source = args.source_run.expanduser().resolve()
    output = args.output_run.expanduser().resolve()
    selected_seeds = _parse_seeds(args.seeds)
    if args.max_steps is not None and args.max_steps < 1:
        raise ValueError("--max-steps must be positive")
    manifest = json.loads((source / "run_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("task", {}).get("id") != "WR-03":
        raise ValueError("replay_wr03_actions.py only accepts WR-03 runs")

    action_files = sorted(
        path
        for path in (source / "episodes").glob("seed_*/actions.npz")
        if selected_seeds is None
        or int(path.parent.name.removeprefix("seed_")) in selected_seeds
    )
    seeds = tuple(int(path.parent.name.removeprefix("seed_")) for path in action_files)
    if not seeds:
        raise FileNotFoundError(f"no saved actions found under {source / 'episodes'}")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output exists and is non-empty: {output}")

    suite = load_eval_suite(Path(manifest["suite"]["path"]))
    contract = PolicyContract.from_hello(manifest["policy_contract"])
    adapter = get_adapter("WR-03")
    writer = ArtifactWriter(output, expected_seeds=seeds)

    replay_manifest = dict(manifest)
    replay_rollout = dict(manifest.get("rollout", {}))
    replay_rollout.update(
        {
            "stop_on_success": False,
            "full_horizon_terminal_validation": True,
        }
    )
    if args.max_steps is not None:
        replay_rollout["max_steps"] = int(args.max_steps)
    replay_manifest.update(
        {
            "run_id": output.name,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "seeds": list(seeds),
            "replay": {
                "source_run": str(source),
                "source_run_id": manifest.get("run_id"),
                "mode": "saved_actions_environment_rescore",
                "evaluator_profile": WR03_REPLAY_PROFILE,
                "max_steps_cap": args.max_steps,
                "camera_observables_disabled": True,
            },
            "rollout": replay_rollout,
            "artifacts": {"record_video": False, "video_cameras": [], "video_fps": 20},
        }
    )
    writer.write_run_manifest(replay_manifest)

    for index, (seed, action_path) in enumerate(zip(seeds, action_files), start=1):
        print(f"[replay] [{index}/{len(seeds)}] seed={seed:06d}", flush=True)
        runtime = adapter.create_episode(
            seed=seed,
            policy_contract=contract,
            suite=suite,
        )
        _disable_camera_observables(runtime)
        session = writer.start_episode(
            seed=seed,
            record_video=False,
            video_cameras=(),
            video_fps=20,
        )
        started = time.monotonic()
        actions = np.load(action_path)["actions"]
        if actions.ndim != 2 or actions.shape[1] != 13:
            raise ValueError(f"unexpected action shape for seed {seed}: {actions.shape}")
        if args.max_steps is not None:
            actions = actions[: args.max_steps]
        diagnostics: dict = {}
        success_step = None
        consumed = 0
        failure_reason = None
        try:
            shared = runtime.metadata["shared_runtime"]
            for step, action in enumerate(actions, start=1):
                transition = shared.step(action)
                session.append_action(action)
                consumed = step
                if transition.diagnostics is not None:
                    diagnostics = dict(transition.diagnostics.details or {})
                    if transition.diagnostics.success and success_step is None:
                        success_step = step
                    if (
                        transition.diagnostics.terminal
                        and not transition.diagnostics.success
                    ):
                        failure_reason = (
                            transition.diagnostics.failure_code or "task_terminal"
                        )
                        break

            if not diagnostics:
                diagnostics = dict(adapter.diagnostics(runtime))
            success = bool(diagnostics.get("success") is True and failure_reason is None)
            used = np.asarray(actions[:consumed], dtype=np.float32)
            source_result_path = action_path.parent / "result.json"
            source_result = (
                json.loads(source_result_path.read_text(encoding="utf-8"))
                if source_result_path.is_file()
                else {}
            )
            result = EpisodeResult(
                seed=seed,
                episode_id=f"wr-03-seed-{seed:06d}",
                task_id="WR-03",
                suite_version=suite.suite_version,
                status="success" if success else "failure",
                success=success,
                success_step=success_step,
                env_steps=consumed,
                policy_calls=0,
                reward_sum=0.0,
                failure_reason=None if success else str(
                    failure_reason or (
                        "terminal_state_not_preserved"
                        if success_step is not None
                        else "saved_actions_exhausted"
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
                episode_metadata=source_result.get(
                    "episode_metadata", runtime.result_metadata
                ),
            )
            session.finish(asdict(result))
            print(
                f"[replay] seed={seed:06d} "
                f"{'SUCCESS' if success else 'FAILURE'} steps={consumed} "
                f"reason={result.failure_reason}",
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
