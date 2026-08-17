#!/usr/bin/env python3
"""Replay a saved WA-03 trajectory against the current task evaluator.

This is a single-seed debugging helper: it replays the exact actions from a
recorded WA-03 episode's trajectory.h5 in the current RoboCasa runtime, lets the
task evaluator run per step, and prints the full staged diagnostics at the end.

Usage:
    python scripts/eval/replay_wa03_debug.py [--seed 0] [--dataset-root ...]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bench.runtime import RoboCasaRuntimeFactory
from bench.tasks import get_task_definition, resolve_builtin_task_plugin
from bench.tasks.contracts import EpisodeRequest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=ROOT / "outputs" / "datasets" / "WA03_unmarked_200seeds_v1",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=None)
    return parser.parse_args()


def find_episode(dataset_root: Path, seed: int) -> Path:
    directory = dataset_root / "train" / "WA-03" / f"seed_{seed:06d}_attempt_00"
    if not (directory / "trajectory.h5").is_file():
        raise FileNotFoundError(f"no WA-03 recording for seed {seed} under {dataset_root}")
    return directory


def main() -> int:
    args = parse_args()
    dataset_root = args.dataset_root.expanduser().resolve()
    episode_dir = find_episode(dataset_root, args.seed)

    metadata = json.loads((episode_dir / "episode.json").read_text(encoding="utf-8"))
    meta = metadata.get("metadata", {})
    resolved = meta.get("resolved_episode", {})
    task_config_file = Path(resolved.get("task_config_file"))
    profile_file = Path(resolved.get("observation_profile_file"))

    definition = get_task_definition("WA-03")
    runtime_api = resolve_builtin_task_plugin("WA-03").load_runtime_api()
    episode = definition.resolve(
        EpisodeRequest(
            task_id="WA-03",
            seed=int(args.seed),
            variant=str(meta.get("variant", "memory")),
            split="evaluation",
            env_split=str(meta.get("env_split", "pretrain")),
            task_config_file=task_config_file,
            observation_profile_file=profile_file,
            allow_partial_calibration=False,
        )
    )
    descriptor = runtime_api.build_descriptor(episode, execute=True)
    runtime = RoboCasaRuntimeFactory(runtime_api).create_from_descriptor(
        descriptor, expert=False
    )
    with h5py.File(episode_dir / "trajectory.h5", "r") as handle:
        actions = np.asarray(handle["data/demo_1/actions"], dtype=np.float32)
    if actions.ndim != 2 or actions.shape[1] != 13 or not np.all(np.isfinite(actions)):
        runtime.close()
        raise ValueError(f"invalid WA-03 action array: {actions.shape}")
    replay = actions if args.max_steps is None else actions[: args.max_steps]

    success_step = None
    diagnostics: dict = {}
    try:
        for step, action in enumerate(replay, start=1):
            transition = runtime.step(action)
            diagnostics = dict(transition.diagnostics.details or {})
            if transition.diagnostics.success and success_step is None:
                success_step = step
            if step % 500 == 0:
                qualified = diagnostics.get("stage_checks", {})
                print(
                    f"  step={step} qualified={qualified}",
                    flush=True,
                )
    finally:
        runtime.close()

    success = bool(diagnostics.get("success") is True)
    print("\n=== WA-03 replay result ===")
    print(f"success: {success}  success_step: {success_step}  steps: {len(replay)}")
    print(f"failure_code: {diagnostics.get('failure_code')}")
    print(f"success_checks: {diagnostics.get('success_checks')}")
    print("stages:")
    for name, values in (diagnostics.get("stages") or {}).items():
        print(f"  {name:12s} observed={values.get('observed')} "
              f"qualified={values.get('qualified')} "
              f"first={values.get('first_observed_step')} "
              f"qualified_step={values.get('qualified_step')} "
              f"reason={values.get('reason')}")
    print("rejected:")
    for name, entries in (diagnostics.get("evaluation", {}).get("rejected_stage_observations") or {}).items():
        print(f"  {name}: {entries}")
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
