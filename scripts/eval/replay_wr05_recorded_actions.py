#!/usr/bin/env python3
"""Replay WR-05 native expert actions against the current environment."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bench.evaluation.artifacts import ArtifactWriter
from bench.evaluation.rollout import EpisodeResult
from bench.runtime import RoboCasaRuntimeFactory
from bench.tasks import get_task_definition, resolve_builtin_task_plugin
from bench.tasks.contracts import EpisodeRequest


def _episodes(root: Path, seeds: set[int] | None) -> list[tuple[int, Path]]:
    result = []
    for path in sorted((root / "train" / "WR-05").glob("seed_*_attempt_*")):
        trajectory = path / "trajectory.h5"
        if not trajectory.is_file():
            continue
        seed = int(path.name.removeprefix("seed_").split("_")[0])
        if seeds and seed not in seeds:
            continue
        result.append((seed, path))
    return result


def _actions(path: Path) -> np.ndarray:
    with h5py.File(path / "trajectory.h5", "r") as handle:
        value = np.asarray(handle["data/demo_1/actions"], dtype=np.float32)
    if value.ndim != 2 or value.shape[1] != 13 or not np.all(np.isfinite(value)):
        raise ValueError(f"invalid WR-05 action array: {value.shape}")
    return np.ascontiguousarray(value)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=ROOT / "outputs/datasets/WR05")
    parser.add_argument("--output-run", type=Path, required=True)
    parser.add_argument("--seeds", default=None)
    args = parser.parse_args()
    wanted = None
    if args.seeds is not None:
        wanted = set()
        for item in str(args.seeds).replace(";", ",").split(","):
            item = item.strip()
            if not item:
                continue
            if "-" in item:
                start, end = (int(part) for part in item.split("-", 1))
                wanted.update(range(start, end + (1 if end >= start else -1), 1 if end >= start else -1))
            else:
                wanted.add(int(item))
    episodes = _episodes(args.dataset_root.resolve(), wanted)
    if not episodes:
        raise FileNotFoundError("no WR-05 trajectories found")
    output = args.output_run.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output exists and is non-empty: {output}")
    definition = get_task_definition("WR-05")
    api = resolve_builtin_task_plugin("WR-05").load_runtime_api()
    writer = ArtifactWriter(output, expected_seeds=tuple(seed for seed, _ in episodes))
    writer.write_run_manifest({
        "schema_version": "membench.run_manifest/v1", "run_id": output.name,
        "created_at": datetime.now(timezone.utc).isoformat(), "task": {"id": "WR-05", "env_name": definition.env_name},
        "seeds": [seed for seed, _ in episodes],
        "replay": {"mode": "recorded_actions_action_only_physics", "action_source": "trajectory.h5[data/demo_1/actions]"},
        "rollout": {"action_repeat": 1, "stop_on_success": False, "full_horizon_terminal_validation": True},
    })
    for seed, directory in episodes:
        actions = _actions(directory)
        episode = definition.resolve(EpisodeRequest(task_id="WR-05", seed=seed, variant="memory", split="train", env_split="pretrain", allow_partial_calibration=False))
        descriptor = api.build_descriptor(episode, execute=True)
        # No expert runtime is created: every transition below comes directly
        # from the recorded action vector and the simulator's contact model.
        runtime = RoboCasaRuntimeFactory(api).create_from_descriptor(descriptor, expert=False)
        video_cameras = ("robot0_agentview_left", "robot0_agentview_right", "robot0_eye_in_hand")
        session = writer.start_episode(seed=seed, record_video=True, video_cameras=video_cameras, video_fps=20)
        started = time.monotonic(); diagnostics = {}; consumed = 0; success_step = None
        try:
            for step, action in enumerate(actions, 1):
                transition = runtime.step(action); session.append_action(action); consumed = step
                observation = transition.observation or {}
                session.append_frames({camera: observation[f"{camera}_image"] for camera in video_cameras})
                diagnostics = dict(transition.diagnostics.details or {})
                if transition.diagnostics.success and success_step is None:
                    success_step = step
            used = actions[:consumed]
            success = bool(diagnostics.get("success") is True)
            result = EpisodeResult(seed=seed, episode_id=f"wr-05-seed-{seed:06d}", task_id="WR-05", suite_version="wr05-recorded-action-replay-v1", status="success" if success else "failure", success=success, success_step=success_step, env_steps=consumed, policy_calls=0, reward_sum=0.0, failure_reason=None if success else "recorded_actions_exhausted", diagnostics=diagnostics, action_mean=used.mean(0).tolist(), action_std=used.std(0).tolist(), action_min=used.min(0).tolist(), action_max=used.max(0).tolist(), policy_actions_consumed=consumed, duration_s=time.monotonic()-started)
            session.finish(asdict(result))
        finally:
            runtime.close()
    print(json.dumps(writer.write_summary(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
