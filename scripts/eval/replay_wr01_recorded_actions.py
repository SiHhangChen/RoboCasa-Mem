#!/usr/bin/env python3
"""Replay WR-01 native expert recordings through the action-only runtime.

The replay runtime is created with ``expert=False`` and receives only the
recorded 13-dimensional action stream.  WR-01 reconstructs storage, the hidden
memory barrier, the retrieval-door decision, and the final pick from physical
state on every control step.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bench.recording.replay_validation import load_recorded_actions, REPLAY_MODE
from bench.runtime import RoboCasaRuntimeFactory
from bench.tasks import get_task_definition, resolve_builtin_task_plugin
from bench.tasks.contracts import EpisodeRequest


TASK_ID = "WR-01"
EVALUATOR_PROFILE = "wr01-recorded-action-only-physics-replay-v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=ROOT / "outputs" / "datasets" / "WR01",
    )
    parser.add_argument(
        "--output-run",
        type=Path,
        default=ROOT / "outputs" / "eval" / "wr01-recorded-action-replay-v1",
    )
    parser.add_argument("--seeds", default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--require-episode-identity", action="store_true")
    return parser.parse_args()


def _parse_seeds(value: str | None) -> set[int]:
    if value is None:
        return set()
    result: set[int] = set()
    for token in str(value).replace(";", ",").split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            start, end = (int(part) for part in token.split("-", 1))
            result.update(range(start, end + (1 if end >= start else -1), 1 if end >= start else -1))
        else:
            result.add(int(token))
    return result


def _resolve_episode_dir(root: Path, value: str | Path, *, seed: int, attempt: int) -> Path:
    candidate = Path(value)
    options = [
        candidate,
        root / candidate,
        root / "train" / TASK_ID / f"seed_{seed:06d}_attempt_{attempt:02d}",
        root / "train" / "WR-01" / f"seed_{seed:06d}_attempt_{attempt:02d}",
    ]
    for path in options:
        if (path / "trajectory.h5").is_file():
            return path
    raise FileNotFoundError(
        f"no trajectory.h5 for seed={seed} under {root}; checked "
        + ", ".join(str(path) for path in options)
    )


def _discover(root: Path, seed_filter: set[int]) -> list[dict[str, Any]]:
    manifest_path = root / "manifest.json"
    items: list[dict[str, Any]] = []
    if manifest_path.is_file():
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        for item in payload.get("items", []):
            if str(item.get("task_id", "")).upper().replace("-", "") != "WR01":
                continue
            seed = int(item["seed"])
            if seed_filter and seed not in seed_filter:
                continue
            attempt = int(item.get("attempt", 0))
            items.append(
                {
                    "seed": seed,
                    "attempt": attempt,
                    "episode_dir": _resolve_episode_dir(
                        root, item.get("episode_dir", ""), seed=seed, attempt=attempt
                    ),
                    "episode_identity": item.get("episode_identity"),
                }
            )
    if not items:
        for directory in sorted((root / "train" / "WR-01").glob("seed_*_attempt_*")):
            if not (directory / "trajectory.h5").is_file():
                continue
            seed = int(directory.name.removeprefix("seed_").split("_", 1)[0])
            if not seed_filter or seed in seed_filter:
                items.append({"seed": seed, "attempt": 0, "episode_dir": directory})
    return sorted(items, key=lambda item: (int(item["seed"]), int(item["attempt"])))


def _recording_metadata(directory: Path) -> dict[str, Any]:
    path = directory / "episode.json"
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    metadata = dict(payload.get("metadata", {}) or {})
    return {
        "episode_identity": metadata.get("episode_identity"),
        "variant": metadata.get("variant", "memory"),
        "split": metadata.get("split", "train"),
        "env_split": metadata.get("env_split", "pretrain"),
        "recorded_steps": payload.get("step_count"),
    }


def _action_stats(actions: np.ndarray) -> dict[str, list[float]]:
    if actions.size == 0:
        empty = [0.0] * 13
        return {"mean": empty, "std": empty, "min": empty, "max": empty}
    return {
        "mean": np.mean(actions, axis=0).astype(float).tolist(),
        "std": np.std(actions, axis=0).astype(float).tolist(),
        "min": np.min(actions, axis=0).astype(float).tolist(),
        "max": np.max(actions, axis=0).astype(float).tolist(),
    }


def main() -> int:
    args = parse_args()
    if args.max_steps is not None and args.max_steps <= 0:
        raise ValueError("--max-steps must be positive")
    root = args.dataset_root.expanduser().resolve()
    output = args.output_run.expanduser().resolve()
    episodes = _discover(root, _parse_seeds(args.seeds))
    if not episodes:
        raise FileNotFoundError(f"no recorded WR-01 episodes found under {root}")
    output.mkdir(parents=True, exist_ok=True)

    definition = get_task_definition(TASK_ID)
    api = resolve_builtin_task_plugin(TASK_ID).load_runtime_api()
    results: list[dict[str, Any]] = []
    manifest = {
        "schema_version": "membench.replay_run/v1",
        "run_id": output.name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "task_id": TASK_ID,
        "dataset_root": str(root),
        "replay": {
            "mode": REPLAY_MODE,
            "evaluator_profile": EVALUATOR_PROFILE,
            "execution_mode": "action_only_physics",
            "expert_runtime": False,
        },
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    for index, item in enumerate(episodes, start=1):
        seed = int(item["seed"])
        directory = Path(item["episode_dir"])
        metadata = _recording_metadata(directory)
        actions = load_recorded_actions(directory / "trajectory.h5")
        request = EpisodeRequest(
            task_id=TASK_ID,
            seed=seed,
            variant=str(metadata.get("variant", "memory")),
            split=str(metadata.get("split", "train")),
            env_split=str(metadata.get("env_split", "pretrain")),
        )
        episode = definition.resolve(request)
        recorded_identity = metadata.get("episode_identity") or item.get("episode_identity")
        if recorded_identity is not None and recorded_identity != episode.identity:
            message = (
                f"episode identity mismatch seed={seed}: "
                f"recorded={recorded_identity} current={episode.identity}"
            )
            if args.require_episode_identity:
                raise ValueError(message)
            print(f"[replay] WARNING {message}", flush=True)

        runtime = None
        started = time.monotonic()
        consumed = 0
        success_step: int | None = None
        diagnostics: dict[str, Any] = {}
        error: str | None = None
        try:
            descriptor = api.build_descriptor(episode, execute=True)
            runtime = RoboCasaRuntimeFactory(api).create_from_descriptor(
                descriptor, expert=False
            )
            replay_actions = actions if args.max_steps is None else actions[: args.max_steps]
            for step, action in enumerate(replay_actions, start=1):
                transition = runtime.step(action)
                consumed = step
                diagnostics = dict(transition.diagnostics.details or {})
                if transition.diagnostics.success and success_step is None:
                    success_step = step
                if step % 250 == 0:
                    storage = diagnostics.get("storage", {})
                    retrieval = diagnostics.get("retrieval", {})
                    physical = diagnostics.get("physical", {})
                    print(
                        f"[replay] seed={seed:06d} step={step} "
                        f"stored={storage.get('stored_levels_by_item', {})} "
                        f"hidden={storage.get('storage_hidden', False)} "
                        f"doors={storage.get('doors', {})} "
                        f"door={retrieval.get('opened_level')} "
                        f"retrieved={retrieval.get('retrieved_item_id')} "
                        f"collisions={physical.get('object_collisions_enabled')} "
                        f"positions={physical.get('item_world_positions', {})}",
                        flush=True,
                    )
            success = bool(diagnostics.get("success") is True)
        except BaseException as exc:
            success = False
            error = f"{type(exc).__name__}: {exc}"
        finally:
            if runtime is not None:
                runtime.close()
        result = {
            "seed": seed,
            "attempt": int(item.get("attempt", 0)),
            "episode_dir": str(directory),
            "recorded_steps": int(actions.shape[0]),
            "replayed_steps": consumed,
            "success": success,
            "success_step": success_step,
            "failure_reason": None if success else (error or "recorded_actions_exhausted"),
            "duration_s": time.monotonic() - started,
            "episode_identity": episode.identity,
            "recorded_episode_identity": recorded_identity,
            "episode_identity_matches": recorded_identity is not None and recorded_identity == episode.identity,
            "action_stats": _action_stats(actions[:consumed]),
            "diagnostics": diagnostics,
        }
        results.append(result)
        (output / f"seed_{seed:06d}.json").write_text(json.dumps(result, indent=2, sort_keys=True, default=str) + "\n")
        print(
            f"[replay] [{index}/{len(episodes)}] seed={seed:06d} "
            f"{'SUCCESS' if success else 'FAILURE'} steps={consumed} "
            f"reason={result['failure_reason']}",
            flush=True,
        )

    summary = {
        "task_id": TASK_ID,
        "episodes": len(results),
        "successes": sum(bool(item["success"]) for item in results),
        "failures": sum(not bool(item["success"]) for item in results),
        "results": results,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n")
    print(json.dumps({key: summary[key] for key in ("task_id", "episodes", "successes", "failures")}, sort_keys=True))
    return 0 if summary["failures"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
