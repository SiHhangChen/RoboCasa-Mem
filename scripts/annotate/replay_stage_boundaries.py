#!/usr/bin/env python
"""Replay expert episodes WITHOUT recording, emit stage boundary frames.

For each requested seed, runs the deterministic expert plan against a fresh
RoboCasa runtime, wraps ``env.step`` to count frames, and records each stage's
end frame via ``on_stage_end``. Output is one JSON object per line (jsonl):

    {"task_id": "...", "seed": 0, "success": true, "total_frames": 2105,
     "stages": [{"index": 0, "name": "navigate_pick_sugar",
                 "kind": "navigate", "end_frame": 0}, ...]}

Stage boundaries recovered here are applied to already-recorded episodes by
``apply_subtask_labels.py``. Measured replay-vs-recorded frame drift is a few
frames per episode, which is negligible for segment-level subtask labels.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bench.runtime import RoboCasaRuntimeFactory
from bench.tasks import get_task_definition, resolve_builtin_task_plugin
from bench.tasks.contracts import EpisodeRequest, ExpertPlanInputs


def _parse_options(values: list[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for item in values:
        key, separator, value = str(item).partition("=")
        if not separator or not key.strip():
            raise ValueError(f"--task-option must be KEY=VALUE, got {item!r}")
        result[key.strip()] = value
    return result


# Per-process worker state, initialized via Pool(initializer=_init_worker).
_WORKER: dict = {}


def _init_worker(task_id: str, variant: str, split: str, env_split: str,
                 options: dict[str, str]) -> None:
    _WORKER["definition"] = get_task_definition(task_id)
    plugin = resolve_builtin_task_plugin(task_id)
    _WORKER["plugin"] = plugin
    _WORKER["runtime_api"] = plugin.load_runtime_api()
    _WORKER["task_id"] = task_id
    _WORKER["variant"] = variant
    _WORKER["split"] = split
    _WORKER["env_split"] = env_split
    _WORKER["options"] = options


def _replay(seed: int) -> dict:
    definition = _WORKER["definition"]
    runtime_api = _WORKER["runtime_api"]
    task_id = _WORKER["task_id"]
    request = EpisodeRequest(
        task_id=task_id,
        seed=seed,
        variant=_WORKER["variant"],
        split=_WORKER["split"],
        env_split=_WORKER["env_split"],
        options=dict(_WORKER["options"]),
        allow_partial_calibration=False,
    )
    episode = definition.resolve(request)
    descriptor = runtime_api.build_descriptor(episode, execute=True)
    plan = definition.build_expert_plan(ExpertPlanInputs.from_descriptor(descriptor))

    runtime = RoboCasaRuntimeFactory(runtime_api).create_from_descriptor(
        descriptor, expert=True
    )
    try:
        runner = runtime.initialized.runner
        env = runtime.env
        orig_step = env.step
        counter = {"n": 0}

        def wrapped_step(action):
            counter["n"] += 1
            return orig_step(action)

        env.step = wrapped_step
        boundaries: list[dict] = []

        def on_stage_end(index, stage, result):
            boundaries.append({
                "index": int(index),
                "name": str(stage.name),
                "kind": str(getattr(stage, "kind", "")),
                "end_frame": int(counter["n"]),
            })

        t0 = time.time()
        result = runner.run(plan, on_stage_end=on_stage_end)
        elapsed = time.time() - t0
        env.step = orig_step
        return {
            "task_id": task_id,
            "seed": seed,
            "success": bool(result.success),
            "total_frames": int(counter["n"]),
            "elapsed_s": round(elapsed, 1),
            "stages": boundaries,
        }
    finally:
        runtime.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-id", default="WA-01")
    parser.add_argument("--seeds", default="0", help="comma-separated seeds")
    parser.add_argument("--variant", default="memory")
    parser.add_argument("--split", default="train")
    parser.add_argument("--env-split", default="pretrain")
    parser.add_argument("--task-option", action="append", default=[])
    parser.add_argument("--output", type=Path, required=True,
                        help="jsonl output path (append mode)")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    options = _parse_options(args.task_option)
    init_args = (args.task_id, args.variant, args.split, args.env_split, options)

    if args.workers <= 1:
        _init_worker(*init_args)
        iterator = (_replay(seed) for seed in seeds)
    else:
        ctx = mp.get_context("fork")
        pool = ctx.Pool(args.workers, initializer=_init_worker, initargs=init_args)
        iterator = pool.imap(_replay, seeds)

    try:
        for record in iterator:
            with args.output.open("a", encoding="utf-8") as fp:
                fp.write(json.dumps(record, sort_keys=True) + "\n")
            print(f"SEED {record['seed']} success={record['success']} "
                  f"total={record['total_frames']} stages={len(record['stages'])} "
                  f"elapsed={record['elapsed_s']}s", flush=True)
    finally:
        if args.workers > 1:
            pool.close()
            pool.join()


if __name__ == "__main__":
    main()
