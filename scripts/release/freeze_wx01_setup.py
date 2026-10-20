#!/usr/bin/env python3
"""Generate and validate seed-specific WX-01 initialization transcripts.

Run in the development simulator environment with MPlib. The resulting
release artifacts need neither planner nor expert code. Existing artifacts
are not overwritten without --replace. This is a build-time operation.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.release.setup_replay import (
    SCHEMA,
    apply_transcript,
    config_identity,
    observation_digests,
    prepare,
)


def freeze(seed, output):
    from bench.runtime import RoboCasaRuntimeFactory, RobotStateExtractor
    from bench.tasks import TaskCatalog
    from bench.tasks.contracts import EpisodeRequest
    from bench.tasks.wx01 import runtime as wx
    from bench.tasks.wx01.execution import manipulation

    episode = TaskCatalog.resolve(EpisodeRequest(task_id="WX-01", seed=seed))
    builder = manipulation.build_manipulation_assembly
    events = []
    original_methods = []
    inside_step = False

    def traced_builder(env, *args, **kwargs):
        assembly = builder(env, *args, **kwargs)
        step, observe = env.step, env._get_observations
        original_methods.append((env, step, observe))

        def traced_step(action, *a, **kw):
            nonlocal inside_step
            events.append({"kind": "step", "action": np.asarray(action, dtype=float).tolist()})
            inside_step = True
            try:
                return step(action, *a, **kw)
            finally:
                inside_step = False

        def traced_observe(*a, **kw):
            if not inside_step:
                if a:
                    raise ValueError("unsupported positional observation arguments")
                events.append(
                    {"kind": "observe", "force_update": bool(kw.get("force_update", False))}
                )
            return observe(*a, **kw)

        env.step, env._get_observations = traced_step, traced_observe
        return assembly

    manipulation.build_manipulation_assembly = traced_builder
    source = None
    replay = None
    initialize = wx._ADAPTER._initialize
    try:
        source = RoboCasaRuntimeFactory(wx.TASK_RUNTIME_API).create(episode, execute=True)
        for env, step, observe in original_methods:
            env.step, env._get_observations = step, observe
        manipulation.build_manipulation_assembly = builder
        payload = {
            "schema_version": SCHEMA,
            "seed": seed,
            "verified": False,
            "config_identity": config_identity(episode.task_config_file),
            "source_config_sha256": hashlib.sha256(
                Path(episode.task_config_file).read_bytes()
            ).hexdigest(),
            "resource_sha256": {
                name: hashlib.sha256(Path(path).read_bytes()).hexdigest()
                for name, path in episode.resources.items()
                if name != "planner_script"
            },
            "task_version": episode.task_version,
            "source_episode_identity": episode.identity,
            "environment_versions": {
                name: importlib.metadata.version(name) for name in ("mujoco", "numpy")
            },
            "events": events,
            "initialization": source.initialized.initialization,
            "expected": {
                "qpos": source.env.sim.data.qpos.tolist(),
                "qvel": source.env.sim.data.qvel.tolist(),
                "time": float(source.env.sim.data.time),
                "observation_sha256": observation_digests(source.observation),
            },
        }
        from bench.tasks.wx01.execution.geometry import resolve_episode_targets
        from bench.tasks.wx01.initialization import (
            apply_initial_object_plan,
            apply_robot_base_initial_state,
        )

        def frozen_initialize(descriptor, env, observation):
            configure = getattr(env, "configure_episode", None)
            if callable(configure):
                configure(descriptor.episode_plan)
            updated, _objects = apply_initial_object_plan(
                env,
                episode=descriptor.episode_plan,
                enabled=bool(descriptor.config.initial_state.objects.get("enabled", True)),
            )
            updated, _base = apply_robot_base_initial_state(
                env,
                updated,
                config=descriptor.config.initial_state.robot_base,
                navigation=descriptor.navigation,
                seed=descriptor.episode_plan.seed,
            )
            targets = resolve_episode_targets(env, descriptor.episode_plan)
            prepare(env, targets)
            updated = apply_transcript(env, payload)
            evaluator = wx.WX01Evaluator()
            evaluator.reset(env.evaluation_snapshot())
            env._wx01_evaluator = evaluator
            return (
                updated,
                payload["initialization"],
                {wx.TASK_EVALUATOR_METADATA_KEY: evaluator},
                targets,
            )

        wx._ADAPTER._initialize = frozen_initialize
        replay = RoboCasaRuntimeFactory(wx.TASK_RUNTIME_API).create(episode, execute=True)
        initial_equal = (
            observation_digests(replay.observation) == payload["expected"]["observation_sha256"]
        )
        if not initial_equal:
            differences = [
                k
                for k, v in observation_digests(replay.observation).items()
                if v != payload["expected"]["observation_sha256"].get(k)
            ]
            raise RuntimeError(f"WX-01 initial observations differ for seed {seed}: {differences}")
        extractors = (RobotStateExtractor(), RobotStateExtractor())
        np.testing.assert_allclose(
            extractors[0].extract(source.env, source.observation),
            extractors[1].extract(replay.env, replay.observation),
            rtol=0,
            atol=1e-9,
        )
        # A nonzero arm/torso/base motion sequence exposes controller-state
        # differences that static qpos/qvel checks alone would miss.
        right = source.env.robots[0].composite_controller.part_controllers["right"]
        arm = np.asarray(source.env.sim.data.qpos[right.qpos_index]).copy()
        action = np.zeros(13)
        action[:7] = arm
        action[7] = 1.0
        for index in range(20):
            action[0] = arm[0] + 0.01 * np.sin(index / 3)
            action[8] = 0.005 * np.sin(index / 4)
            left = source.step(action)
            other = replay.step(action)
            np.testing.assert_allclose(
                source.env.sim.data.qpos, replay.env.sim.data.qpos, rtol=0, atol=1e-9
            )
            np.testing.assert_allclose(
                extractors[0].extract(source.env, source.observation),
                extractors[1].extract(replay.env, replay.observation),
                rtol=0,
                atol=1e-9,
            )
            if observation_digests(source.observation) != observation_digests(replay.observation):
                raise RuntimeError(f"WX-01 observations differ after action {index}")
            if asdict(left.diagnostics) != asdict(other.diagnostics):
                raise RuntimeError(f"WX-01 diagnostics differ after action {index}")
            if wx.TASK_RUNTIME_API.report_stages(
                source.initialized
            ) != wx.TASK_RUNTIME_API.report_stages(replay.initialized):
                raise RuntimeError(f"WX-01 stage verdicts differ after action {index}")
        payload["verified"] = True
        payload["validation"] = {
            "initial_observations_equal": True,
            "initial_state37_equal": True,
            "replayed_model_actions": 20,
            "trajectory_observations_equal": True,
            "diagnostics_and_stages_equal": True,
            "state_atol": 1e-9,
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(".building")
        temporary.write_text(json.dumps(payload, indent=2) + "\n")
        temporary.replace(output)
        print(
            json.dumps(
                {
                    "seed": seed,
                    "setup_steps": sum(e["kind"] == "step" for e in events),
                    "verified": True,
                    "output": str(output),
                }
            ),
            flush=True,
        )
    finally:
        manipulation.build_manipulation_assembly = builder
        wx._ADAPTER._initialize = initialize
        for runtime in (source, replay):
            if runtime is not None:
                runtime.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(1050, 1100)))
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "configs/benchmark/setup_replays/wx01"
    )
    parser.add_argument("--replace", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Skip existing verified artifacts.")
    parser.add_argument("--workers", type=int, default=1, help="Independent simulator processes.")
    parser.add_argument("--log-dir", type=Path, default=ROOT / "outputs/release/wx01_setup")
    args = parser.parse_args()
    pending = []
    for seed in dict.fromkeys(args.seeds):
        output = args.output_dir / f"{seed}.json"
        if output.exists() and not args.replace:
            payload = json.loads(output.read_text())
            if args.resume and payload.get("verified") and payload.get("seed") == seed:
                print(json.dumps({"seed": seed, "status": "already_verified"}), flush=True)
                continue
            raise FileExistsError(output)
        pending.append(seed)
    if args.workers == 1:
        for seed in pending:
            freeze(seed, args.output_dir / f"{seed}.json")
    elif args.workers > 1:
        args.log_dir.mkdir(parents=True, exist_ok=True)

        def worker(seed):
            log = args.log_dir / f"{seed}.log"
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--seeds",
                str(seed),
                "--output-dir",
                str(args.output_dir.resolve()),
            ]
            if args.replace:
                command.append("--replace")
            with log.open("w") as stream:
                result = subprocess.run(
                    command, stdout=stream, stderr=subprocess.STDOUT, check=False
                )
            if result.returncode:
                raise RuntimeError(f"WX-01 seed {seed} failed; see {log}")
            return seed

        failures = []
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            jobs = {pool.submit(worker, seed): seed for seed in pending}
            for job in as_completed(jobs):
                try:
                    print(json.dumps({"seed": job.result(), "verified": True}), flush=True)
                except (RuntimeError, OSError) as exc:
                    failures.append(str(exc))
                    print(str(exc), file=sys.stderr, flush=True)
        if failures:
            raise RuntimeError("\n".join(failures))
    else:
        parser.error("--workers must be positive")
