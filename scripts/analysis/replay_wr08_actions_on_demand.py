#!/usr/bin/env python3
"""Run the production WR-08 action evaluator with audited camera scheduling.

Only replay, with no video output, may elide unconsumed camera samples. All
observable timers still advance normally; every policy input and the final
frame is rendered at its original sample time. Physics and consumed image
digests support comparison against --render-every-step. No simulator state is
restored or written. This script never modifies the recording implementation.
"""

from __future__ import annotations

import argparse
import functools
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from bench.evaluation import replay as replay_module
from scripts.eval import replay_recorded_actions


class CameraSchedule:
    def __init__(self, env, initial_observation, *, stride, horizon, every_step):
        self.env = env
        self.stride = int(stride)
        self.horizon = int(horizon)
        if self.stride <= 0 or self.horizon <= 0:
            raise ValueError("camera stride and horizon must be positive")
        self.every_step = bool(every_step)
        self.control_step = 0
        self.render_now = True
        self.render_calls = 0
        self.elided_calls = 0
        self.original_step = env.step
        self.sensors = []
        self.physics_digest = hashlib.sha256()
        self.input_digest = hashlib.sha256()
        self.metadata = {
            "schedule": "every_step" if every_step else "policy_observation_and_final",
            "stride": self.stride, "physics_steps": 0, "observation_calls": 0,
            "render_calls": 0, "elided_calls": 0,
        }
        for camera in env.camera_names:
            name = f"{camera}_image"
            if name not in env._observables or name not in initial_observation:
                raise ValueError(f"required camera observable missing: {name}")
        for camera in env.camera_names:
            name = f"{camera}_image"
            observable = env._observables[name]
            original = observable._sensor
            self.sensors.append((observable, original))
            observable.set_sensor(self.wrap_sensor(original, camera, initial_observation))
        env.step = self.step

    def wrap_sensor(self, original, camera, initial):
        rgb = np.asarray(initial[f"{camera}_image"])
        depth_key = f"{camera}_depth"
        depth = initial.get(depth_key)

        @functools.wraps(original)
        def sensor(cache):
            nonlocal rgb, depth
            if self.render_now:
                rgb = original(cache)
                depth = cache.get(depth_key)
                self.render_calls += 1
            else:
                if depth is not None:
                    cache[depth_key] = depth
                self.elided_calls += 1
            return rgb

        return sensor

    def step(self, action):
        self.control_step += 1
        self.render_now = (self.every_step or self.control_step % self.stride == 0
                           or self.control_step == self.horizon)
        result = self.original_step(action)
        data = self.env.sim.data
        for value in (action, data.qpos, data.qvel, data.act, data.ctrl, [data.time]):
            self.physics_digest.update(np.asarray(value, dtype=np.float64).tobytes())
        self.metadata.update(
            physics_steps=self.control_step,
            physics_sha256=self.physics_digest.hexdigest(),
            render_calls=self.render_calls, elided_calls=self.elided_calls,
        )
        return result

    def record_input(self, observation):
        if not self.render_now:
            raise RuntimeError("policy requested an unscheduled camera observation")
        self.input_digest.update(np.asarray(observation.state).tobytes())
        for name, image in sorted(observation.images.items()):
            self.input_digest.update(name.encode())
            self.input_digest.update(np.asarray(image).tobytes())
        self.metadata["observation_calls"] += 1
        self.metadata["consumed_observation_sha256"] = self.input_digest.hexdigest()

    def close(self):
        self.env.step = self.original_step
        for observable, sensor in self.sensors:
            observable._sensor = sensor


def main():
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument("--render-every-step", action="store_true")
    args, forwarded = parser.parse_known_args()
    cli_args = replay_recorded_actions.parse_args(forwarded)
    if replay_module.canonical_task_id(cli_args.task_id) != "WR-08":
        raise ValueError("camera scheduling has only been scoped to WR-08")
    if cli_args.record_video:
        raise ValueError("on-demand replay cannot record per-step video")
    original_adapter = replay_module.ReplayEvaluationAdapter

    class AuditedAdapter(original_adapter):
        def create_episode(self, *, seed, policy_contract, suite):
            runtime = super().create_episode(seed=seed, policy_contract=policy_contract, suite=suite)
            schedule = CameraSchedule(
                runtime.env, runtime.observation, stride=suite.rollout.replan_steps,
                horizon=len(self.replay.actions), every_step=args.render_every_step,
            )
            runtime.metadata["camera_schedule"] = schedule
            runtime.result_metadata["replay_camera_schedule"] = schedule.metadata
            return runtime

        def build_observation(self, runtime):
            observation = super().build_observation(runtime)
            runtime.metadata["camera_schedule"].record_input(observation)
            return observation

        def close(self, runtime):
            schedule = runtime.metadata.get("camera_schedule")
            if schedule is not None:
                schedule.close()
            super().close(runtime)

    replay_module.ReplayEvaluationAdapter = AuditedAdapter
    try:
        code = replay_recorded_actions.main(forwarded)
    finally:
        replay_module.ReplayEvaluationAdapter = original_adapter
    manifest_path = cli_args.output_run.resolve() / "run_manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text())
        manifest["replay"]["camera_schedule"] = "every_step" if args.render_every_step else "policy_observation_and_final"
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
