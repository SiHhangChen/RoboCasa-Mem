#!/usr/bin/env python3
"""Run the baseline WR07 policy against a plug-in robot observation driver."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bench.evaluation.client import RemotePolicyClient
from bench.policy.protocol import CanonicalObservation, validate_observation
from bench.real_robot.contracts import CAMERA_ALIASES, RUNTIME_PROFILE
from scripts.eval.real.replay_remote_dataset import (
    _image_hwc_uint8,
    _raw_image_hwc_uint8,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", required=True)
    parser.add_argument(
        "--driver", required=True, help="Importable module exporting create_robot_io()"
    )
    parser.add_argument("--color", choices=("brown", "white", "pink"), required=True)
    parser.add_argument("--episode-id", required=True)
    parser.add_argument("--max-decisions", type=int, default=100)
    parser.add_argument("--replan-steps", type=int, default=10)
    parser.add_argument(
        "--execute", action="store_true", help="Requires driver contract verification"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--api-key-env", default="MEMBENCH_POLICY_API_KEY")
    parser.add_argument("--timeout-s", type=float, default=120.0)
    return parser.parse_args()


def run(
    *,
    client: RemotePolicyClient,
    robot,
    args: argparse.Namespace,
    on_record: Callable[[dict], None] | None = None,
) -> dict:
    contract = client.contract
    if (
        contract.runtime_profile != RUNTIME_PROFILE
        or contract.policy.get("family") != "baseline_wr07_omega"
    ):
        raise ValueError("Connected server is not the WR07 baseline Omega policy")
    choices = contract.policy.get("prompt_choices")
    if not isinstance(choices, dict) or set(choices) != {"brown", "white", "pink"}:
        raise ValueError("Server did not declare the trained WR07 color prompts")
    if args.replan_steps != contract.policy.get("decision_stride_frames"):
        raise ValueError("replan-steps must match the WR07 training decision stride")
    prompt = choices[args.color]
    records = []
    step = 0
    held = False
    try:
        if args.execute and robot.verify_execution_contract(contract) is not True:
            raise ValueError("Robot driver did not verify the action execution contract")
        client.reset(args.episode_id)
        for decision in range(args.max_decisions):
            state, images = robot.observe()
            if not isinstance(images, dict) or set(images) != set(CAMERA_ALIASES):
                raise ValueError("Robot driver must return front/left/right raw RGB images")
            raw = {alias: _raw_image_hwc_uint8(images[alias]) for alias in CAMERA_ALIASES}
            observation = CanonicalObservation(
                task_id="WR-07",
                state=np.ascontiguousarray(np.asarray(state, dtype=np.float32)),
                images={
                    alias: _image_hwc_uint8(
                        raw[alias],
                        target_size=contract.observation.image_size,
                        preserve_aspect=True,
                    )
                    for alias in CAMERA_ALIASES
                },
                prompt=prompt,
                extra_modalities={"raw_images": raw},
            )
            validate_observation(observation, contract)
            started = time.monotonic()
            response = client.infer(observation, episode_id=args.episode_id, step=step)
            elapsed_ms = (time.monotonic() - started) * 1000.0
            actions = response.actions[: args.replan_steps]
            if args.execute:
                robot.execute_chunk(actions, frequency_hz=contract.control_frequency_hz)
            records.append(
                {
                    "decision": decision,
                    "frame": step,
                    "prompt": prompt,
                    "state": observation.state.tolist(),
                    "actions": actions.tolist(),
                    "request_ms": elapsed_ms,
                    "server_timing": dict(response.timing),
                }
            )
            if on_record is not None:
                on_record(records[-1])
            step += len(actions)
    except BaseException:
        robot.hold()
        held = True
        raise
    finally:
        if args.execute and not held:
            robot.hold()
    return {
        "episode_id": args.episode_id,
        "color": args.color,
        "prompt": prompt,
        "mode": "execute" if args.execute else "shadow",
        "policy_contract": contract.to_hello(),
        "records": records,
    }


def main() -> None:
    args = parse_args()
    if args.max_decisions <= 0:
        raise ValueError("max-decisions must be positive")
    module = importlib.import_module(args.driver)
    robot = module.create_robot_io()
    client = RemotePolicyClient(
        args.endpoint,
        api_key=os.environ.get(args.api_key_env),
        timeout_s=args.timeout_s,
    )
    try:
        client.connect()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        journal = args.output.with_suffix(".jsonl")
        with journal.open("w", encoding="utf-8") as handle:
            handle.write(json.dumps({"policy_contract": client.contract.to_hello()}) + "\n")
            handle.flush()

            def record(row: dict) -> None:
                handle.write(json.dumps(row) + "\n")
                handle.flush()

            result = run(client=client, robot=robot, args=args, on_record=record)
    finally:
        client.close()
        robot.close()
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"output={args.output.resolve()}")


if __name__ == "__main__":
    main()
