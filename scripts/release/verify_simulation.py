#!/usr/bin/env python3
"""Compare source and exported runtime observations and equal-action trajectories.

This requires the pinned simulator environment. Each runtime loads in its own
process so candidate imports cannot fall back to development task code.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
IDS = ("WX-01", "WA-01", "WA-05", "WR-01", "WR-08", "TS02", "TS03", "TM01", "TM02")
PROBE = r"""
import dataclasses, hashlib, importlib.abc, json, sys
from pathlib import Path
import numpy as np
root=Path(sys.argv[1]).resolve()
sys.path.insert(0,str(root))
if sys.argv[4]=='candidate':
    class Block(importlib.abc.MetaPathFinder):
        def find_spec(self,fullname,*args):
            if '.expert' in fullname or fullname.startswith('bench.recording') or fullname in ('mplib','bench.execution.planning.mplib_planner'):
                raise AssertionError('forbidden import: '+fullname)
    sys.meta_path.insert(0,Block())
from bench.runtime import RoboCasaRuntimeFactory, RobotStateExtractor
from bench.tasks import TaskCatalog, resolve_builtin_task_plugin
from bench.tasks.contracts import EpisodeRequest
from bench.evaluation.report_stages import validate_report_stages
from bench.policy.protocol import PolicyContract
from bench.core.identity import canonical_hash

def normalized(value):
    if dataclasses.is_dataclass(value): return normalized(dataclasses.asdict(value))
    if isinstance(value,np.ndarray): return value.tolist()
    if isinstance(value,dict): return {k:normalized(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)): return [normalized(v) for v in value]
    if isinstance(value,(np.integer,np.floating)): return value.item()
    if isinstance(value,str): return value.replace(str(root),'$ROOT')
    return value

task=sys.argv[2]
seed=int(sys.argv[3])
plugin=resolve_builtin_task_plugin(task)
api=plugin.load_runtime_api()
episode=TaskCatalog.resolve(EpisodeRequest(task_id=task,seed=seed))
runtime=RoboCasaRuntimeFactory(api).create(episode,execute=True)
extractor=RobotStateExtractor()
result={'task_id':task,'seed':seed,'prompt':runtime.initialized.descriptor.prompt.policy_prompt,'samples':[]}
try:
    right=runtime.env.robots[0].composite_controller.part_controllers['right']
    arm=np.asarray(runtime.env.sim.data.qpos[right.qpos_index]).copy()
    action=np.zeros(13)
    action[:7]=arm
    action[7]=1
    for step in range(21):
        diagnostics=normalized(runtime.diagnostics())
        stages=api.report_stages(runtime.initialized)
        validate_report_stages(stages,expected_task_id=task,expected_profile_id=api.report_stage_profile_id)
        result['samples'].append({
            'step':step,'qpos':runtime.env.sim.data.qpos.tolist(),
            'qvel':runtime.env.sim.data.qvel.tolist(),
            'state37':extractor.extract(runtime.env,runtime.observation).tolist(),
            'observation_sha256':{k:hashlib.sha256(np.ascontiguousarray(v).tobytes()).hexdigest()
                                  for k,v in runtime.observation.items() if isinstance(v,np.ndarray)},
            'diagnostics':diagnostics,'report_stages':normalized(stages)})
        if step<20:
            action[0]=arm[0]+0.01*np.sin(step/3)
            action[8]=0.005*np.sin(step/4)
            runtime.step(action)
    Path(sys.argv[5]).write_text(json.dumps(result,sort_keys=True))
finally:
    runtime.close()
"""


def verify(candidate, output, tasks, seed):
    import numpy as np

    directory = output.parent / "simulation_evidence"
    directory.mkdir(parents=True, exist_ok=True)

    def probe(root, kind, task):
        path = directory / f"{task}-{seed}-{kind}.json"
        log = path.with_suffix(".log")
        with log.open("w") as stream:
            result = subprocess.run(
                [sys.executable, "-c", PROBE, str(root), task, str(seed), kind, str(path)],
                cwd=directory,
                stdout=stream,
                stderr=subprocess.STDOUT,
                check=False,
            )
        if result.returncode:
            raise RuntimeError(f"{kind} {task} failed; see {log}")
        return json.loads(path.read_text())

    verified = []
    for task in tasks:
        with ThreadPoolExecutor(max_workers=2) as pool:
            source_job = pool.submit(probe, ROOT, "source", task)
            release_job = pool.submit(probe, candidate, "candidate", task)
            source, release = source_job.result(), release_job.result()
        assert source["prompt"] == release["prompt"], task
        for left, right in zip(source["samples"], release["samples"], strict=True):
            for field in ("qpos", "qvel", "state37"):
                np.testing.assert_allclose(
                    left[field],
                    right[field],
                    rtol=0,
                    atol=1e-9,
                    err_msg=f"{task}/{left['step']}/{field}",
                )
            for field in ("observation_sha256", "diagnostics", "report_stages"):
                assert left[field] == right[field], f"{task}/{left['step']}/{field} changed"
        verified.append(
            {
                "task_id": task,
                "seed": seed,
                "initial_observations_equal": True,
                "state37_equal": True,
                "action_steps": 20,
                "trajectory_observations_equal": True,
                "diagnostics_and_report_stages_equal": True,
                "state_atol": 1e-9,
            }
        )
        output.write_text(
            json.dumps(
                {"schema_version": "membench.simulation_equivalence/v1", "checks": verified},
                indent=2,
            )
            + "\n"
        )
        print(json.dumps(verified[-1]), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate", type=Path)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "outputs/release/simulation_equivalence.json"
    )
    parser.add_argument("--tasks", nargs="+", default=list(IDS), choices=IDS)
    parser.add_argument("--seed", type=int, default=1050)
    args = parser.parse_args()
    verify(args.candidate.resolve(), args.output.resolve(), args.tasks, args.seed)
