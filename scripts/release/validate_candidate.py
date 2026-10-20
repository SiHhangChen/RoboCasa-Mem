#!/usr/bin/env python3
"""Verify isolation, all recorded suites, and all 450 resolved task inputs.

Produces a candidate input lock. Physical simulation validation is separate;
this lock does not mark the candidate ready for public release.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROBE = r"""
import dataclasses, importlib.abc, json, sys
from pathlib import Path
root=Path(sys.argv[1]).resolve()
sys.path.insert(0,str(root))
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, *args):
        if '.expert' in fullname or fullname.startswith('bench.recording') or fullname in ('mplib','bench.execution.planning.mplib_planner'):
            raise AssertionError('forbidden import: '+fullname)
sys.meta_path.insert(0,Block())
from bench.evaluation.suite import load_eval_suite
from bench.tasks import TaskCatalog, list_builtin_task_plugins, resolve_builtin_task_plugin
from bench.tasks.contracts import EpisodeRequest
from bench.core.identity import canonical_hash

def normalized(value):
    if dataclasses.is_dataclass(value): return normalized(dataclasses.asdict(value))
    if isinstance(value,Path): return str(value).replace(str(root),'$ROOT')
    if isinstance(value,dict): return {k:normalized(v) for k,v in value.items() if k!='planner_script'}
    if isinstance(value,(list,tuple)): return [normalized(v) for v in value]
    if isinstance(value,(set,frozenset)): return sorted(normalized(v) for v in value)
    if hasattr(value,'value'): return normalized(value.value)
    if isinstance(value,str): return value.replace(str(root),'$ROOT')
    return value

sources=json.loads((root/'configs/benchmark/evaluation_suite_sources.json').read_text())
result={'task_ids':TaskCatalog.ids(),'plugin_ids':[p.task_id for p in list_builtin_task_plugins()],
        'suites':{},'episodes':{}}
for entry in sources['entries']:
    suite=load_eval_suite(root/entry['suite_file'])
    result['suites'][entry['family']+'/'+suite.task_id]={
        'hash':suite.content_hash,'rollout':dataclasses.asdict(suite.rollout),
        'seeds':suite.seeds,'options':suite.task_options,'env_split':suite.env_split}
    api=resolve_builtin_task_plugin(suite.task_id).load_runtime_api()
    if sys.argv[3]=='candidate':
        assert api.build_expert_runtime is None
        assert TaskCatalog.get(suite.task_id).build_expert_plan is None
    for seed in suite.seeds:
        # Suites share task inputs, but retain distinct rollout configurations.
        key=suite.task_id+'/'+str(seed)+'/'+canonical_hash(suite.task_options)
        if key in result['episodes']: continue
        options=dict(suite.task_options)
        request=EpisodeRequest(task_id=suite.task_id,seed=seed,env_split=suite.env_split,
            variant=options.get('variant','memory'),options=options.get('task_options',{}),
            task_config_file=root/options['task_config_file'],
            observation_profile_file=root/options['observation_profile_file'])
        episode=TaskCatalog.resolve(request)
        descriptor=api.build_descriptor(episode,True)
        physical={'task_id':episode.task_id,'version':episode.task_version,
                  'facts':episode.facts,'public':episode.public,'max_horizon':episode.max_horizon,
                  'environment':descriptor.environment,'actions':descriptor.actions,
                  'navigation':descriptor.navigation,'prompt':descriptor.prompt,'sensors':descriptor.sensors,
                  'config':descriptor.config}
        result['episodes'][key]={'physical':normalized(physical),
            'config_hash':episode.config_hash,'resource_hash':episode.resource_hash}
Path(sys.argv[2]).write_text(json.dumps(result,sort_keys=True))
"""


def validate(candidate, output):
    with tempfile.TemporaryDirectory(prefix="membench-inputs-") as temporary:

        def probe(root, kind):
            path = Path(temporary) / f"{kind}.json"
            subprocess.run(
                [sys.executable, "-I", "-c", PROBE, str(root), str(path), kind],
                cwd=temporary,
                check=True,
            )
            return json.loads(path.read_text())

        with ThreadPoolExecutor(max_workers=2) as pool:
            baseline = pool.submit(probe, ROOT, "source")
            release = pool.submit(probe, candidate, "candidate")
            source, isolated = baseline.result(), release.result()
    ids = ["WX-01", "WA-01", "WA-05", "WR-01", "WR-08", "TS02", "TS03", "TM01", "TM02"]
    assert isolated["task_ids"] == isolated["plugin_ids"] == ids
    assert source["suites"] == isolated["suites"], "suite execution settings changed"
    assert source["episodes"].keys() == isolated["episodes"].keys()
    changed = []
    identities = {}
    for key, value in source["episodes"].items():
        other = isolated["episodes"][key]
        if value["physical"] != other["physical"]:
            changed.append(key)
        identities[key] = {
            "source_config_hash": value["config_hash"],
            "source_resource_hash": value["resource_hash"],
            "release_config_hash": other["config_hash"],
            "release_resource_hash": other["resource_hash"],
            "task_version": other["physical"]["version"],
        }
    if changed:
        details = {
            key: {
                field: (
                    source["episodes"][key]["physical"][field],
                    isolated["episodes"][key]["physical"][field],
                )
                for field in source["episodes"][key]["physical"]
                if source["episodes"][key]["physical"][field]
                != isolated["episodes"][key]["physical"][field]
            }
            for key in changed[:1]
        }
        raise AssertionError(f"{len(changed)} task inputs differ: {json.dumps(details)[:4000]}")
    payload = {
        "schema_version": "membench.evaluation_input_lock/v1",
        "status": "candidate_inputs_verified",
        "release_ready": False,
        "task_order": ids,
        "suite_count": len(isolated["suites"]),
        "episode_count": len(identities),
        "suites": isolated["suites"],
        "episode_identities": identities,
        "identity_change": "Removed unused planner_script resource; physical task inputs identical.",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n")
    print(
        json.dumps(
            {
                "suites_verified": len(isolated["suites"]),
                "task_inputs_verified": len(identities),
                "lock": str(output),
            }
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    candidate = args.candidate.resolve()
    validate(candidate, args.output or candidate / "configs/benchmark/evaluation_input_lock.json")
