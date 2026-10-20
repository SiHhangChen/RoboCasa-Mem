from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TASKS = ("wx01", "wa01", "wa05", "wr01", "wr08", "ts02", "ts03", "tm01", "tm02")


def test_evaluation_runtimes_never_import_optional_experts_or_planner():
    code = """
import sys, importlib, importlib.abc
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, *args):
        if '.expert' in fullname or fullname.startswith('bench.recording') or fullname in ('mplib','bench.execution.planning.mplib_planner'):
            raise AssertionError('forbidden import: ' + fullname)
sys.meta_path.insert(0,Block())
from bench.tasks import TaskCatalog
from bench.tasks.contracts import EpisodeRequest
for task in sys.argv[1:]:
    api=importlib.import_module('bench.tasks.'+task+'.runtime').TASK_RUNTIME_API
    episode=TaskCatalog.resolve(EpisodeRequest(task_id=api.build_descriptor.__self__.task_id,seed=1050))
    descriptor=api.build_descriptor(episode,True)
    assert descriptor.control_frequency_hz == 20
    assert descriptor.sensors.image_size == (224,224)
"""
    subprocess.run([sys.executable, "-c", code, *TASKS], cwd=ROOT, check=True)


def test_lazy_expert_callback_loads_only_when_requested(monkeypatch):
    import importlib
    from types import SimpleNamespace

    from bench.runtime.task_local import lazy_expert_runtime

    imports = []
    monkeypatch.setattr(
        importlib,
        "import_module",
        lambda name: imports.append(name) or SimpleNamespace(build=lambda x: {"value": x}),
    )
    builder = lazy_expert_runtime("optional.expert:build")
    assert imports == []
    assert builder(3) == {"value": 3}
    assert imports == ["optional.expert"]
