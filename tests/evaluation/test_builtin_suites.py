from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from bench.evaluation.suite import load_eval_suite
from bench.tasks import list_builtin_task_plugins, resolve_builtin_task_plugin


REPO_ROOT = Path(__file__).resolve().parents[2]
SUITES_DIR = REPO_ROOT / "configs" / "eval" / "suites"
ALL_SUITE_PATHS = tuple(sorted(SUITES_DIR.glob("*.yaml")))
SMOKE_SUITE_PATHS = tuple(sorted(SUITES_DIR.glob("*_smoke_v1.yaml")))
RECORDING_SEEDS = frozenset(range(200))


@pytest.mark.parametrize("suite_path", ALL_SUITE_PATHS, ids=lambda path: path.name)
def test_all_builtin_suites_load_and_reference_the_registered_task_config(
    suite_path: Path,
):
    suite = load_eval_suite(suite_path)
    plugin = resolve_builtin_task_plugin(suite.task_id)

    assert suite.task_id == plugin.canonical_id
    task_config_option = suite.task_options.get("task_config_file")
    assert isinstance(task_config_option, str)
    assert task_config_option

    task_config_path = Path(task_config_option)
    assert not task_config_path.is_absolute()
    resolved_task_config = (REPO_ROOT / task_config_path).resolve()
    assert resolved_task_config.is_file()
    assert resolved_task_config == plugin.default_task_config.resolve()


def test_smoke_suites_cover_every_builtin_task_with_the_frozen_contract():
    plugins = list_builtin_task_plugins()
    suites_by_task = {}

    for suite_path in SMOKE_SUITE_PATHS:
        suite = load_eval_suite(suite_path)
        assert suite.task_id not in suites_by_task
        suites_by_task[suite.task_id] = (suite_path, suite)

    assert set(suites_by_task) == {plugin.canonical_id for plugin in plugins}

    wx_smoke = load_eval_suite(SUITES_DIR / "wx01_smoke_v1.yaml")
    for plugin in plugins:
        suite_path, suite = suites_by_task[plugin.canonical_id]
        expected_name = f"{plugin.canonical_id.replace('-', '').lower()}_smoke_v1"
        assert suite_path.name == f"{expected_name}.yaml"
        assert suite.suite_version == expected_name
        assert suite.env_split == "pretrain"
        assert suite.seeds == (1000, 1001, 1002)
        assert RECORDING_SEEDS.isdisjoint(suite.seeds)
        assert suite.rollout.replan_steps == 50
        assert suite.rollout.action_repeat == 1
        assert suite.rollout.stop_on_success is (suite.task_id not in {"WR-04", "WR-06"})
        assert suite.rollout.ignore_done is True
        assert suite.artifacts == wx_smoke.artifacts
        assert suite.transport == wx_smoke.transport
        assert suite.task_options.get("observation_profile_file", "").endswith(
            "configs/benchmark/observation_profiles/rgbd_v3.yaml"
        )

        task_config = yaml.safe_load(plugin.default_task_config.read_text(encoding="utf-8"))
        assert suite.rollout.max_steps == task_config["episode"]["max_horizon"]
