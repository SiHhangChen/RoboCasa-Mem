from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from bench.evaluation.report_profiles import REPORT_STAGE_PROFILES
from bench.evaluation.suite import load_eval_suite
from bench.tasks import resolve_builtin_task_plugin


REPO_ROOT = Path(__file__).resolve().parents[2]
SUITES_DIR = REPO_ROOT / "configs" / "eval" / "suites"
BENCHMARK_TASK_IDS = tuple(profile.task_id for profile in REPORT_STAGE_PROFILES)
EXPECTED_SUITE_NAMES = {
    f"{task_id.replace('-', '').lower()}_v1.yaml"
    for task_id in BENCHMARK_TASK_IDS
}
ALL_SUITE_PATHS = tuple(sorted(SUITES_DIR.glob("*.yaml")))


def test_formal_suite_directory_contains_one_suite_per_benchmark_task() -> None:
    actual_names = {path.name for path in ALL_SUITE_PATHS}

    assert actual_names == EXPECTED_SUITE_NAMES
    assert len(ALL_SUITE_PATHS) == len(BENCHMARK_TASK_IDS) == 18
    assert all("smoke" not in name for name in actual_names)
    assert all("gr00t" not in name for name in actual_names)
    assert all("pi05" not in name for name in actual_names)


@pytest.mark.parametrize("suite_path", ALL_SUITE_PATHS, ids=lambda path: path.name)
def test_canonical_suites_share_the_frozen_benchmark_contract(
    suite_path: Path,
) -> None:
    suite = load_eval_suite(suite_path)
    plugin = resolve_builtin_task_plugin(suite.task_id)
    expected_name = f"{suite.task_id.replace('-', '').lower()}_v1"

    assert suite.task_id in BENCHMARK_TASK_IDS
    assert suite_path.name == f"{expected_name}.yaml"
    assert suite.suite_version == expected_name
    assert suite.task_id == plugin.canonical_id
    assert suite.env_split == "pretrain"
    if suite.task_id == "TS01":
        assert len(suite.seeds) == 36
        assert len(set(suite.seeds)) == 36
    else:
        assert suite.seeds == tuple(range(1050, 1100))
    assert suite.rollout.replan_steps == (20 if suite.task_id == "TS02" else 50)
    assert suite.rollout.action_repeat == 1
    assert suite.rollout.stop_on_success is True
    assert suite.rollout.ignore_done is True
    assert suite.artifacts.record_video is True
    assert suite.artifacts.video_cameras == (
        "agentview_left",
        "agentview_right",
        "eye_in_hand",
    )
    assert suite.artifacts.video_fps == 20
    assert suite.transport.timeout_s == 30.0
    assert suite.transport.connect_attempts == 10
    assert suite.transport.connect_retry_delay_s == 2.0
    assert suite.transport.inference_retries == 0

    task_config_option = suite.task_options.get("task_config_file")
    assert isinstance(task_config_option, str)
    resolved_task_config = (REPO_ROOT / task_config_option).resolve()
    assert resolved_task_config == plugin.default_task_config.resolve()
    assert resolved_task_config.is_file()
    assert suite.task_options.get("observation_profile_file", "").endswith(
        "configs/benchmark/observation_profiles/rgbd_v3.yaml"
    )

    task_config = yaml.safe_load(
        plugin.default_task_config.read_text(encoding="utf-8")
    )
    assert suite.rollout.max_steps == task_config["episode"]["max_horizon"]
