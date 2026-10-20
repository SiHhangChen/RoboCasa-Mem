from __future__ import annotations

from pathlib import Path

import pytest

from bench.evaluation.suite import expand_seeds, load_eval_suite


def _write_suite(tmp_path: Path, content: str) -> Path:
    path = tmp_path / "suite.yaml"
    path.write_text(content.strip() + "\n", encoding="utf-8")
    return path


def test_expand_seed_range_uses_exclusive_stop():
    assert expand_seeds({"start": 1000, "stop": 1004}) == (1000, 1001, 1002, 1003)


def test_load_suite_and_hash(tmp_path: Path):
    path = _write_suite(
        tmp_path,
        """
suite_version: wx01_test
task:
  id: WX-01
  env_split: pretrain
seeds: [1000, 1001]
rollout:
  max_steps: 20
  replan_steps: 4
artifacts:
  record_video: false
transport:
  inference_retries: 0
""",
    )

    suite = load_eval_suite(path)

    assert suite.task_id == "WX-01"
    assert suite.seeds == (1000, 1001)
    assert suite.content_hash.startswith("sha256:")


def test_inference_retries_are_rejected_for_protocol_v1(tmp_path: Path):
    path = _write_suite(
        tmp_path,
        """
suite_version: test
task: {id: WX-01}
seeds: [1]
rollout: {max_steps: 2, replan_steps: 1}
transport: {inference_retries: 1}
""",
    )
    with pytest.raises(ValueError, match="retries"):
        load_eval_suite(path)


@pytest.mark.parametrize(
    "content",
    [
        """
suite_version: test
task: {id: WX-01}
seeds: [1]
rollout: {max_steps: 2, replan_steps: 1}
unexpected: true
""",
        """
suite_version: test
task: {id: WX-01, typo: true}
seeds: [1]
rollout: {max_steps: 2, replan_steps: 1}
""",
        """
suite_version: test
task: {id: WX-01}
seeds: {start: 1, stop: 2, typo: 1}
rollout: {max_steps: 2, replan_steps: 1}
""",
        """
suite_version: test
task: {id: WX-01}
seeds: [1]
rollout: {max_steps: 2, replan_steps: 1, typo: true}
""",
        """
suite_version: test
task: {id: WX-01}
seeds: [1]
rollout: {max_steps: 2, replan_steps: 1}
artifacts: {typo: true}
""",
        """
suite_version: test
task: {id: WX-01}
seeds: [1]
rollout: {max_steps: 2, replan_steps: 1}
transport: {typo: true}
""",
    ],
)
def test_unknown_fields_are_rejected_at_every_schema_level(
    tmp_path: Path,
    content: str,
):
    with pytest.raises(ValueError, match="Unknown field"):
        load_eval_suite(_write_suite(tmp_path, content))


@pytest.mark.parametrize(
    ("section", "message"),
    [
        ('artifacts: {record_video: "false"}', "record_video"),
        ("artifacts: {video_cameras: agentview_right}", "video_cameras"),
        ('rollout: {max_steps: 2, replan_steps: 1, stop_on_success: "false"}', "stop_on_success"),
        ('transport: {connect_attempts: "10"}', "connect_attempts"),
        ('task: {id: WX-01, env_split: 1}', "env_split"),
    ],
)
def test_yaml_values_are_not_implicitly_coerced(
    tmp_path: Path,
    section: str,
    message: str,
):
    task = "task: {id: WX-01}" if not section.startswith("task:") else section
    rollout = (
        "rollout: {max_steps: 2, replan_steps: 1}"
        if not section.startswith("rollout:")
        else section
    )
    optional = "" if section.startswith(("task:", "rollout:")) else section
    content = f"""
suite_version: test
{task}
seeds: [1]
{rollout}
{optional}
"""
    with pytest.raises(ValueError, match=message):
        load_eval_suite(_write_suite(tmp_path, content))


@pytest.mark.parametrize(
    ("transport", "message"),
    [
        ("{timeout_s: .nan}", "finite"),
        ("{timeout_s: .inf}", "finite"),
        ("{connect_retry_delay_s: -.inf}", "finite"),
        ("{connect_retry_delay_s: -0.01}", "non-negative"),
    ],
)
def test_transport_numbers_must_be_finite_and_retry_delay_non_negative(
    tmp_path: Path,
    transport: str,
    message: str,
):
    path = _write_suite(
        tmp_path,
        f"""
suite_version: test
task: {{id: WX-01}}
seeds: [1]
rollout: {{max_steps: 2, replan_steps: 1}}
transport: {transport}
""",
    )
    with pytest.raises(ValueError, match=message):
        load_eval_suite(path)


def test_zero_retry_delay_is_allowed(tmp_path: Path):
    path = _write_suite(
        tmp_path,
        """
suite_version: test
task: {id: WX-01}
seeds: [1]
rollout: {max_steps: 2, replan_steps: 1}
transport: {connect_retry_delay_s: 0}
""",
    )

    assert load_eval_suite(path).transport.connect_retry_delay_s == 0.0


def test_seed_range_step_requires_an_integer():
    with pytest.raises(ValueError, match="step"):
        expand_seeds({"start": 1, "stop": 3, "step": "1"})
