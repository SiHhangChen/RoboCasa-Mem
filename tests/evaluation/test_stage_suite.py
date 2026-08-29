from pathlib import Path

from bench.evaluation.stage_suite import load_stage_eval_suite


ROOT = Path(__file__).resolve().parents[2]


def test_wa05_gr00t_stage_suite_loads_base_contract() -> None:
    suite = load_stage_eval_suite(
        ROOT / "configs/eval/stages/wa05_gr00t_r40_v1.yaml"
    )
    assert suite.task_id == "WA-05"
    assert suite.base_suite.rollout.replan_steps == 40
    assert suite.stage_ids == ("cup_filled", "cup_staged", "success")
    assert suite.targets[0].source == "main_rollout"
    assert suite.targets[1].source == "expert_prefix"


def test_wa05_formal_stage_suite_freezes_prefix_derived_budgets() -> None:
    suite = load_stage_eval_suite(
        ROOT / "configs/eval/stages/wa05_gr00t_r40_formal_v1.yaml"
    )
    assert suite.stage_suite_version == "wa05_gr00t_stage_r40_formal_v1"
    assert suite.expert_prefix_root == (
        ROOT / "outputs/stage_prefixes/wa05_formal_v1"
    ).resolve()
    assert suite.budgets == {
        "cup_filled": 1146,
        "cup_staged": 804,
        "success": 366,
    }
