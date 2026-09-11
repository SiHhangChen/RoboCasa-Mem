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


def test_wa05_pi05_formal_stage_suite_uses_shared_seed_set() -> None:
    suite = load_stage_eval_suite(
        ROOT / "configs/eval/stages/wa05_pi05_r20_formal_v1.yaml"
    )
    assert suite.stage_suite_version == "wa05_pi05_stage_r20_formal_v1"
    assert suite.base_suite.seeds == tuple(range(1050, 1100))
    assert suite.base_suite.rollout.replan_steps == 20
    assert suite.budgets == {
        "cup_filled": 1146,
        "cup_staged": 804,
        "success": 366,
    }
    assert suite.expert_prefix_root == (
        ROOT / "outputs/stage_prefixes/wa05_formal_v1"
    ).resolve()


def test_wa01_pi05_pilot_stage_suite_declares_public_funnel() -> None:
    suite = load_stage_eval_suite(
        ROOT / "configs/eval/stages/wa01_pi05_r50_pilot_v1.yaml"
    )
    assert suite.task_id == "WA-01"
    assert suite.base_suite.seeds == tuple(range(1050, 1100))
    assert suite.base_suite.rollout.replan_steps == 50
    assert suite.stage_ids == (
        "sugar_added",
        "one_cup_ready",
        "two_cups_ready",
        "select",
        "success",
    )


def test_wa01_gr00t_pilot_stage_suite_uses_r20_and_shared_seeds() -> None:
    suite = load_stage_eval_suite(
        ROOT / "configs/eval/stages/wa01_gr00t_r20_pilot_v1.yaml"
    )
    assert suite.task_id == "WA-01"
    assert suite.base_suite.seeds == tuple(range(1050, 1100))
    assert suite.base_suite.rollout.replan_steps == 20
    assert suite.stage_ids == (
        "sugar_added",
        "one_cup_ready",
        "two_cups_ready",
        "select",
        "success",
    )


def test_ts02_gr00t_pilot_stage_suite_uses_r20_and_public_stages() -> None:
    suite = load_stage_eval_suite(
        ROOT / "configs/eval/stages/ts02_gr00t_r20_pilot_v1.yaml"
    )
    assert suite.task_id == "TS02"
    assert suite.base_suite.seeds == tuple(range(1050, 1100))
    assert suite.base_suite.rollout.replan_steps == 20
    assert suite.stage_ids == (
        "one_drawer_checked",
        "target_found",
        "search_stopped_after_target",
        "button_pressed",
        "correct_press_count",
    )


def test_ts02_gr00t_formal_stage_suite_freezes_prefix_budgets() -> None:
    suite = load_stage_eval_suite(
        ROOT / "configs/eval/stages/ts02_gr00t_r20_formal_v1.yaml"
    )
    assert suite.stage_suite_version == "ts02_gr00t_stage_r20_formal_v1"
    assert suite.budgets == {
        "one_drawer_checked": 978,
        "target_found": 978,
        "search_stopped_after_target": 978,
        "button_pressed": 326,
        "correct_press_count": 600,
    }


def test_ts02_requested_gr00t_r40_and_pi05_r50_suites_share_budgets() -> None:
    gr00t = load_stage_eval_suite(
        ROOT / "configs/eval/stages/ts02_gr00t_r40_formal_v1.yaml"
    )
    pi05 = load_stage_eval_suite(
        ROOT / "configs/eval/stages/ts02_pi05_r50_formal_v1.yaml"
    )

    assert gr00t.base_suite.rollout.replan_steps == 40
    assert pi05.base_suite.rollout.replan_steps == 50
    assert gr00t.budgets == pi05.budgets
    assert gr00t.expert_prefix_root == pi05.expert_prefix_root


def test_tm02_b1_diagnostic_stage_suite_loads() -> None:
    suite = load_stage_eval_suite(
        ROOT / "configs/eval/stages/tm02_gr00t_r40_b1_diagnostic_pilot_v1.yaml"
    )

    assert suite.task_id == "TM02"
    assert suite.stage_ids == (
        "faucet_on",
        "pot_filled",
        "cabinet_opened",
        "can_removed_from_cabinet",
        "can_released_in_place_region",
    )
