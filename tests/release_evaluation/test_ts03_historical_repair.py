from copy import deepcopy
import json
from pathlib import Path
import shutil

import pytest

from scripts.release.repair_ts03_stage_report import repair_result, verify_bundle

BUNDLE = (
    Path(__file__).resolve().parents[2]
    / "docs/historical_result_repairs/ts03_pi05_r40"
)


def original_result():
    return json.loads((BUNDLE / "original_result.json").read_text())


def test_historical_report_recovery_preserves_verdict_and_not_applicable_stages():
    original = original_result()
    untouched = deepcopy(original)
    repaired = repair_result(original)
    assert original == untouched
    assert repaired == json.loads((BUNDLE / "repaired_result.json").read_text())
    stages = repaired["report_stages"]["stages"]
    assert stages["sugar_1"]["first_pass_step"] == 367
    assert stages["success"]["first_pass_step"] == repaired["success_step"] == 595
    for name in ("sugar_2", "sugar_3"):
        assert stages[name] == {
            "applicable": False, "passed": None, "first_pass_step": None
        }


def test_recovery_refuses_a_report_that_contradicts_original_diagnostics():
    original = original_result()
    original[""]["stages"]["sugar_1"]["first_pass_step"] += 1
    with pytest.raises(ValueError, match="contradicts diagnostics"):
        repair_result(original)


def test_recovery_refuses_to_overwrite_an_existing_report():
    original = original_result()
    original["report_stages"] = deepcopy(original[""])
    with pytest.raises(ValueError, match="misplaced report"):
        repair_result(original)


def test_bundle_verifies_and_detects_modified_summary(tmp_path):
    verify_bundle(BUNDLE)
    output = tmp_path / "repair"
    shutil.copytree(BUNDLE, output)
    summary = json.loads((output / "summary.json").read_text())
    assert summary["report_stages"]["reported_episodes"] == 50
    assert summary["publishable"] is True
    assert summary["counts"]["success"] == 14
    summary["counts"]["success"] = 15
    (output / "summary.json").write_text(json.dumps(summary))
    with pytest.raises(ValueError, match="hash mismatch"):
        verify_bundle(output)
