import json
import shutil
from pathlib import Path

import pytest

from scripts.release.validate_installation_evidence import runtime_files, sha, validate


ROOT = Path(__file__).resolve().parents[2]
CANDIDATE = ROOT if (ROOT / "docs/installation_evidence").is_dir() and (ROOT / "configs/benchmark/evaluation_input_lock.json").is_file() else ROOT / "release/evaluation"


def test_recorded_installation_evidence_matches_candidate():
    assert validate(CANDIDATE)["scope"]["case_count"] == 450


@pytest.mark.parametrize("damage", ["missing_case", "blocked_read"])
def test_incomplete_or_nonisolated_run_cannot_be_finalized(tmp_path, damage):
    for folder in ("bench", "configs", "scripts"):
        (tmp_path / folder).symlink_to(CANDIDATE / folder, target_is_directory=True)
    evidence = tmp_path / "docs/installation_evidence"
    shutil.copytree(CANDIDATE / "docs/installation_evidence", evidence)
    coverage_path = evidence / "full_input_coverage.json"
    coverage = json.loads(coverage_path.read_text())
    if damage == "missing_case":
        coverage["checks"].pop()
    else:
        coverage["checks"][0]["blocked_development_accesses"] = ["/original/asset.xml"]
    coverage_path.write_text(json.dumps(coverage))
    manifest_path = evidence / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["files"][coverage_path.name] = sha(coverage_path)
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="450 official inputs|incomplete installed runtime"):
        validate(tmp_path)


def test_runtime_snapshot_detects_code_changes_and_ignores_documentation(tmp_path):
    module = tmp_path / "bench/runtime.py"
    module.parent.mkdir()
    module.write_text("VALUE = 1\n")
    before = runtime_files(tmp_path)
    (tmp_path / "README.md").write_text("Updated installation instructions")
    assert runtime_files(tmp_path) == before
    module.write_text("VALUE = 2\n")
    assert runtime_files(tmp_path) != before
