from scripts.release.finalize_candidate import release_files
from scripts.release.validate_installation_evidence import sha


def test_release_inventory_excludes_local_environments_and_builds(tmp_path):
    source = tmp_path / "bench/task.py"
    source.parent.mkdir()
    source.write_text("value = 1\n")
    evidence = tmp_path / "docs/installation_evidence/test.log"
    evidence.parent.mkdir(parents=True)
    evidence.write_text("passed\n")
    for name in ("outputs/environment/secret.py", ".git/config", "build/lib/bench/task.py", "bench/__pycache__/task.pyc", "tests/.pytest_cache/state"):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("local state")
    assert release_files(tmp_path) == {
        "bench/task.py": sha(source),
        "docs/installation_evidence/test.log": sha(evidence),
    }
