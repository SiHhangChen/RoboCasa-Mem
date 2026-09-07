import hashlib
import io
import json
import tarfile

import pytest

from scripts.analysis.wr08_recording_provenance import (
    validate_recording_provenance, verified_source_archive,
)


def snapshot(directory, files):
    directory.mkdir()
    hashes = {name: hashlib.sha256(content).hexdigest() for name, content in files.items()}
    path = directory / "source_hashes.json"
    path.write_text(json.dumps({"files": hashes}))
    with tarfile.open(directory / "source.tar.gz", "w:gz") as archive:
        for name, content in files.items():
            member = tarfile.TarInfo(name)
            member.size = len(content)
            archive.addfile(member, io.BytesIO(content))
    return path


@pytest.fixture
def source_batches(tmp_path):
    files = {"bench/tasks/wr08/environment.py": b"unchanged dynamics",
             "bench/tasks/wr08/execution/geometry.py": b"original selector",
             "bench/tasks/wr08/execution/pick_controller.py": b"original collision preflight"}
    old = snapshot(tmp_path / "r13", files)
    new_files = {**files, "bench/tasks/wr08/execution/geometry.py": b"lower fallback",
                 "bench/tasks/wr08/execution/pick_controller.py": b"safety margin",
                 "scripts/analysis/wr08_recording_provenance.py": b"audit"}
    current = snapshot(tmp_path / "r14", new_files)
    native = {0: {"episode_dir": str(tmp_path / "original" / "seed0")},
              26: {"episode_dir": str(tmp_path / "repair" / "seed26")}}
    manifest = tmp_path / "sources.json"
    batches = [{"dataset_root": str(tmp_path / "original"), "source_snapshot": str(old)},
               {"dataset_root": str(tmp_path / "repair"), "source_snapshot": str(current)}]
    manifest.write_text(json.dumps({"batches": batches}))
    return native, manifest, current, old, batches, files, new_files


def test_original_and_repair_keep_distinct_verified_archives(source_batches):
    native, manifest, current, old, *_ = source_batches
    result = validate_recording_provenance(native, manifest, current)
    assert result["verified"] and not result["errors"]
    assert result["batches"][0]["seeds"] == [0]
    assert result["batches"][1]["seeds"] == [26]
    assert result["batches"][0]["source_snapshot"] == str(old)
    assert result["batches"][1]["differences_from_final"] == []
    assert len(result["batches"][0]["differences_from_final"]) == 2
    assert len(result["snapshots"]) == 2 and all(s["archive_verified"] for s in result["snapshots"])


@pytest.mark.parametrize("change", ["environment", "new_runtime", "dropped_source"])
def test_non_expert_changes_prevent_mixed_batch_qualification(source_batches, tmp_path, change):
    native, manifest, _, _, _, _, new_files = source_batches
    if change == "environment":
        new_files["bench/tasks/wr08/environment.py"] = b"changed physics"
    elif change == "new_runtime":
        new_files["bench/tasks/wr08/teleport.py"] = b"new runtime"
    else:
        del new_files["bench/tasks/wr08/environment.py"]
    current = snapshot(tmp_path / "invalid_final", new_files)
    result = validate_recording_provenance(native, manifest, current)
    assert not result["verified"] and result["errors"]


@pytest.mark.parametrize("case", ["missing", "overlapping"])
def test_each_recording_needs_one_unambiguous_source_batch(source_batches, tmp_path, case):
    native, manifest, current, old, batches, *_ = source_batches
    if case == "missing":
        batches.pop()
    else:
        batches.append({"dataset_root": str(tmp_path), "source_snapshot": str(old)})
    manifest.write_text(json.dumps({"batches": batches}))
    result = validate_recording_provenance(native, manifest, current)
    assert not result["verified"]
    assert any("exactly one source batch" in error for error in result["errors"])


def test_source_archive_hash_corruption_is_rejected(source_batches):
    _, _, _, old, *_ = source_batches
    data = json.loads(old.read_text())
    data["files"]["bench/tasks/wr08/environment.py"] = "0" * 64
    old.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="archive hash mismatch"):
        verified_source_archive(old)


def test_snapshot_member_set_must_match_archive(source_batches):
    _, _, _, old, *_ = source_batches
    data = json.loads(old.read_text())
    del data["files"]["bench/tasks/wr08/environment.py"]
    old.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="file set differs"):
        verified_source_archive(old)


def test_bad_archive_is_reported_without_qualifying_data(source_batches):
    native, manifest, current, old, *_ = source_batches
    old.with_name("source.tar.gz").write_bytes(b"invalid archive")
    result = validate_recording_provenance(native, manifest, current)
    assert not result["verified"] and result["errors"]
