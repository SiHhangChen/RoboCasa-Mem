import hashlib
import json
from pathlib import Path

import pytest

from scripts.release.verify_assets import load_manifest, verify


def manifest_for(tmp_path):
    content = b"pinned mesh"
    relative = "robocasa/robocasa/models/assets/mesh.obj"
    path = tmp_path / relative
    path.parent.mkdir(parents=True)
    path.write_bytes(content)
    digest = hashlib.sha256(content).hexdigest()
    manifest = {
        "schema_version": "membench.simulator_assets/v1",
        "file_count": 1,
        "total_bytes": len(content),
        "archive": {"bytes": len(content), "sha256": digest},
        "files": [{"path": relative, "bytes": len(content), "sha256": digest}],
    }
    return manifest, path


def test_restored_assets_detect_missing_and_same_size_modified_content(tmp_path):
    manifest, path = manifest_for(tmp_path)
    assert verify(manifest, root=tmp_path)["passed"] is True
    path.write_bytes(b"edited mesh")
    result = verify(manifest, root=tmp_path)
    assert result["passed"] is False
    assert result["failures"][0]["reason"] == "content_mismatch"
    path.unlink()
    assert verify(manifest, root=tmp_path)["failures"][0]["reason"] == "missing"


def test_archive_identity_can_be_checked_without_installed_assets(tmp_path):
    manifest, archive = manifest_for(tmp_path)
    result = verify(manifest, archive=archive)
    assert result["passed"] is True
    assert result["checked_files"] == 0
    archive.write_bytes(b"edited mesh")
    assert verify(manifest, archive=archive)["failures"][0]["reason"] == "archive_mismatch"


@pytest.mark.parametrize("unsafe", ["../outside", "/absolute", "robocasa/../outside"])
def test_manifest_rejects_unsafe_paths(tmp_path, unsafe):
    manifest, _ = manifest_for(tmp_path)
    manifest["files"][0]["path"] = unsafe
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="invalid or duplicate"):
        load_manifest(path)


def test_manifest_rejects_duplicate_entries(tmp_path):
    manifest, _ = manifest_for(tmp_path)
    manifest["files"] *= 2
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="duplicate"):
        load_manifest(path)


def test_release_inventory_and_verification_are_consistent():
    root = Path(__file__).resolve().parents[2]
    path = root / "configs/benchmark/simulator_assets.json"
    manifest = load_manifest(path)
    evidence = json.loads((root / "docs/asset_evidence/current_bundle_verification.json").read_text())
    assert evidence["manifest_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert manifest["file_count"] == evidence["checked_files"] == 3035
    assert evidence["passed"] is True and not evidence["failures"]
    assert manifest["scope"]["nine_task_coverage_verified"] is False
    models = {entry["path"] for entry in manifest["files"]}
    base = "robocasa/robocasa/models/assets/objects/objaverse"
    for sponge in (0, 1, 2, 3, 4, 5, 6, 8):
        assert f"{base}/sponge/sponge_{sponge}/model.xml" in models
    for milk in (0, 1):
        assert f"{base}/milk/milk_{milk}/model.xml" in models
