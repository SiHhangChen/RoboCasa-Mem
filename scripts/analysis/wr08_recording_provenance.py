"""Audit source archives for original WR-08 recordings and expert-only repairs.

Old recordings keep their original source attribution. Only the two task-local
pick implementation files (and their README) may differ from the final source
snapshot; environment, task facts, assets, controllers and scoring must match.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
import tarfile


ROOT = Path(__file__).resolve().parents[2]
ALLOWED_HISTORICAL_DIFFERENCES = frozenset({
    "bench/tasks/wr08/execution/geometry.py",
    "bench/tasks/wr08/execution/pick_controller.py",
    "bench/tasks/wr08/README.md",
})


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return (ROOT / path).resolve() if not path.is_absolute() else path.resolve()


def verified_source_archive(snapshot_path: Path) -> dict:
    snapshot_path = snapshot_path.resolve()
    snapshot = json.loads(snapshot_path.read_text())
    files = snapshot["files"]
    if not files:
        raise ValueError("empty source snapshot")
    for name in files:
        path = PurePosixPath(name)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("source snapshot contains an unsafe archive name")
    archive_path = snapshot_path.with_name("source.tar.gz")
    with tarfile.open(archive_path, "r:gz") as archive:
        members = archive.getmembers()
        names = [member.name for member in members]
        if len(set(names)) != len(names) or set(names) != set(files):
            raise ValueError("source archive file set differs from snapshot")
        for member in members:
            if not member.isfile():
                raise ValueError(f"source archive member is not a regular file: {member.name}")
            with archive.extractfile(member) as source:
                digest = hashlib.sha256(source.read()).hexdigest()
            if digest != files[member.name]:
                raise ValueError(f"source archive hash mismatch: {member.name}")
    return {
        "source_snapshot": str(snapshot_path), "source_archive": str(archive_path),
        "snapshot_sha256": hashlib.sha256(snapshot_path.read_bytes()).hexdigest(),
        "archive_sha256": hashlib.sha256(archive_path.read_bytes()).hexdigest(),
        "files": files, "archive_verified": True,
    }


def validate_recording_provenance(native: dict[int, dict], manifest_path: Path,
                                  current_snapshot_path: Path) -> dict:
    report = {"manifest": str(manifest_path.resolve()), "verified": False,
              "errors": [], "batches": [], "snapshots": []}
    try:
        current = verified_source_archive(current_snapshot_path)
        manifest = json.loads(manifest_path.read_text())
        batches = manifest["batches"]
        if not batches:
            raise ValueError("no recording source batches")
        cached = {current["source_snapshot"]: current}
        roots = []
        for batch in batches:
            root = resolve_path(batch["dataset_root"])
            snapshot_path = resolve_path(batch["source_snapshot"])
            key = str(snapshot_path)
            if key not in cached:
                cached[key] = verified_source_archive(snapshot_path)
            source = cached[key]
            missing = sorted(set(source["files"]) - set(current["files"]))
            if missing:
                report["errors"].append(f"{snapshot_path}: historical source coverage dropped: {missing}")
            differences = sorted(name for name, digest in source["files"].items()
                                 if current["files"].get(name) != digest)
            unexpected = sorted(set(differences) - ALLOWED_HISTORICAL_DIFFERENCES)
            if unexpected:
                report["errors"].append(f"{snapshot_path}: non-expert source changed: {unexpected}")
            added = sorted(set(current["files"]) - set(source["files"]))
            runtime_added = [name for name in added if not name.startswith(("scripts/analysis/", "tests/"))]
            if runtime_added:
                report["errors"].append(f"{snapshot_path}: runtime source coverage added: {runtime_added}")
            roots.append(root)
            report["batches"].append({
                "dataset_root": str(root), "source_snapshot": key, "seeds": [],
                "differences_from_final": differences, "added_analysis_or_test_files": added,
            })
        for seed, row in sorted(native.items()):
            episode = Path(row["episode_dir"]).resolve()
            matching = [index for index, root in enumerate(roots) if episode.is_relative_to(root)]
            if len(matching) != 1:
                report["errors"].append(f"seed {seed}: expected exactly one source batch, found {len(matching)}")
                continue
            report["batches"][matching[0]]["seeds"].append(seed)
        report["snapshots"] = [{key: value for key, value in source.items() if key != "files"}
                               for source in cached.values()]
    except (OSError, ValueError, KeyError, TypeError, tarfile.TarError) as exc:
        report["errors"].append(f"{type(exc).__name__}: {exc}")
    report["verified"] = bool(native) and not report["errors"]
    return report
