#!/usr/bin/env python3
"""Check a pinned simulator archive and/or restored files without a simulator."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re

DEFAULT_MANIFEST = Path(__file__).resolve().parents[2] / "configs/benchmark/simulator_assets.json"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest(path: Path) -> dict:
    manifest = json.loads(path.read_text())
    if manifest["schema_version"] != "membench.simulator_assets/v1":
        raise ValueError("unsupported simulator asset manifest")
    seen = set()
    for entry in manifest["files"]:
        name = entry["path"]
        relative = PurePosixPath(name)
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or "\\" in name
            or relative.as_posix() != name
            or not relative.parts
            or relative.parts[0] not in {"robocasa", "robosuite"}
            or name in seen
        ):
            raise ValueError(f"invalid or duplicate asset path: {name}")
        if not re.fullmatch(r"[0-9a-f]{64}", entry["sha256"]):
            raise ValueError(f"invalid asset digest: {name}")
        if type(entry["bytes"]) is not int or entry["bytes"] < 0:
            raise ValueError(f"invalid asset size: {name}")
        seen.add(name)
    if manifest["file_count"] != len(seen):
        raise ValueError("asset file count mismatch")
    if manifest["total_bytes"] != sum(entry["bytes"] for entry in manifest["files"]):
        raise ValueError("asset total size mismatch")
    return manifest


def verify(manifest: dict, *, root: Path | None = None, archive: Path | None = None) -> dict:
    if root is None and archive is None:
        raise ValueError("provide an archive or restored asset root")
    failures = []
    checked = 0
    if archive is not None:
        expected = manifest["archive"]
        if not archive.is_file():
            failures.append({"path": str(archive), "reason": "missing_archive"})
        elif archive.stat().st_size != expected["bytes"] or sha(archive) != expected["sha256"]:
            failures.append({"path": str(archive), "reason": "archive_mismatch"})
    if root is not None:
        for entry in manifest["files"]:
            path = root / entry["path"]
            checked += 1
            if not path.is_file():
                failures.append({"path": entry["path"], "reason": "missing"})
            elif path.stat().st_size != entry["bytes"] or sha(path) != entry["sha256"]:
                failures.append({"path": entry["path"], "reason": "content_mismatch"})
    return {
        "schema_version": "membench.simulator_asset_verification/v1",
        "passed": not failures,
        "checked_files": checked,
        "root": str(root.resolve()) if root is not None else None,
        "archive": str(archive.resolve()) if archive is not None else None,
        "failures": failures,
        "note": "File identity check only; task coverage and simulator acceptance are separate.",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--root", type=Path, help="parent of the robocasa/ and robosuite/ checkouts")
    parser.add_argument("--archive", type=Path, help="verify archive bytes before extracting")
    parser.add_argument("--output", type=Path, help="save verification evidence")
    args = parser.parse_args()
    if args.root is None and args.archive is None:
        parser.error("provide --root and/or --archive")
    result = verify(load_manifest(args.manifest), root=args.root, archive=args.archive)
    result["manifest_sha256"] = sha(args.manifest)
    payload = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload)
    print(payload, end="")
    raise SystemExit(0 if result["passed"] else 1)
