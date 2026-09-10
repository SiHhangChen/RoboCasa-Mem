#!/usr/bin/env python3
"""Freeze WR-08 validation sources while retaining parent dependency coverage."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile

ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-snapshot", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--note", required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    parent = json.loads(args.parent_snapshot.read_text())
    paths = set(parent["files"])
    paths.update(str(path.relative_to(ROOT)) for path in (ROOT / "bench/tasks/wr08").rglob("*")
                 if path.is_file() and "__pycache__" not in path.parts)
    paths.update(str(path.relative_to(ROOT)) for path in (ROOT / "scripts/analysis").glob("*wr08*.py"))
    paths.update({
        "robocasa/robocasa/models/fixtures/cabinets.py",
        "robocasa/robocasa/models/fixtures/fixture.py",
        "robocasa/robocasa/models/assets/objects/objaverse/milk/milk_0/model.xml",
        "robocasa/robocasa/models/assets/objects/objaverse/milk/milk_1/model.xml",
    })
    for name in paths:
        if not (ROOT / name).is_file():
            raise FileNotFoundError(name)
    hashes = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in sorted(paths)}
    snapshot = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "worktree_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT)),
        "note": args.note, "parent_snapshot": str(args.parent_snapshot), "files": hashes,
    }
    args.output_dir.mkdir(parents=True)
    with tarfile.open(args.output_dir / "source.tar.gz", "w:gz") as archive:
        for name in sorted(paths):
            archive.add(ROOT / name, arcname=name)
    (args.output_dir / "source_hashes.json").write_text(json.dumps(snapshot, indent=2) + "\n")
    print(json.dumps({"files": len(hashes), "output_dir": str(args.output_dir)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
