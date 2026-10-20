#!/usr/bin/env python
"""Rebuild and verify the canonical summary for one evaluation run.

Normal rollouts write ``summary.json`` automatically. This command is a
recovery and audit entry point for rebuilding that file exclusively from
committed episode artifacts; it does not derive task-specific verdicts.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bench.evaluation.artifacts import ArtifactWriter  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    return parser.parse_args(argv)


def summarize_run(run_dir: Path) -> dict[str, Any]:
    resolved = run_dir.expanduser().resolve()
    manifest_path = resolved / "run_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"run manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    seeds = manifest.get("seeds")
    if not isinstance(seeds, list):
        raise ValueError("run manifest seeds must be a list")
    writer = ArtifactWriter(resolved, expected_seeds=seeds)
    return writer.write_summary()


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary = summarize_run(args.run_dir)
    print(json.dumps(summary, sort_keys=True))
    return 0 if summary.get("publishable", False) else 2


if __name__ == "__main__":
    raise SystemExit(main())
