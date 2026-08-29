#!/usr/bin/env python3
"""Rebuild a stage evaluation summary from committed per-seed results."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bench.evaluation.stage_artifacts import StageArtifactWriter  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args(argv)
    writer = StageArtifactWriter(args.run_dir)
    print(writer.write_summary())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
