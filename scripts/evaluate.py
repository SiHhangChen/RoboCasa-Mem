#!/usr/bin/env python
"""Unified policy-neutral evaluation entrypoint."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bench.evaluation.suite import load_eval_suite
from bench.tasks import TaskCatalog


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate a policy on a MobileMemBench suite")
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args(argv)
    suite = load_eval_suite(args.suite)
    try:
        canonical = TaskCatalog.canonical_id(suite.task_id)
    except KeyError:
        raise SystemExit(f"unknown task id in suite: {suite.task_id}")
    print(f"suite={suite.suite_version} task={canonical} seeds={list(suite.seeds)}")
    if args.preflight_only:
        return 0
    raise SystemExit("evaluation transport execution is available through scripts/eval/run_remote.py during Phase 4")


if __name__ == "__main__":
    raise SystemExit(main())
