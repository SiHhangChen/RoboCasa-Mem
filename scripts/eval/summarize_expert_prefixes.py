#!/usr/bin/env python3
"""Validate frozen expert prefixes and compute stage-length budgets."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bench.evaluation.expert_prefix import load_expert_prefix  # noqa: E402
from bench.evaluation.prefix_statistics import summarize_expert_prefixes  # noqa: E402
from bench.evaluation.stage_integration import (  # noqa: E402
    get_stage_evaluation_integration,
)
from bench.evaluation.stage_suite import load_stage_eval_suite  # noqa: E402


DEFAULT_STAGE_SUITE = ROOT / "configs/eval/stages/wa05_gr00t_r40_v1.yaml"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-suite", type=Path, default=DEFAULT_STAGE_SUITE)
    parser.add_argument("--prefix-root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--require-complete",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    suite = load_stage_eval_suite(args.stage_suite)
    integration = get_stage_evaluation_integration(
        suite.task_id,
        stage_ids=suite.stage_ids,
    )
    integration.validate_stage_suite(suite.stage_ids)
    root = (
        suite.expert_prefix_root
        if args.prefix_root is None
        else args.prefix_root.expanduser().resolve()
    )
    artifacts = []
    load_errors: dict[str, str] = {}
    for seed in suite.base_suite.seeds:
        try:
            artifacts.append(load_expert_prefix(root, seed))
        except Exception as exc:
            load_errors[str(seed)] = f"{type(exc).__name__}: {exc}"
    summary = summarize_expert_prefixes(
        artifacts,
        stage_order=suite.stage_ids,
        stage_segments=integration.budget_segments,
        expected_seeds=suite.base_suite.seeds,
    )
    summary.update(
        {
            "task_id": suite.task_id,
            "prefix_root": str(root),
            "stage_suite": {
                "path": str(suite.source_path),
                "version": suite.stage_suite_version,
                "hash": suite.content_hash,
            },
            "load_errors": load_errors,
        }
    )
    summary["ready_for_budget_freeze"] = bool(
        summary["ready_for_budget_freeze"] and not load_errors
    )
    rendered = json.dumps(summary, indent=2, sort_keys=True)
    print(rendered)
    if args.output is not None:
        _atomic_json(args.output.expanduser().resolve(), summary)
    if args.require_complete and not summary["ready_for_budget_freeze"]:
        return 2
    return 0


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    except Exception:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass
        raise


if __name__ == "__main__":
    raise SystemExit(main())
