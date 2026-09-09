#!/usr/bin/env python3
"""Freeze prefix-derived stage budgets into a formal stage suite."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import tempfile
import sys

import yaml


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bench.evaluation.expert_prefix import load_expert_prefix  # noqa: E402
from bench.evaluation.prefix_statistics import summarize_expert_prefixes  # noqa: E402
from bench.evaluation.stage_integration import (  # noqa: E402
    get_stage_evaluation_integration,
)
from bench.evaluation.stage_suite import load_stage_eval_suite  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-suite", type=Path, required=True)
    parser.add_argument("--prefix-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stage-suite-version", required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    suite = load_stage_eval_suite(args.stage_suite)
    integration = get_stage_evaluation_integration(
        suite.task_id,
        stage_ids=suite.stage_ids,
    )
    integration.validate_stage_suite(suite.stage_ids)
    prefix_root = (
        suite.expert_prefix_root
        if args.prefix_root is None
        else args.prefix_root.expanduser().resolve()
    )
    artifacts = [
        load_expert_prefix(prefix_root, seed)
        for seed in suite.base_suite.seeds
    ]
    summary = summarize_expert_prefixes(
        artifacts,
        stage_order=suite.stage_ids,
        stage_segments=integration.budget_segments,
        expected_seeds=suite.base_suite.seeds,
    )
    if not summary["ready_for_budget_freeze"]:
        raise RuntimeError("expert prefixes are not complete and valid")
    budgets = summary["recommended_budgets"]
    if any(budgets[stage_id] is None for stage_id in suite.stage_ids):
        raise RuntimeError("prefix summary did not produce every stage budget")

    payload = {
        "stage_suite_version": str(args.stage_suite_version),
        "task_id": suite.task_id,
        "base_suite": str(suite.base_suite.source_path),
        "expert_prefix_root": str(prefix_root),
        "stages": [
            {
                "id": target.id,
                "source": target.source,
                "model_step_budget": int(budgets[target.id]),
            }
            for target in suite.targets
        ],
    }
    output = args.output.expanduser().resolve()
    _atomic_yaml(output, payload)
    print(f"formal_stage_suite={output}")
    print(f"budgets={budgets}")
    return 0


def _atomic_yaml(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            yaml.safe_dump(payload, handle, sort_keys=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


if __name__ == "__main__":
    raise SystemExit(main())
