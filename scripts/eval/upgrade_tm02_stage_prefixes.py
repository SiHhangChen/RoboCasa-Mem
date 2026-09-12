#!/usr/bin/env python3
"""Derive TM02 v2 prefixes from validated v1 event history without overwriting it."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bench.evaluation.expert_prefix import load_expert_prefix, write_expert_prefix  # noqa: E402
from bench.evaluation.stage_suite import load_stage_eval_suite  # noqa: E402
from bench.tasks.tm02.evaluation import STAGE_ORDER  # noqa: E402
from bench.tasks.tm02.prefix_migration import upgrade_v1_prefix  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--stage-suite", type=Path, required=True)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--seed", type=int, action="append", dest="seeds")
    args = parser.parse_args(argv)
    suite = load_stage_eval_suite(args.stage_suite)
    if suite.task_id != "TM02" or suite.stage_ids != STAGE_ORDER:
        raise ValueError("destination must be a TM02 five-stage suite")
    source = args.source_root.expanduser().resolve()
    destination = (args.output_root or suite.expert_prefix_root).expanduser().resolve()
    if source == destination:
        raise ValueError("source and destination prefix roots must differ")
    seeds = tuple(args.seeds) if args.seeds else suite.base_suite.seeds
    if len(seeds) != len(set(seeds)) or set(seeds) - set(suite.base_suite.seeds):
        raise ValueError("migration seeds must be unique members of the destination suite")

    # Preflight every source and destination before creating any artifacts.
    upgraded = []
    for seed in seeds:
        target = destination / f"seed_{seed:06d}"
        if target.exists():
            raise FileExistsError(f"refusing to overwrite an existing prefix: {target}")
        old = load_expert_prefix(source, seed)
        if old.seed != seed:
            raise ValueError(f"source manifest seed does not match directory: {seed}")
        new = upgrade_v1_prefix(old)
        upgraded.append(replace(new, provenance={
            **new.provenance,
            "source_prefix_root": str(source),
            "source_stage_suite_version": old.provenance.get("stage_suite_version"),
            "source_stage_suite_hash": old.provenance.get("stage_suite_hash"),
            "stage_suite_version": suite.stage_suite_version,
            "stage_suite_hash": suite.content_hash,
        }))
    for artifact in upgraded:
        write_expert_prefix(destination, artifact)
        print(f"seed={artifact.seed} boundaries={dict(artifact.stage_boundary_steps)}")
    print(f"migrated={len(upgraded)} prefix_root={destination} actions_unchanged=True")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
