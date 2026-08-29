#!/usr/bin/env python3
"""Build frozen low-level WA-05 expert trajectories for stage evaluation."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bench.evaluation.canonical_stage import resolve_suite_episode  # noqa: E402
from bench.evaluation.expert_prefix import (  # noqa: E402
    capture_expert_trajectory,
    load_expert_prefix,
    write_expert_prefix,
)
from bench.evaluation.stage_suite import load_stage_eval_suite  # noqa: E402
from bench.runtime import RoboCasaRuntimeFactory  # noqa: E402
from bench.tasks import TaskCatalog, resolve_builtin_task_plugin  # noqa: E402
from bench.tasks.contracts import ExpertPlanInputs  # noqa: E402
from bench.tasks.wa05.evaluation import STAGE_ORDER  # noqa: E402
from bench.tasks.wa05.stage_evaluation import (  # noqa: E402
    privileged_stage_state,
    stage_completed,
)


DEFAULT_STAGE_SUITE = ROOT / "configs/eval/stages/wa05_gr00t_r40_v1.yaml"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-suite", type=Path, default=DEFAULT_STAGE_SUITE)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--seed", action="append", type=int, dest="seeds")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--allow-invalid", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    stage_suite = load_stage_eval_suite(args.stage_suite)
    if stage_suite.task_id != "WA-05":
        raise ValueError("this prefix builder currently supports WA-05 only")
    suite = stage_suite.base_suite
    output_root = (
        stage_suite.expert_prefix_root
        if args.output_root is None
        else args.output_root.expanduser().resolve()
    )
    seeds = tuple(args.seeds) if args.seeds else suite.seeds
    definition = TaskCatalog.get(stage_suite.task_id)
    runtime_api = resolve_builtin_task_plugin(stage_suite.task_id).load_runtime_api()
    if runtime_api is None:
        raise RuntimeError("WA-05 has no canonical runtime API")

    invalid = 0
    for seed in seeds:
        manifest_path = output_root / f"seed_{seed:06d}" / "manifest.json"
        if manifest_path.is_file() and not args.overwrite:
            artifact = load_expert_prefix(output_root, seed)
            print(
                f"seed={seed} status=existing success={artifact.success} "
                f"actions={artifact.actions.shape[0]} "
                f"boundaries={dict(artifact.stage_boundary_steps)}"
            )
            invalid += int(not artifact.success)
            continue

        episode = resolve_suite_episode(suite, seed)
        descriptor = runtime_api.build_descriptor(episode, True)
        plan = definition.build_expert_plan(ExpertPlanInputs.from_descriptor(descriptor))
        runtime = RoboCasaRuntimeFactory(runtime_api).create_from_descriptor(
            descriptor,
            expert=True,
        )
        try:
            artifact = capture_expert_trajectory(
                runtime,
                plan,
                stage_ids=STAGE_ORDER,
                stage_state_builder=privileged_stage_state,
                stage_completed=stage_completed,
                provenance={
                    "stage_suite_version": stage_suite.stage_suite_version,
                    "stage_suite_hash": stage_suite.content_hash,
                    "base_suite_version": suite.suite_version,
                    "base_suite_hash": suite.content_hash,
                    "git_revision": _git_revision(),
                },
            )
        finally:
            runtime.close()
        write_expert_prefix(output_root, artifact)
        print(
            f"seed={seed} status=written success={artifact.success} "
            f"actions={artifact.actions.shape[0]} boundaries={dict(artifact.stage_boundary_steps)} "
            f"failure={artifact.failure_reason}"
        )
        invalid += int(not artifact.success)

    print(f"prefix_root={output_root}")
    print(f"attempted={len(seeds)} invalid={invalid}")
    return 0 if invalid == 0 or args.allow_invalid else 2


def _git_revision() -> str:
    result = subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


if __name__ == "__main__":
    raise SystemExit(main())
