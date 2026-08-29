#!/usr/bin/env python3
"""Run expert-prefix stage evaluation against a remote policy server."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bench.evaluation.adapter import get_adapter, preflight_adapter  # noqa: E402
from bench.evaluation.canonical_stage import (  # noqa: E402
    CanonicalStageTask,
    RecordedExpertController,
    RemotePolicyModelAdapter,
)
from bench.evaluation.client import RemotePolicyClient  # noqa: E402
from bench.evaluation.expert_prefix import load_expert_prefix  # noqa: E402
from bench.evaluation.main_stage import import_main_rollout_stage  # noqa: E402
from bench.evaluation.stage_artifacts import StageArtifactWriter  # noqa: E402
from bench.evaluation.stage_rollout import (  # noqa: E402
    INVALID_PREFIX,
    StageEvaluationResult,
    evaluate_stage,
)
from bench.evaluation.stage_suite import load_stage_eval_suite  # noqa: E402
from bench.tasks import register_builtin_task_plugins  # noqa: E402
from bench.tasks.wa05.evaluation import STAGE_ORDER  # noqa: E402
from bench.tasks.wa05.stage_evaluation import (  # noqa: E402
    build_stage_specs,
    privileged_stage_state,
)


DEFAULT_STAGE_SUITE = ROOT / "configs/eval/stages/wa05_gr00t_r40_v1.yaml"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--stage-suite", type=Path, default=DEFAULT_STAGE_SUITE)
    parser.add_argument("--prefix-root", type=Path)
    parser.add_argument("--main-rollout-run", type=Path)
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs/stage_eval")
    parser.add_argument("--run-id")
    parser.add_argument("--target-stage", action="append", dest="target_stages")
    parser.add_argument("--seed", action="append", type=int, dest="seeds")
    parser.add_argument("--api-key-env", default="MEMBENCH_POLICY_API_KEY")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--record-video", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    stage_suite = load_stage_eval_suite(args.stage_suite)
    if stage_suite.task_id != "WA-05":
        raise ValueError("the first stage runner integration currently supports WA-05 only")
    suite = stage_suite.base_suite
    prefix_root = (
        stage_suite.expert_prefix_root
        if args.prefix_root is None
        else args.prefix_root.expanduser().resolve()
    )
    targets = tuple(args.target_stages or [target.id for target in stage_suite.targets[1:]])
    for target in targets:
        config = stage_suite.target(target)
        if config.source != "expert_prefix":
            raise ValueError(
                f"target {target!r} uses {config.source}; extract it from the main rollout"
            )
    seeds = tuple(args.seeds) if args.seeds else suite.seeds

    register_builtin_task_plugins()
    adapter = get_adapter(stage_suite.task_id)
    client = RemotePolicyClient(
        args.endpoint,
        api_key=os.environ.get(args.api_key_env),
        timeout_s=suite.transport.timeout_s,
        connect_attempts=suite.transport.connect_attempts,
        connect_retry_delay_s=suite.transport.connect_retry_delay_s,
    )
    try:
        contract = client.connect()
        preflight = preflight_adapter(adapter, suite=suite, policy_contract=contract)
        if suite.rollout.replan_steps > contract.action.horizon:
            raise ValueError(
                f"replan_steps={suite.rollout.replan_steps} exceeds policy horizon "
                f"{contract.action.horizon}"
            )
        if args.preflight_only:
            print(
                f"status=ok task={stage_suite.task_id} stages={','.join(targets)} "
                f"replan_steps={suite.rollout.replan_steps} horizon={contract.action.horizon}"
            )
            return 0

        run_id = args.run_id or _default_run_id(contract.server_id)
        writer = StageArtifactWriter(args.output_root / run_id)
        main_target = stage_suite.targets[0]
        summary_targets = (
            (main_target.id, *targets)
            if args.main_rollout_run is not None
            else targets
        )
        writer.write_manifest(
            {
                "schema_version": "membench.stage_run_manifest/v1",
                "run_id": run_id,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "stage_suite": {
                    "path": str(stage_suite.source_path),
                    "version": stage_suite.stage_suite_version,
                    "hash": stage_suite.content_hash,
                },
                "base_suite": {
                    "path": str(suite.source_path),
                    "version": suite.suite_version,
                    "hash": suite.content_hash,
                },
                "task_id": stage_suite.task_id,
                "targets": list(summary_targets),
                "target_sources": {
                    target.id: target.source
                    for target in stage_suite.targets
                    if target.id in summary_targets
                },
                "seeds": list(seeds),
                "expert_prefix_root": str(prefix_root),
                "main_rollout_run": (
                    None
                    if args.main_rollout_run is None
                    else str(args.main_rollout_run.expanduser().resolve())
                ),
                "policy_contract": contract.to_hello(),
                "preflight": preflight,
            }
        )
        if args.main_rollout_run is not None:
            imported = import_main_rollout_stage(
                writer,
                args.main_rollout_run,
                task_id=stage_suite.task_id,
                target_stage=main_target.id,
                model_step_budget=main_target.model_step_budget,
                replan_steps=suite.rollout.replan_steps,
                seeds=seeds,
                expected_suite_hash=suite.content_hash,
                expected_policy_contract=contract.to_hello(),
                resume=args.resume,
            )
            print(
                f"stage={main_target.id} source=main_rollout "
                f"imported={len(imported)}"
            )
        record_video = (
            suite.artifacts.record_video
            if args.record_video is None
            else bool(args.record_video)
        )
        specs = build_stage_specs(stage_suite.budgets)
        for target in targets:
            for seed in seeds:
                if args.resume and writer.completed(target, seed):
                    print(f"stage={target} seed={seed} status=skipped")
                    continue
                try:
                    prefix = load_expert_prefix(prefix_root, seed)
                except Exception as exc:
                    result = _invalid_prefix_result(
                        stage_suite.task_id,
                        target,
                        seed,
                        f"expert_prefix_load_error:{type(exc).__name__}",
                    )
                    writer.write_result(
                        result,
                        source="expert_prefix",
                        evaluator_metadata={"error": str(exc)},
                    )
                    print(f"stage={target} seed={seed} status={result.status}")
                    continue
                if not prefix.success:
                    result = _invalid_prefix_result(
                        stage_suite.task_id,
                        target,
                        seed,
                        f"expert_prefix_invalid:{prefix.failure_reason or 'unknown'}",
                    )
                    writer.write_result(
                        result,
                        source="expert_prefix",
                        expert_prefix_manifest=prefix.manifest(),
                    )
                    print(f"stage={target} seed={seed} status={result.status}")
                    continue
                try:
                    prefix_actions = prefix.prefix_for_target(STAGE_ORDER, target)
                except Exception as exc:
                    result = _invalid_prefix_result(
                        stage_suite.task_id,
                        target,
                        seed,
                        f"expert_prefix_boundary_error:{type(exc).__name__}",
                    )
                    writer.write_result(
                        result,
                        source="expert_prefix",
                        expert_prefix_manifest=prefix.manifest(),
                        evaluator_metadata={"error": str(exc)},
                    )
                    print(f"stage={target} seed={seed} status={result.status}")
                    continue
                task = CanonicalStageTask(
                    task_id=stage_suite.task_id,
                    stage_specs=specs,
                    adapter=adapter,
                    suite=suite,
                    policy_contract=contract,
                    privileged_state_builder=privileged_stage_state,
                    expected_episode_identity=prefix.episode_identity,
                )
                model = RemotePolicyModelAdapter(client=client, task=task)
                expert = RecordedExpertController(prefix_actions)
                session = writer.start_episode(
                    target_stage=target,
                    seed=seed,
                    task=task,
                    record_video=record_video,
                    video_cameras=suite.artifacts.video_cameras,
                    video_fps=suite.artifacts.video_fps,
                )
                try:
                    result = evaluate_stage(
                        task,
                        target,
                        seed,
                        model,
                        expert,
                        expert_prefix_budget=int(prefix_actions.shape[0]),
                        replan_steps=suite.rollout.replan_steps,
                        record_logs=False,
                        logger=session,
                    )
                    session.finish(
                        result,
                        source="expert_prefix",
                        request_metrics=model.request_metrics,
                        expert_prefix_manifest=prefix.manifest(),
                        evaluator_metadata={
                            "model_reset_calls": model.reset_calls,
                            "replan_steps": suite.rollout.replan_steps,
                        },
                    )
                except BaseException:
                    session.abort()
                    task.close()
                    raise
                print(
                    f"stage={target} seed={seed} status={result.status} "
                    f"prefix_steps={result.expert_prefix_steps} "
                    f"model_steps={result.model_control_steps}"
                )
        summary = writer.write_summary()
        print(f"run_dir={writer.run_dir}")
        print(f"summary={summary}")
        return 0
    finally:
        client.close()


def _default_run_id(server_id: str) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    safe = "".join(
        character if character.isalnum() or character in "-_" else "-"
        for character in server_id
    )
    return f"wa05-stage-{safe}-{timestamp}"


def _invalid_prefix_result(
    task_id: str,
    target_stage: str,
    seed: int,
    failure_reason: str,
) -> StageEvaluationResult:
    return StageEvaluationResult(
        task_id=str(task_id),
        target_stage=str(target_stage),
        seed=int(seed),
        status=INVALID_PREFIX,
        success=False,
        success_step=None,
        expert_prefix_steps=0,
        model_control_steps=0,
        total_model_inference_calls=0,
        environment_steps=0,
        handoff_observation_id=None,
        failure_reason=str(failure_reason),
    )


if __name__ == "__main__":
    raise SystemExit(main())
