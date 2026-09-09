#!/usr/bin/env python3
"""Finish the WA-03 GR00T checkpoint and stage-evaluation campaign."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import time


ROOT = Path(__file__).resolve().parents[3]
OUTPUT_ROOT = Path(os.environ.get("OUTPUT_ROOT", ROOT / "outputs/eval")).resolve()
STAGE_OUTPUT_ROOT = Path(
    os.environ.get("STAGE_OUTPUT_ROOT", ROOT / "outputs/stage_eval")
).resolve()
PREFIX_ROOT = Path(
    os.environ.get(
        "PREFIX_ROOT", ROOT / "outputs/stage_prefixes/wa03_gr00t_r40_v1"
    )
).resolve()
STAGE_SUITE = os.environ.get(
    "STAGE_SUITE", "configs/eval/stages/wa03_gr00t_r40_formal_v1.yaml"
)
STAGE_RECORD_VIDEO = os.environ.get("STAGE_RECORD_VIDEO", "0")
POLL_SECONDS = int(os.environ.get("POLL_SECONDS", "60"))

STEPS = ("080000", "090000", "100000")
PORT_BY_STEP = {"080000": "57103", "090000": "57104", "100000": "57105"}
GPU_BY_STEP = {"080000": "4", "090000": "5", "100000": "6"}


def summary_path(step: str) -> Path:
    return OUTPUT_ROOT / f"wa03-gr00t-{step}-50seeds-r40" / "summary.json"


def load_summary(step: str) -> dict[str, object] | None:
    path = summary_path(step)
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def summary_complete(step: str) -> bool:
    summary = load_summary(step)
    return bool(
        summary
        and summary.get("completed_task_episodes") == 50
        and summary.get("pending") == 0
    )


def prefix_count() -> int:
    if not PREFIX_ROOT.is_dir():
        return 0
    return sum(1 for path in PREFIX_ROOT.glob("seed_*/manifest.json") if path.is_file())


def score(step: str) -> tuple[float, ...]:
    summary = load_summary(step)
    if summary is None:
        raise RuntimeError(f"missing completed summary for checkpoint {step}")
    report_stages = summary.get("report_stages") or {}
    stages = report_stages.get("stages") or {}
    return (
        float(summary.get("overall_success_rate", 0.0)),
        float((stages.get("three_cups_ready") or {}).get("rate", 0.0)),
        float((stages.get("two_cups_ready") or {}).get("rate", 0.0)),
        float((stages.get("one_cup_ready") or {}).get("rate", 0.0)),
        float((stages.get("sugar_added") or {}).get("rate", 0.0)),
    )


def run_launcher(step: str, command: str, **extra: str) -> None:
    run_id = f"wa03-gr00t-{step}-50seeds-r40"
    environment = {
        **os.environ,
        "NO_PROXY": "127.0.0.1,localhost",
        "no_proxy": "127.0.0.1,localhost",
        "PORT": PORT_BY_STEP[step],
        "GPU": GPU_BY_STEP[step],
        "RENDER_GPU": GPU_BY_STEP[step],
        "CKPT_DIR": f"/data/chensihang/ckpts/gr00t/wa03/{step}/pretrained_model",
        "RUN_ID": run_id,
        "MAIN_ROLLOUT_RUN": str(OUTPUT_ROOT / run_id),
        "PREFIX_ROOT": str(PREFIX_ROOT),
        "STAGE_OUTPUT_ROOT": str(STAGE_OUTPUT_ROOT),
        "STAGE_RUN_ID": f"wa03-gr00t-{step}-stage-50seeds-r40-formal-v1",
        "STAGE_SUITE": STAGE_SUITE,
        "STAGE_RECORD_VIDEO": STAGE_RECORD_VIDEO,
        **extra,
    }
    subprocess.run(
        [str(ROOT / "scripts/eval/gr00t/eval_gr00t_wa03.sh"), command],
        cwd=ROOT,
        env=environment,
        check=True,
    )


def main() -> int:
    while prefix_count() < 50:
        print(f"[campaign] prefixes={prefix_count()}/50", flush=True)
        time.sleep(POLL_SECONDS)

    while not all(summary_complete(step) for step in STEPS):
        for step in STEPS:
            summary = load_summary(step) or {}
            print(
                f"[campaign] {step} "
                f"completed={summary.get('completed_task_episodes', 0)}/50 "
                f"success={float(summary.get('overall_success_rate', 0.0)):.4f}",
                flush=True,
            )
        time.sleep(POLL_SECONDS)

    for step in STEPS:
        print(f"[campaign] score {step}={score(step)}", flush=True)
    best_step = max(STEPS, key=score)
    print(f"[campaign] selected checkpoint={best_step}", flush=True)

    run_launcher(best_step, "stage-preflight")
    run_launcher(best_step, "stage-eval")
    run_launcher(best_step, "stage-summary")
    print(
        "[campaign] complete "
        f"stage_run={STAGE_OUTPUT_ROOT}/"
        f"wa03-gr00t-{best_step}-stage-50seeds-r40-formal-v1",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
