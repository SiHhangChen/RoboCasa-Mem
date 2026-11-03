#!/usr/bin/env python
"""Compute paper SCR for one task's complete 50-seed policy evaluation run.

Read committed episode verdicts, never model-specific diagnostics or a batch
summary. No policy server, simulator, checkpoint, or paper PDF is needed.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bench.evaluation.report_stages import validate_report_stages  # noqa: E402
from bench.evaluation.schemas import validate_episode_result  # noqa: E402
from bench.evaluation.summary import aggregate_episode_results  # noqa: E402
from bench.task_ids import normalize_task_id  # noqa: E402


METRIC_ID = "robocasa-mem.paper-scr/v1"
EPISODE_COUNT = 50
# Freeze the stage set as well as its source profile: changing a profile must
# not silently change the metric. MilkMatch has six paper stages, not seven.
PAPER_STAGES = {
    "WX-01": ("PutBackCup", "WX-01/report-v2", (
        ("Clean", "wipe_completed"), ("Tool", "cleaning_tool_released"),
        ("Return", "success"))),
    "WA-01": ("SweetCup", "WA-01/report-v1", (
        ("Sugar", "sugar_added"), ("Stage-1", "one_cup_ready"),
        ("Stage-2", "two_cups_ready"), ("Select", "success"))),
    "WA-05": ("TempReport", "WA-05/report-v1", (
        ("Fill", "cup_filled"), ("Place", "cup_staged"), ("Success", "success"))),
    "WR-01": ("CanRecall", "WR-01/report-v2", (
        ("Close-1", "one_can_stored"), ("Close-2", "two_cans_stored"),
        ("Reopen", "correct_cabinet_opened"), ("Success", "success"))),
    "WR-08": ("MilkMatch", "WR-08/report-v3", (
        ("P1", "one_item_picked"), ("D1", "one_item_placed"),
        ("C1", "one_item_matched"), ("P2", "two_items_picked"),
        ("D2", "two_items_placed"), ("C2", "two_items_matched"))),
    "TS-02": ("DrawerSearch", "TS-02/report-v1", (
        ("Open/Close", "one_drawer_checked"), ("Find", "target_found"),
        ("Stop", "search_stopped_after_target"), ("Button", "button_pressed"),
        ("Count", "correct_press_count"))),
    "TS-03": ("SugarCount", "TS-03/report-v2", (
        ("Cube-1", "sugar_1"), ("Cube-2", "sugar_2"),
        ("Cube-3", "sugar_3"), ("Success", "success"))),
    "TM-01": ("HeatAndInspect", "TM-01/report-v2", (
        ("Stove", "burner_on"), ("Inspect", "one_drawer_inspected"),
        ("Either", "one_thread_completed"), ("Both", "both_threads_completed"),
        ("Success", "success"))),
    "TM-02": ("FillAndRetrieve", "TM-02/report-v2", (
        ("Faucet", "faucet_on"), ("Retrieve", "pot_filled_and_can_retrieved"),
        ("Either", "one_thread_completed"), ("Both", "both_threads_completed"),
        ("Success", "success"))),
}
NOTES = [
    "Paper Section 3.4 / Eq.(3): SCR is the unweighted mean of per-stage "
    "passed/applicable rates, using the stage sets in Tables 6/7.",
    "An episode failing before an applicable stage remains in that stage's "
    "denominator. In SugarCount, Cube-2/Cube-3 apply only when the requested "
    "count is at least 2/3. Applicability comes from saved report_stages.",
    "CanRecall Close-1 means closing either first door; Close-2 means both "
    "doors closed. No upper/lower closing order is required. "
    "Historical field names one_can_stored/two_cans_stored "
    "represent these door stages, not can counts.",
    "MilkMatch uses P1,D1,C1,P2,D2,C2; the extra public success stage is "
    "excluded from SCR. SR always uses the original terminal success verdict.",
    "FillAndRetrieve Retrieve is B1: retrieval and release in the placement "
    "region while the faucet is running and the pot is filled, not just a pickup.",
    "This is one 50-seed run. Repeated-round means and sample standard "
    "deviations must be calculated from separate run scores.",
]


def _read_object(path: Path) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value, hashlib.sha256(raw).hexdigest()


def compute_run_scr(run_dir: Path) -> dict[str, Any]:
    """Validate one complete run and derive its score without modifying inputs."""
    run = run_dir.expanduser().resolve()
    manifest, manifest_hash = _read_object(run / "run_manifest.json")
    if manifest.get("schema_version") != "membench.run_manifest/v1":
        raise ValueError("unsupported run_manifest schema")
    seeds = manifest.get("seeds")
    if (not isinstance(seeds, list) or len(seeds) != EPISODE_COUNT
            or any(type(seed) is not int or seed < 0 for seed in seeds)
            or len(set(seeds)) != EPISODE_COUNT):
        raise ValueError("run_manifest must declare exactly 50 distinct non-negative integer seeds")
    task = manifest.get("task")
    if not isinstance(task, dict) or not isinstance(task.get("id"), str) or normalize_task_id(task["id"]) not in PAPER_STAGES:
        raise ValueError("run_manifest must identify one of the nine benchmark tasks")
    task_id = normalize_task_id(task["id"])
    task_name, profile_id, specification = PAPER_STAGES[task_id]
    selected_ids = [stage_id for _, stage_id in specification]
    public_ids = selected_ids + (["success"] if task_id == "WR-08" else [])
    contract = manifest.get("policy_contract")
    if not isinstance(contract, dict) or not isinstance(contract.get("policy"), dict):
        raise ValueError("run_manifest must contain policy_contract.policy")

    episodes_dir = run / "episodes"
    expected_dirs = {f"seed_{seed:06d}" for seed in seeds}
    actual_dirs = {path.name for path in episodes_dir.glob("seed_*") if path.is_dir()}
    if actual_dirs != expected_dirs:
        raise ValueError(
            f"incomplete 50-seed run: missing={sorted(expected_dirs - actual_dirs)}, "
            f"unexpected={sorted(actual_dirs - expected_dirs)}"
        )
    results, episodes = [], []
    for seed in sorted(seeds):
        directory = episodes_dir / f"seed_{seed:06d}"
        try:
            for name in ("actions.npz", "request_metrics.jsonl"):
                if not (directory / name).is_file():
                    raise ValueError(f"missing required artifact: {name}")
            raw, result_hash = _read_object(directory / "result.json")
            result = validate_episode_result(raw, expected_seed=seed)
            if result["status"] not in {"success", "failure"}:
                raise ValueError(f"episode status={result['status']!r}; resume/retry the evaluation first")
            if result.get("task_id") != task_id:
                raise ValueError(f"episode task_id does not match manifest task {task_id}")
            report = validate_report_stages(
                result.get("report_stages"), expected_task_id=task_id,
                expected_profile_id=profile_id, expected_success=result["success"],
            )
            if report["order"] != public_ids:
                raise ValueError(f"report stage order does not match {METRIC_ID}")
        except (OSError, ValueError, TypeError) as exc:
            raise ValueError(f"seed {seed}: {exc}") from exc
        result["report_stages"] = report
        results.append(result)
        episodes.append({"seed": seed, "success": result["success"],
            "result_path": str((directory / "result.json").relative_to(run)),
            "result_sha256": result_hash,
            "stages": {name: report["stages"][name] for name in selected_ids}})

    summary = aggregate_episode_results(results, expected_seeds=seeds)
    if not summary["publishable"]:
        raise ValueError("run did not pass the canonical result completeness checks")
    stages, rates = {}, []
    for column, name in specification:
        values = summary["report_stages"]["stages"][name]
        if values["denominator"] == 0:
            raise ValueError(
                f"stage {name!r} has no applicable episodes; paper SCR is undefined "
                "for this seed set (the stage cannot be silently dropped or scored zero)"
            )
        rate = Fraction(values["passed"], values["denominator"])
        rates.append(rate)
        stages[name] = {"paper_stage": column, **values}
    scr = sum(rates, Fraction(0)) / len(rates)
    return {"schema_version": "robocasa-mem.run_scr/v1", "metric_id": METRIC_ID,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "run_id": manifest.get("run_id", run.name), "run_dir": str(run),
        "task_id": task_id, "task_name": task_name, "profile_id": profile_id,
        "policy": contract["policy"], "checkpoint": contract.get("checkpoint"),
        "suite": manifest.get("suite"), "rollout": manifest.get("rollout"),
        "source_revision": manifest.get("client_code_revision"),
        "manifest_sha256": manifest_hash,
        "episode_count": len(results), "seeds": sorted(seeds),
        "counts": summary["counts"], "publishable": True,
        "scr": float(scr), "scr_exact_fraction": str(scr),
        "task_success_rate": summary["success_rate"],
        "stage_count": len(stages), "stage_order": selected_ids,
        "excluded_public_stages_from_scr": [name for name in public_ids if name not in selected_ids],
        "stages": stages, "episodes": episodes, "definition_notes": NOTES}


def _csv_text(rows: list[dict[str, Any]]) -> str:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def write_report(report: dict[str, Any], output_dir: Path) -> None:
    output = output_dir.expanduser().resolve()
    episodes = Path(report["run_dir"]) / "episodes"
    if output == episodes or episodes in output.parents:
        raise ValueError("output directory must be outside the raw episodes directory")
    stage_rows = [{"task_id": report["task_id"], "stage": name, **report["stages"][name]}
                  for name in report["stage_order"]]
    episode_rows = [{"task_id": report["task_id"], "seed": episode["seed"],
        "stage": name, "paper_stage": report["stages"][name]["paper_stage"],
        **episode["stages"][name], "task_success": episode["success"],
        "result_path": episode["result_path"]}
        for episode in report["episodes"] for name in report["stage_order"]]
    policy = report["policy"].get("name", report["policy"].get("family", "unknown"))
    lines = [f"# {report['task_id']} / {report['task_name']} — {policy}", "",
        f"Run: `{report['run_id']}`; 50 seeds; metric: `{METRIC_ID}`.", "",
        f"**SCR: {report['scr']:.2%}; SR: {report['task_success_rate']:.2%}.**", "",
        "| Stage | Source field | Passed / applicable | Not applicable | Rate |",
        "|---|---|---:|---:|---:|"]
    for name in report["stage_order"]:
        stage = report["stages"][name]
        lines.append(f"| {stage['paper_stage']} | `{name}` | "
            f"{stage['passed']}/{stage['denominator']} | {stage['not_applicable']} | {stage['rate']:.2%} |")
    lines += ["", "Metric notes:", ""] + [f"- {note}" for note in NOTES]
    lines += ["", "Source run: `" + report["run_dir"] + "`.", "",
        "Scores are recomputed from result.json files; an existing summary.json is not used or overwritten.", ""]
    files = {"stages.csv": "\ufeff" + _csv_text(stage_rows),
        "episode_stages.csv": "\ufeff" + _csv_text(episode_rows),
        "SCR.md": "\n".join(lines),
        "scr.json": json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n"}
    output.mkdir(parents=True, exist_ok=True)
    # Stage the completed reports before replacing earlier derived reports.
    with tempfile.TemporaryDirectory(prefix=".scr-", dir=output) as temporary:
        for name, content in files.items():
            (Path(temporary) / name).write_text(content, encoding="utf-8")
        for name in files:
            os.replace(Path(temporary) / name, output / name)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, help="One task/policy run with 50 seeds and run_manifest.json")
    parser.add_argument("--output-dir", type=Path, help="Derived reports directory (default: RUN_DIR/scr)")
    args = parser.parse_args(argv)
    try:
        report = compute_run_scr(args.run_dir)
        output = (args.output_dir or args.run_dir / "scr").expanduser().resolve()
        write_report(report, output)
    except (OSError, ValueError, TypeError, KeyError) as exc:
        print(f"SCR error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"task_id": report["task_id"], "policy": report["policy"],
        "episode_count": report["episode_count"], "scr": report["scr"],
        "task_success_rate": report["task_success_rate"], "output_dir": str(output)},
        ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
