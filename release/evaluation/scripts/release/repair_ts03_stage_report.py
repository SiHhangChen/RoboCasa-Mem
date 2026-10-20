#!/usr/bin/env python3
"""Recover the misplaced TS03 seed 1056 report into a separate audit bundle.

The source run is read only. This repairs a field name, without simulation,
model inference, inferred verdicts, or changes to the evaluator's rules.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bench.evaluation.report_stages import validate_report_stages
from bench.evaluation.summary import aggregate_episode_results

RUN_ID = "ts03-pi05-20000-r40-50seeds-v1"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def checked_report(result: dict, report: dict) -> None:
    """Cross-check saved public verdicts against saved internal diagnostics."""
    validate_report_stages(
        report,
        expected_task_id="TS03",
        expected_profile_id="TS03/report-v2",
        expected_success=result["success"],
    )
    diagnostics = result["diagnostics"]
    if diagnostics["success"] != result["success"]:
        raise ValueError("diagnostics and episode success differ")
    for name, verdict in report["stages"].items():
        internal = diagnostics["stages"][name]
        applicable = internal["applicable"]
        passed = internal["qualified"] if applicable else None
        step = internal["qualified_step"] if passed else None
        if verdict != {
            "applicable": applicable,
            "passed": passed,
            "first_pass_step": step,
        }:
            raise ValueError(f"seed {result['seed']}: {name} contradicts diagnostics")


def repair_result(original: dict) -> dict:
    if original.get("task_id") != "TS03" or original.get("seed") != 1056:
        raise ValueError("repair applies only to TS03 seed 1056")
    if "report_stages" in original or "" not in original:
        raise ValueError("expected a misplaced report under the empty field name")
    checked_report(original, original[""])
    repaired = deepcopy(original)
    repaired["report_stages"] = repaired.pop("")
    return repaired


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def verify_bundle(bundle: Path) -> dict:
    provenance = json.loads((bundle / "provenance.json").read_text())
    if provenance["source_run_id"] != RUN_ID:
        raise ValueError("unexpected source run")
    for name, digest in provenance["files"].items():
        if sha(bundle / name) != digest:
            raise ValueError(f"repair bundle hash mismatch: {name}")
    original = json.loads((bundle / "original_result.json").read_text())
    repaired = json.loads((bundle / "repaired_result.json").read_text())
    if repair_result(original) != repaired:
        raise ValueError("repair must change only the report field name")
    before = json.loads((bundle / "original_summary.json").read_text())
    after = json.loads((bundle / "summary.json").read_text())
    if not after["publishable"] or not after["report_stages"]["complete"]:
        raise ValueError("repaired stage reports remain incomplete")
    if after["report_stages"]["reported_episodes"] != 50:
        raise ValueError("expected 50 stage reports")
    for key in before:
        if key not in {"publishable", "report_stages"} and before[key] != after[key]:
            raise ValueError(f"repair changed unrelated summary field: {key}")
    return provenance


def repair_run(source: Path, output: Path) -> dict:
    source = source.resolve()
    output = output.resolve()
    if output.is_relative_to(source) or source.is_relative_to(output):
        raise ValueError("output must be separate from the source run")
    if output.exists():
        raise FileExistsError(output)
    manifest = json.loads((source / "run_manifest.json").read_text())
    if manifest["run_id"] != RUN_ID:
        raise ValueError("unexpected source run")
    paths = sorted((source / "episodes").glob("seed_*/result.json"))
    items = [json.loads(path.read_text()) for path in paths]
    if sorted(item["seed"] for item in items) != list(range(1050, 1100)):
        raise ValueError("expected exactly seeds 1050–1099")
    original_summary = json.loads((source / "summary.json").read_text())
    before = aggregate_episode_results(items, range(1050, 1100))
    if any(original_summary.get(key) != value for key, value in before.items()):
        raise ValueError("source summary does not match the saved episode results")
    repaired_items = []
    for item in items:
        if item["task_id"] != "TS03":
            raise ValueError("unexpected task")
        repaired = repair_result(item) if item["seed"] == 1056 else deepcopy(item)
        checked_report(repaired, repaired["report_stages"])
        repaired_items.append(repaired)
    summary = aggregate_episode_results(repaired_items, range(1050, 1100))
    # Preserve extra artifact-validation fields from the original summary.
    summary = {**original_summary, **summary}
    if original_summary.get("invalid_episode_artifacts"):
        raise ValueError("source has invalid episode artifacts")
    original_path = source / "episodes/seed_001056/result.json"
    output.mkdir(parents=True)
    (output / "original_result.json").write_bytes(original_path.read_bytes())
    (output / "original_summary.json").write_bytes((source / "summary.json").read_bytes())
    write_json(output / "repaired_result.json", repair_result(json.loads(original_path.read_text())))
    write_json(output / "summary.json", summary)
    provenance = {
        "schema_version": "membench.historical_report_repair/v1",
        "source_run_id": RUN_ID,
        "source_run": str(source),
        "source_manifest_sha256": sha(source / "run_manifest.json"),
        "source_result_sha256": {
            str(path.relative_to(source)): sha(path) for path in paths
        },
        "operation": "Rename empty top-level key to report_stages for seed 1056 only.",
        "verification": "All 50 stage reports validated against schema and saved internal diagnostics; summaries recomputed from all 50 results.",
        "simulation_or_model_rerun": False,
        "source_modified": False,
        "files": {
            name: sha(output / name)
            for name in ("original_result.json", "original_summary.json", "repaired_result.json", "summary.json")
        },
    }
    write_json(output / "provenance.json", provenance)
    verify_bundle(output)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_run", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    summary = repair_run(args.source_run, args.output)
    print(json.dumps({"publishable": summary["publishable"], "report_stages": summary["report_stages"]}))
