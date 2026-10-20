#!/usr/bin/env python3
"""Record completed checks and refresh the candidate file manifest.

This records verification evidence, not a public-release approval. Installed
acceptance is recorded only when its evidence matches the candidate runtime.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from repair_ts03_stage_report import verify_bundle
from verify_assets import load_manifest
from validate_installation_evidence import validate as validate_installation

ROOT = Path(__file__).resolve().parents[2]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def finalize(candidate):
    inventory = candidate / "configs/benchmark/setup_replays/wx01"
    paths = list(inventory.glob("[0-9]*.json"))
    if {int(p.stem) for p in paths} != set(range(1050, 1100)):
        raise ValueError("candidate must contain all 50 WX-01 initialization transcripts")
    entries = []
    for path in sorted(paths):
        artifact = json.loads(path.read_text())
        if not artifact["verified"] or not artifact["validation"]["trajectory_observations_equal"]:
            raise ValueError(f"unverified WX-01 transcript: {path}")
        entries.append(
            {
                "seed": artifact["seed"],
                "file": path.name,
                "sha256": sha(path),
                "setup_steps": sum(e["kind"] == "step" for e in artifact["events"]),
                "validation": artifact["validation"],
            }
        )
    manifest = {
        "schema_version": "membench.wx01_setup_manifest/v1",
        "seed_range": [1050, 1100],
        "task_version": "1.0.0",
        "transcript_count": 50,
        "note": "Current source-generated initialization; setup precedes model control and is excluded from rollout budgets.",
        "entries": entries,
    }
    (inventory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    inputs = json.loads((candidate / "configs/benchmark/evaluation_input_lock.json").read_text())
    if inputs["episode_count"] != 450 or inputs["suite_count"] != 27:
        raise ValueError("missing full task input comparison")
    checks = json.loads((candidate / "docs/simulation_equivalence.json").read_text())
    expected = {"WX-01", "WA-01", "WA-05", "WR-01", "WR-08", "TS02", "TS03", "TM01", "TM02"}
    if {c["task_id"] for c in checks["checks"]} != expected:
        raise ValueError("missing nine-task simulation comparison")
    path = candidate / "release_candidate.json"
    result = json.loads(path.read_text())
    result["validation"] = {
        "suite_count": 27,
        "resolved_episode_count": 450,
        "wx01_setup_seeds": 50,
        "simulation_task_count": 9,
        "simulation_seed": 1050,
        "simulation_action_steps_per_task": 20,
        "wx01_setup_action_steps_per_seed": 20,
        "historical_full_model_rollouts_rerun": False,
    }
    result["remaining_release_requirements"] = [
        "Verify the recorded asset archive covers all nine candidate tasks and their configured variants.",
        "Accept an isolated installation of the current candidate using only pinned simulator sources and the recorded asset archive.",
    ]
    asset_path = candidate / "configs/benchmark/simulator_assets.json"
    assets = load_manifest(asset_path)
    asset_evidence_path = candidate / "docs/asset_evidence/current_bundle_verification.json"
    asset_evidence = json.loads(asset_evidence_path.read_text())
    if (
        asset_evidence["manifest_sha256"] != sha(asset_path)
        or asset_evidence["passed"] is not True
        or asset_evidence["checked_files"] != assets["file_count"]
        or asset_evidence["failures"]
    ):
        raise ValueError("missing or inconsistent installed-asset verification")
    asset_audit_path = candidate / "docs/evaluation_asset_audit.json"
    asset_audit = json.loads(asset_audit_path.read_text())
    if (
        asset_audit["manifest_sha256"] != sha(asset_path)
        or asset_audit["archive_sha256"] != assets["archive"]["sha256"]
        or asset_audit["current_verification_sha256"] != sha(asset_evidence_path)
    ):
        raise ValueError("asset audit does not match the pinned archive verification")
    result["simulator_assets"] = {
        "status": "existing_archive_inventoried_and_locally_verified",
        "manifest": str(asset_path.relative_to(candidate)),
        "manifest_sha256": sha(asset_path),
        "archive_sha256": assets["archive"]["sha256"],
        "file_count": assets["file_count"],
        "verification": str(asset_evidence_path.relative_to(candidate)),
        "verification_sha256": sha(asset_evidence_path),
        "audit": str(asset_audit_path.relative_to(candidate)),
        "audit_sha256": sha(asset_audit_path),
        "nine_task_coverage_verified": False,
        "clean_candidate_installation_verified": False,
    }
    installation_manifest = candidate / "docs/installation_evidence/manifest.json"
    if installation_manifest.is_file():
        accepted = validate_installation(candidate)
        result["installation_acceptance"] = {
            "status": "passed",
            "scope": "Same-host isolated Python environment, independent pinned simulator sources and restored archive; 450 official task/seed inputs, three action steps each.",
            "manifest": str(installation_manifest.relative_to(candidate)),
            "manifest_sha256": sha(installation_manifest),
            "accepted_wheel_sha256": accepted["wheel_sha256"],
            "runtime_files_match_accepted_wheel": True,
            "model_inference": False,
        }
        result["simulator_assets"]["status"] = "verified_for_all_450_official_inputs_in_isolated_installation"
        result["simulator_assets"]["nine_task_coverage_verified"] = True
        result["simulator_assets"]["clean_candidate_installation_verified"] = True
        result["remaining_release_requirements"] = [
            "Freeze the final release source revision and provide release users with the pinned asset archive."
        ]
    else:
        result.pop("installation_acceptance", None)
    repair = candidate / "docs/historical_result_repairs/ts03_pi05_r40"
    if repair.is_dir():
        verify_bundle(repair)
        result["historical_result_repairs"] = {
            "ts03_pi05_r40_seed1056": {
                "status": "repaired_and_verified",
                "provenance": str((repair / "provenance.json").relative_to(candidate)),
                "summary": str((repair / "summary.json").relative_to(candidate)),
                "provenance_sha256": sha(repair / "provenance.json"),
            }
        }
    else:
        result.pop("historical_result_repairs", None)
        result["remaining_release_requirements"].append(
            "Resolve missing report_stages for historical TS03 pi05 seed 1056 before publishing its complete stage results."
        )
    result["release_ready"] = False
    implementation = {
        str(p.relative_to(candidate)): sha(p)
        for p in sorted((candidate / "bench").rglob("*"))
        if p.is_file() and p.suffix in (".py", ".yaml", ".json", ".xml")
    }
    identity = {
        "schema_version": "membench.release_identity/v1",
        "version": "0.1.0rc1",
        "source_revision": result["source_revision"],
        "implementation_sha256": hashlib.sha256(
            json.dumps(implementation, sort_keys=True).encode()
        ).hexdigest(),
        "suite_inventory_sha256": sha(
            candidate / "configs/benchmark/evaluation_suite_sources.json"
        ),
        "input_lock_sha256": sha(candidate / "configs/benchmark/evaluation_input_lock.json"),
        "wx01_setup_manifest_sha256": sha(inventory / "manifest.json"),
        "simulator_environment_sha256": sha(
            candidate / "configs/benchmark/simulator_environment.json"
        ),
        "simulator_assets_sha256": sha(asset_path),
    }
    (candidate / "configs/benchmark/release_identity.json").write_text(
        json.dumps(identity, indent=2) + "\n"
    )
    ignored = {
        "__pycache__",
        ".pytest_cache",
        "build",
        "dist",
        "MobileMemBench.egg-info",
        "mobilemembench.egg-info",
    }
    result["files"] = {
        str(p.relative_to(candidate)): sha(p)
        for p in sorted(candidate.rglob("*"))
        if p.is_file()
        and p != path
        and not any(part in ignored for part in p.relative_to(candidate).parts)
    }
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["validation"]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate", type=Path)
    args = parser.parse_args()
    finalize(args.candidate.resolve())
