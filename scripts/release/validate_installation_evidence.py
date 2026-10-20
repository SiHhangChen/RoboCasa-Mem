#!/usr/bin/env python3
"""Validate recorded installed-wheel acceptance against current runtime files."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

TASKS = ("WX-01", "WA-01", "WA-05", "WR-01", "WR-08", "TS02", "TS03", "TM01", "TM02")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def runtime_files(root: Path) -> dict[str, str]:
    """Files that drive acceptance, excluding evidence and release metadata."""
    files = {}
    for folder in ("bench", "configs", "scripts"):
        for path in sorted((root / folder).rglob("*")):
            if not path.is_file() or path.suffix not in {".py", ".yaml", ".json", ".xml"}:
                continue
            if "__pycache__" in path.parts:
                continue
            relative = path.relative_to(root).as_posix()
            if relative in {
                "configs/benchmark/release_identity.json",
                "configs/benchmark/simulator_environment.json",
                "configs/benchmark/simulator_assets.json",
            } or relative.startswith("scripts/release/"):
                continue
            files[relative] = sha(path)
    return files


def config_hash(candidate: Path, episode: dict, *, allow_partial_calibration: bool) -> str:
    """Recompute the content identity, explicitly retaining calibration policy."""
    input_hashes = {
        "task_config:0": "sha256:" + sha(candidate / "bench/tasks" / episode["task_id"].lower().replace("-", "") / "resources/task.yaml"),
        "observation_profile": "sha256:" + sha(candidate / "configs/benchmark/observation_profiles" / (episode["observation_profile"] + ".yaml")),
    }
    # Relocate resource paths recorded under the installed wheel to the candidate.
    for name, original in episode["resources"].items():
        parts = Path(original).parts
        start = parts.index("bench")
        path = candidate.joinpath(*parts[start:])
        input_hashes["resource:" + name] = "sha256:" + sha(path) if path.is_file() else "missing"
    payload = {
        "schema_version": "membench.resolved_episode_inputs/v1",
        "task_id": episode["task_id"], "seed": episode["seed"],
        "env_split": episode["env_split"], "variant": episode["variant"],
        "task_options": episode["options"],
        "allow_partial_calibration": allow_partial_calibration,
        "control_frequency_hz": 20, "input_hashes": input_hashes,
    }
    return "sha256:" + hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def validate(candidate: Path) -> dict:
    directory = candidate / "docs/installation_evidence"
    index = json.loads((directory / "manifest.json").read_text())
    required = {
        "full_input_coverage.json", "restored_assets.json", "installation_setup.json",
        "software_checks.json", "installed_suite_selectors.json", "requirements-tested.txt",
        "worker.py", "installed_software_tests.log", "installed_websocket_tests.log",
    }
    if not required <= index["files"].keys():
        raise ValueError("installation evidence manifest is missing required files")
    for name, digest in index["files"].items():
        path = (directory / name).resolve()
        if not path.is_relative_to(directory.resolve()) or sha(path) != digest:
            raise ValueError(f"installation evidence hash mismatch: {name}")
    if runtime_files(candidate) != index["runtime_files_sha256"]:
        raise ValueError("current runtime differs from the accepted installed wheel")
    assets = json.loads((candidate / "configs/benchmark/simulator_assets.json").read_text())
    if assets["archive"]["sha256"] != index["asset_archive_sha256"]:
        raise ValueError("accepted asset archive differs from the current archive")
    coverage = json.loads((directory / "full_input_coverage.json").read_text())
    lock = json.loads((candidate / "configs/benchmark/evaluation_input_lock.json").read_text())
    identities = {
        (key.split("/")[0], int(key.split("/")[1])): value
        for key, value in lock["episode_identities"].items()
    }
    environment = json.loads((candidate / "configs/benchmark/simulator_environment.json").read_text())
    expected = {(task, seed) for task in TASKS for seed in range(1050, 1100)}
    checks = coverage["checks"]
    if len(checks) != 450 or {(r["task_id"], r["seed"]) for r in checks} != expected:
        raise ValueError("installation evidence must cover exactly the 450 official inputs")
    for check in checks:
        if (
            check["passed"] is not True
            or check["env_steps"] != 3
            or check["isolated_python"] is not True
            or check["blocked_development_accesses"]
            or len(check["images"]) != 3
            or check["state_dim"] != 37
            or check["action_dim"] != 13
            or check["control_frequency_hz"] != 20
            or check["policy_model_loaded"] is not False
            or any(image["shape"] != [224, 224, 3] for image in check["images"].values())
        ):
            raise ValueError(f"incomplete installed runtime check: {check['task_id']}/{check['seed']}")
        episode = check["resolved_episode"]
        if episode["seed"] != check["seed"] or episode["task_id"] != check["task_id"]:
            raise ValueError("installed episode identity differs from the requested case")
        identity = identities[check["task_id"], check["seed"]]
        if (
            episode["config_hash"] != config_hash(candidate, episode, allow_partial_calibration=False)
            or identity["release_config_hash"] != config_hash(candidate, episode, allow_partial_calibration=True)
            or episode["resource_hash"] != identity["release_resource_hash"]
            or episode["task_version"] != identity["task_version"]
        ):
            raise ValueError("installed episode differs from the release input lock")
        for name in ("mujoco", "numpy", "numba", "scipy", "Pillow", "robocasa", "robosuite"):
            if check["versions"][name] != environment["dependencies"][name]:
                raise ValueError(f"accepted simulator version differs: {name}")
    if coverage["passed"] is not True:
        raise ValueError("installed coverage did not pass")
    restored = json.loads((directory / "restored_assets.json").read_text())
    if restored["passed"] is not True or restored["checked_files"] != assets["file_count"]:
        raise ValueError("isolated asset restoration did not pass")
    setup = json.loads((directory / "installation_setup.json").read_text())
    if (
        setup["wheel"]["sha256"] != index["wheel_sha256"]
        or setup["asset_archive"]["sha256"] != index["asset_archive_sha256"]
        or setup["requirements"]["sha256"] != sha(directory / "requirements-tested.txt")
        or setup["worker_sha256"] != sha(directory / "worker.py")
        or coverage["worker_sha256"] != setup["worker_sha256"]
        or setup["python"] != environment["python"]
    ):
        raise ValueError("installed setup provenance differs from accepted artifacts")
    for name in ("robocasa", "robosuite"):
        if setup["sources"][name]["revision"] != environment["sources"][name]["revision"]:
            raise ValueError(f"accepted simulator source revision differs: {name}")
    software = json.loads((directory / "software_checks.json").read_text())
    if not software["software"]["passed"] or not software["localhost_websocket"]["passed"]:
        raise ValueError("installed software checks did not pass")
    selectors = json.loads((directory / "installed_suite_selectors.json").read_text())
    if selectors["passed"] is not True or selectors["suite_count"] != 27:
        raise ValueError("installed suite selector verification is incomplete")
    return index


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate", type=Path, nargs="?", default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()
    candidate = args.candidate.resolve()
    accepted = validate(candidate)
    release = json.loads((candidate / "release_candidate.json").read_text())
    for name, digest in release["files"].items():
        path = (candidate / name).resolve()
        if not path.is_relative_to(candidate) or sha(path) != digest:
            raise ValueError(f"release file hash mismatch: {name}")
    print(json.dumps({"status": "passed", "scope": accepted["scope"], "release_files_verified": len(release["files"])}))
