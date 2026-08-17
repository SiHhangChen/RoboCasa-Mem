#!/usr/bin/env python3
"""Merge collected WX01 manual-grasp poses into the single-wall grasp candidates file.

Each collected JSON is a ``manual_grasp_pose`` written by ``calibrate_anchor_pose.py``
(with ``--pose-name`` matching a candidate ``name``). This script replaces each
candidate's ``eef_site_in_object`` with the freshly calibrated value while keeping
the file's name / wall_azimuth_deg / metadata order.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_TEMPLATE = ROOT / "bench/tasks/wx01/resources/poses/blocking_cup_single_wall_grasp_candidates.json"


def _collect_payloads(path: Path) -> dict[str, dict]:
    """Map pose_name -> manual_grasp_pose payload for a file or directory of files."""

    payloads: dict[str, dict] = {}
    sources = list(path.glob("*.json")) if path.is_dir() else [path]
    for source in sources:
        payload = json.loads(source.read_text(encoding="utf-8"))
        if str(payload.get("config_kind", "")) != "manual_grasp_pose":
            raise ValueError(f"not a manual_grasp_pose payload: {source}")
        pose_name = str(payload.get("pose_name") or source.stem).strip()
        if pose_name in payloads:
            raise ValueError(f"duplicate pose_name {pose_name!r}: {source}")
        payloads[pose_name] = payload
    return payloads


def _candidate_payload(payload: dict) -> dict:
    selected = payload.get("selected") or {}
    eef_in_object = (
        selected.get("eef_site_in_object")
        if isinstance(selected, dict)
        else None
    )
    if not isinstance(eef_in_object, dict):
        raise ValueError(
            "collected manual_grasp_pose has no selected.eef_site_in_object"
        )
    return {
        "p": list(eef_in_object.get("p") or []),
        "q_xyzw": list(eef_in_object.get("q_xyzw") or []),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--collected",
        type=Path,
        required=True,
        help="A manual_grasp_pose JSON or a directory of them.",
    )
    parser.add_argument(
        "--template",
        type=Path,
        default=DEFAULT_TEMPLATE,
        help="Candidates file holding name / wall_azimuth_deg / metadata (default: WX01 candidates).",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    template = json.loads(args.template.read_text(encoding="utf-8"))
    candidates = list(template.get("candidates") or [])
    if not candidates:
        raise ValueError(f"template has no candidates: {args.template}")

    collected = _collect_payloads(args.collected)
    missing = [str(c.get("name")) for c in candidates if c.get("name") not in collected]
    if missing:
        raise SystemExit(f"missing collected poses for: {', '.join(missing)}")

    updated = []
    for candidate in candidates:
        name = str(candidate["name"])
        eef = _candidate_payload(collected[name])
        candidate = dict(candidate)
        candidate["eef_site_in_object"] = eef
        updated.append(candidate)
    template = dict(template)
    template["candidates"] = updated

    if args.dry_run:
        print(
            json.dumps(template, indent=2, sort_keys=True)
        )
        return 0
    output = args.output or args.template
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(template, indent=2, sort_keys=True) + "\n")
    print(f"merged {len(updated)} candidates -> {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
