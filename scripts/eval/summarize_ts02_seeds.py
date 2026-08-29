#!/usr/bin/env python3
"""Build a per-seed TS02 search and stopping report from evaluation artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bench.tasks.ts02.domain.episode import sample_target_fixture  # noqa: E402


TARGET_LABELS = {
    "cabinet_slot_0": "左 (slot_0)",
    "cabinet_slot_1": "中 (slot_1)",
    "cabinet_slot_2": "右 (slot_2)",
}
FIELDNAMES = (
    "Checkpoint",
    "Seed",
    "目标位置",
    "找到步骤",
    "正确停止",
    "最终失败原因",
)


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _nested(value: Any, *keys: str) -> Any:
    current = value
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _checkpoint_name(run: Path) -> str:
    match = re.search(r"ts02-gr00t-(\d+)", run.name)
    return match.group(1) if match else run.name


def _target_fixture(episode: dict[str, Any]) -> str | None:
    diagnostics = _mapping(episode.get("diagnostics"))
    fixture = diagnostics.get("target_fixture")
    if fixture is None:
        fixture = _nested(
            episode,
            "episode_metadata",
            "resolved_episode",
            "facts",
            "target_fixture",
        )
    return str(fixture) if fixture is not None else None


def _failure_reason(episode: dict[str, Any]) -> str:
    status = str(episode.get("status", "unknown"))
    if status == "success":
        return "-"
    diagnostics = _mapping(episode.get("diagnostics"))
    reason = (
        episode.get("failure_reason")
        or episode.get("error")
        or diagnostics.get("failure_code")
        or diagnostics.get("failure_attribution")
    )
    if reason is None:
        reason = status
    return f"{status}: {reason}" if status in {"error", "cancelled"} else str(reason)


def _row(run: Path, episode: dict[str, Any]) -> dict[str, str]:
    target_fixture = _target_fixture(episode)
    seed = int(episode["seed"])
    if target_fixture is None:
        target_fixture = sample_target_fixture(seed)
    found_step = _nested(
        episode,
        "report_stages",
        "stages",
        "target_found",
        "first_pass_step",
    )
    target_found = bool(
        _nested(episode, "report_stages", "stages", "target_found", "passed")
    )
    stopped = _nested(
        episode,
        "report_stages",
        "stages",
        "search_stopped_after_target",
        "passed",
    )
    stopped_label = "是" if stopped is True else "否" if target_found else "-"
    return {
        "Checkpoint": _checkpoint_name(run),
        "Seed": str(seed),
        "目标位置": TARGET_LABELS.get(target_fixture, target_fixture or "unknown"),
        "找到步骤": str(found_step) if found_step is not None else "-",
        "正确停止": stopped_label,
        "最终失败原因": _failure_reason(episode),
    }


def load_rows(runs: list[Path]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for run in runs:
        for path in sorted((run / "episodes").glob("seed_*/result.json")):
            episode = json.loads(path.read_text(encoding="utf-8"))
            rows.append(_row(run, episode))
    return sorted(rows, key=lambda row: (row["Checkpoint"], int(row["Seed"])))


def _markdown(rows: list[dict[str, str]]) -> str:
    headings = ("Checkpoint", "Seed", "目标位置", "找到步骤", "正确停止", "最终失败原因")
    lines = [
        "| " + " | ".join(headings) + " |",
        "| " + " | ".join("---" for _ in headings) + " |",
    ]
    for row in rows:
        values = [
            row["Checkpoint"],
            row["Seed"],
            row["目标位置"],
            row["找到步骤"],
            row["正确停止"],
            row["最终失败原因"],
        ]
        lines.append("| " + " | ".join(_markdown_cell(value) for value in values) + " |")
    return "\n".join(lines) + "\n"


def _markdown_cell(value: str) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    args = parser.parse_args()

    runs = [path.expanduser().resolve() for path in args.runs]
    missing = [str(path) for path in runs if not (path / "episodes").is_dir()]
    if missing:
        raise FileNotFoundError(f"Run episode directories are missing: {missing}")
    rows = load_rows(runs)

    output_md = args.output_md.expanduser().resolve()
    output_csv = args.output_csv.expanduser().resolve()
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text(_markdown(rows), encoding="utf-8")
    with output_csv.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    print(f"rows={len(rows)}")
    print(f"markdown={output_md}")
    print(f"csv={output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
