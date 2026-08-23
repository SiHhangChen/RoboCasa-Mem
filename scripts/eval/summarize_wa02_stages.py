#!/usr/bin/env python3
"""Summarize the authoritative WA-02 seven-stage evaluation funnel.

Only episodes containing ``diagnostics.stages.*.qualified`` for every stage
enter the formal funnel denominator. Older WA-02 artifacts did not record a
reliable cup-pick history; they are emitted as ``legacy_partial`` episode rows
with recoverable values, but are excluded from the seven-stage rates.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
EVAL_ROOT = ROOT / "outputs" / "eval"
EVALUATOR_PROFILE = "wa02-seven-stage-v1"
STAGES = (
    "first_cup_pick",
    "first_cup_fill",
    "first_cup_ready",
    "second_cup_pick",
    "second_cup_fill",
    "second_cup_ready",
    "selection_success",
)
STAGE_LABELS = {
    "first_cup_pick": "第一杯取杯",
    "first_cup_fill": "第一杯加水",
    "first_cup_ready": "第一杯就位",
    "second_cup_pick": "第二杯取杯",
    "second_cup_fill": "第二杯加水",
    "second_cup_ready": "第二杯就位",
    "selection_success": "选择成功",
}


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _manifest(run: Path) -> dict[str, Any]:
    path = run / "run_manifest.json"
    return _read_json(path) if path.is_file() else {}


def collect_runs() -> list[Path]:
    """Discover WA-02 rollout directories under the evaluation root."""

    runs: list[Path] = []
    if not EVAL_ROOT.is_dir():
        return runs
    for path in sorted(EVAL_ROOT.rglob("run_manifest.json")):
        run = path.parent
        if not (run / "episodes").is_dir():
            continue
        manifest = _read_json(path)
        task = manifest.get("task", {}) or {}
        task_id = task.get("id") or manifest.get("task_id")
        if str(task_id).upper() == "WA-02" or "wa02" in run.name.lower():
            runs.append(run)
    return runs


def load_episodes(run: Path) -> list[dict[str, Any]]:
    """Read standard episode artifacts, including optional ``nice`` reruns."""

    episodes = run / "episodes"
    if not episodes.is_dir():
        return []
    paths = list(episodes.glob("seed_*/result.json"))
    paths.extend(episodes.glob("nice/seed_*/result.json"))
    return [_read_json(path) for path in sorted(paths)]


def _authoritative_stages(
    episode: dict[str, Any],
) -> dict[str, bool] | None:
    diagnostics = episode.get("diagnostics", {}) or {}
    stages = diagnostics.get("stages", {}) or {}
    if not isinstance(stages, dict) or not all(
        isinstance(stages.get(stage), dict) for stage in STAGES
    ):
        return None
    return {stage: bool(stages[stage].get("qualified")) for stage in STAGES}


def _episode_facts(episode: dict[str, Any]) -> dict[str, Any]:
    metadata = episode.get("episode_metadata", {}) or {}
    resolved = metadata.get("resolved_episode", {}) or {}
    facts = resolved.get("facts", {}) or {}
    return dict(facts) if isinstance(facts, dict) else {}


def _legacy_partial_stages(
    episode: dict[str, Any],
) -> dict[str, bool | None]:
    """Recover only facts that old WA-02 result files actually preserved."""

    diagnostics = episode.get("diagnostics", {}) or {}
    oracle = diagnostics.get("oracle", {}) or {}
    facts = _episode_facts(episode)
    hot_cup = facts.get("hot_cup_id", oracle.get("hot_cup_id"))
    cold_cup = facts.get("cold_cup_id", oracle.get("cold_cup_id"))
    temperature_order = facts.get("fill_temperature_order")
    if not (
        isinstance(temperature_order, list | tuple)
        and len(temperature_order) == 2
        and hot_cup in {"cup_A", "cup_B"}
        and cold_cup in {"cup_A", "cup_B"}
    ):
        cup_order: tuple[str | None, str | None] = (None, None)
        expected_temperatures: tuple[str | None, str | None] = (None, None)
    else:
        by_temperature = {"hot": str(hot_cup), "cold": str(cold_cup)}
        normalized = tuple(str(value).lower() for value in temperature_order)
        if set(normalized) == {"hot", "cold"}:
            cup_order = (by_temperature[normalized[0]], by_temperature[normalized[1]])
            expected_temperatures = normalized
        else:
            cup_order = (None, None)
            expected_temperatures = (None, None)

    fill = diagnostics.get("fill", {}) or {}
    recorded_temperatures = fill.get("recorded_temperature_by_cup", {}) or {}
    service = diagnostics.get("service", {}) or {}
    recorded_slots = service.get("recorded_slot_by_cup", {}) or {}
    slot_by_cup = facts.get("service_slot_by_cup", oracle.get("service_slot_by_cup", {}))
    selection = diagnostics.get("selection", {}) or {}
    requested_cup = facts.get("requested_cup_id", oracle.get("requested_cup_id"))

    first, second = cup_order
    first_temperature, second_temperature = expected_temperatures

    def filled(cup: str | None, temperature: str | None) -> bool | None:
        if cup is None or temperature is None:
            return None
        return recorded_temperatures.get(cup) == temperature

    def ready(cup: str | None) -> bool | None:
        if cup is None or not isinstance(slot_by_cup, dict) or cup not in slot_by_cup:
            return None
        return recorded_slots.get(cup) == slot_by_cup[cup]

    final_taken = selection.get("final_taken_cup_id")
    selected_correctly = (
        None if requested_cup not in {"cup_A", "cup_B"} else final_taken == requested_cup
    )
    return {
        "first_cup_pick": None,
        "first_cup_fill": filled(first, first_temperature),
        "first_cup_ready": ready(first),
        "second_cup_pick": None,
        "second_cup_fill": filled(second, second_temperature),
        "second_cup_ready": ready(second),
        "selection_success": selected_correctly,
    }


def stage_values(episode: dict[str, Any]) -> tuple[str, dict[str, bool | None]]:
    authoritative = _authoritative_stages(episode)
    if authoritative is not None:
        _validate_monotonic(authoritative, context=f"seed={episode.get('seed', '?')}")
        return EVALUATOR_PROFILE, authoritative
    return "legacy_partial", _legacy_partial_stages(episode)


def stage_hits(episode: dict[str, Any]) -> dict[str, bool | None]:
    """Return episode stage values while preserving unknown legacy picks."""

    return stage_values(episode)[1]


def _validate_monotonic(values: dict[str, bool], *, context: str) -> None:
    for previous, current in zip(STAGES, STAGES[1:]):
        if values[current] and not values[previous]:
            raise ValueError(
                f"non-monotonic WA-02 stages ({context}): "
                f"{current}=true after {previous}=false"
            )


def _status(episode: dict[str, Any]) -> str:
    return str(
        episode.get(
            "status",
            "success" if episode.get("success") is True else "failure",
        )
    )


def episode_row(run: Path, episode: dict[str, Any]) -> dict[str, Any]:
    source, values = stage_values(episode)
    diagnostics = episode.get("diagnostics", {}) or {}
    evaluation = diagnostics.get("evaluation", {}) or {}
    status = _status(episode)
    included = source == EVALUATOR_PROFILE and status in {"success", "failure"}
    return {
        "run_id": _manifest(run).get("run_id", run.name),
        "seed": episode.get("seed", ""),
        "status": status,
        "success": bool(episode.get("success") is True),
        "stage_source": source,
        "included_in_funnel": included,
        "failure_reason": episode.get("failure_reason")
        or diagnostics.get("failure_code")
        or "",
        "evaluator_profile": evaluation.get("profile_id", source),
        "env_steps": episode.get("env_steps", ""),
        **{f"stage_{stage}": values[stage] for stage in STAGES},
    }


def _checkpoint_step(manifest: dict[str, Any]) -> str | int:
    checkpoint = (manifest.get("policy_contract", {}) or {}).get("checkpoint", {}) or {}
    step = checkpoint.get("step")
    if step not in (None, ""):
        return step
    uri = str(checkpoint.get("uri", "")).replace("file://", "")
    parts = Path(uri).parts
    for index, part in enumerate(parts):
        if part.isdigit() and index > 0 and parts[index - 1].upper() == "WA02":
            return part
    return ""


def summarize_run(run: Path) -> dict[str, Any]:
    episodes = load_episodes(run)
    manifest = _manifest(run)
    counts = {"success": 0, "failure": 0, "error": 0, "cancelled": 0}
    hits = {stage: 0 for stage in STAGES}
    n_ok = 0
    n_funnel = 0
    n_authoritative = 0
    n_legacy_partial = 0

    for episode in episodes:
        status = _status(episode)
        counts[status] = counts.get(status, 0) + 1
        if status in {"success", "failure"}:
            n_ok += 1
        source, values = stage_values(episode)
        if source == EVALUATOR_PROFILE:
            n_authoritative += 1
        else:
            n_legacy_partial += 1
        if source != EVALUATOR_PROFILE or status not in {"success", "failure"}:
            continue
        n_funnel += 1
        for stage in STAGES:
            hits[stage] += int(values[stage] is True)

    _validate_monotonic(hits, context=f"run={run.name}")
    if n_authoritative and n_legacy_partial:
        profile = "mixed"
    elif n_authoritative:
        profile = EVALUATOR_PROFILE
    else:
        profile = "legacy_partial"
    seeds = manifest.get("seeds", []) or []
    task = manifest.get("task", {}) or {}
    rollout = manifest.get("rollout", {}) or {}
    return {
        "run_id": manifest.get("run_id", run.name),
        "task_id": task.get("id", manifest.get("task_id", "WA-02")),
        "evaluator_profile": profile,
        "checkpoint_step": _checkpoint_step(manifest),
        "replan_steps": rollout.get("replan_steps", ""),
        "seeds_start": seeds[0] if seeds else "",
        "seeds_end": seeds[-1] if seeds else "",
        "n_episodes": len(episodes),
        "n_ok": n_ok,
        "n_funnel": n_funnel,
        "n_authoritative": n_authoritative,
        "n_legacy_partial": n_legacy_partial,
        "n_success": counts.get("success", 0),
        "n_failure": counts.get("failure", 0),
        "n_error": counts.get("error", 0),
        "n_cancelled": counts.get("cancelled", 0),
        **{f"{stage}_n": hits[stage] for stage in STAGES},
        **{
            f"{stage}_pct": round(100.0 * hits[stage] / n_funnel, 1)
            if n_funnel
            else 0.0
            for stage in STAGES
        },
        "created_at": manifest.get("created_at", ""),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="*", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=EVAL_ROOT / "wa02_stage_summary.csv",
    )
    parser.add_argument(
        "--episodes-output",
        type=Path,
        default=EVAL_ROOT / "wa02_episode_stage_results.csv",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    runs = [
        path.expanduser().resolve()
        for path in (args.runs or collect_runs())
        if path.is_dir()
    ]
    summaries = [summarize_run(run) for run in runs]

    summary_fields = [
        "run_id",
        "task_id",
        "evaluator_profile",
        "checkpoint_step",
        "replan_steps",
        "seeds_start",
        "seeds_end",
        "n_episodes",
        "n_ok",
        "n_funnel",
        "n_authoritative",
        "n_legacy_partial",
        "n_success",
        "n_failure",
        "n_error",
        "n_cancelled",
    ]
    for stage in STAGES:
        summary_fields.extend((f"{stage}_n", f"{stage}_pct"))
    summary_fields.append("created_at")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(summaries)

    rows = [episode_row(run, episode) for run in runs for episode in load_episodes(run)]
    episode_fields = [
        "run_id",
        "seed",
        "status",
        "success",
        "stage_source",
        "included_in_funnel",
        "failure_reason",
        "evaluator_profile",
        "env_steps",
        *[f"stage_{stage}" for stage in STAGES],
    ]
    args.episodes_output.parent.mkdir(parents=True, exist_ok=True)
    with args.episodes_output.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=episode_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    print("WA-02 七阶段达成率（仅新版 evaluator；error/cancelled 不进分母）")
    print("run", "n_funnel", *[STAGE_LABELS[stage] for stage in STAGES], sep="\t")
    for row in summaries:
        print(
            row["run_id"],
            row["n_funnel"],
            *[f"{row[f'{stage}_pct']:.1f}" for stage in STAGES],
            sep="\t",
        )
    print(f"已写入 {args.output}")
    print(f"已写入 {args.episodes_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
