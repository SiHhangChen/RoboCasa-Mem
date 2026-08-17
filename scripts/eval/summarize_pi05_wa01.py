#!/usr/bin/env python3
"""将 pi05 WA-01 评测 run 汇总为 stage summary 并增量写回 CSV。

指标语义（bench/tasks/wa01/evaluation.py）：
  sugar_n     糖被投递到某杯咖啡（sugar_consumed）
  one_cup_n   至少一杯咖啡放在 dining counter
  ready_n     两杯曾一起就位（cups_together_before_selection）
  selection_n 完成加糖杯的选择
  correct_n   选择正确
  success_n   完整成功（evaluate_success）

用法:
    python scripts/eval/summarize_pi05_wa01.py [run_dir ...]

不传 run_dir 时自动扫描 outputs/eval/wa01-pi05-lora-* 目录。
CSV 中已存在的 run_id 会被更新，其余按 created_at 排序追加。
"""

from __future__ import annotations

import csv
import json
import sys
from datetime import datetime
from pathlib import Path

CSV_PATH = Path("outputs/eval/pi05_wa01_stage_summary.csv")
COLUMNS = [
    "run_id", "checkpoint_step", "seeds_start", "seeds_end",
    "n_episodes", "n_success", "n_failure", "n_error", "n_cancelled", "n_ok",
    "sugar_n", "sugar_pct", "one_cup_n", "one_cup_pct",
    "ready_n", "ready_pct", "selection_n", "selection_pct",
    "correct_n", "correct_pct", "success_n", "success_pct",
    "created_at",
]


def _load(run_dir: Path) -> dict:
    summary = json.loads((run_dir / "summary.json").read_text())
    manifest = json.loads((run_dir / "run_manifest.json").read_text())
    step = None
    uri = manifest["policy_contract"]["checkpoint"].get("uri", "")
    tail = uri.rstrip("/").split("/")[-1]
    if tail.isdigit():
        step = tail
    return {
        "run_id": manifest["run_id"],
        "checkpoint_step": step,
        "seeds": manifest.get("seeds", []),
        "summary": summary,
        "manifest": manifest,
        "episodes_dir": run_dir / "episodes",
    }


def _episode_metrics(episode: dict) -> tuple[bool, bool, bool, bool, bool, bool]:
    """从单个 episode result.json 提取六类阶段指标计数。

    新结果使用 evaluator 的 ``stages.*.qualified``，确保后置阶段不会绕过
    前置阶段。没有新字段的历史结果继续使用结束快照兼容推断。
    """

    d = episode.get("diagnostics", {})
    stages = d.get("stages", {})
    if isinstance(stages, dict) and all(
        isinstance(stages.get(name), dict)
        for name in ("sugar", "one_ready", "ready", "selection", "correct", "success")
    ):
        return (
            bool(stages["sugar"].get("qualified")),
            bool(stages["one_ready"].get("qualified")),
            bool(stages["ready"].get("qualified")),
            bool(stages["selection"].get("qualified")),
            bool(stages["correct"].get("qualified")),
            bool(stages["success"].get("qualified")),
        )
    sugar = bool(d.get("sugar", {}).get("consumed"))
    coffee_a = d.get("coffee_A", {})
    coffee_b = d.get("coffee_B", {})
    one_cup = bool(coffee_a.get("on_dining_counter") or coffee_b.get("on_dining_counter"))
    ready = bool(
        episode.get("status") == "success"
        or (
            coffee_a.get("on_dining_counter")
            and coffee_b.get("on_dining_counter")
            and coffee_a.get("upright")
            and coffee_b.get("upright")
        )
    )
    selected = bool(d.get("selection", {}).get("selected_coffee_id"))
    correct = bool(d.get("selection", {}).get("correct"))
    success = bool(episode.get("success"))
    return sugar, one_cup, ready, selected, correct, success


def _metrics(entry: dict) -> dict:
    counts = entry["summary"]["counts"]
    n = entry["summary"]["episodes_total"]
    agg = [0, 0, 0, 0, 0, 0]
    ep_dir = entry["episodes_dir"]
    valid = 0
    results = sorted(ep_dir.glob("*/result.json"))
    for path in results:
        ep = json.loads(path.read_text())
        # 过程指标只统计有效 episode（failure/success），连接中断/出错不计入，
        # 避免把基础设施 error 误当作策略未完成任务。
        if ep.get("status") not in ("failure", "success"):
            continue
        valid += 1
        for i, v in enumerate(_episode_metrics(ep)):
            agg[i] += int(v)
    sugar_n, one_cup_n, ready_n, selection_n, correct_n, success_n = agg

    def pct(k: int) -> float:
        return round(100.0 * k / valid, 1) if valid else 0.0

    seeds = sorted(entry["seeds"])
    return {
        "run_id": entry["run_id"],
        "checkpoint_step": entry["checkpoint_step"],
        "seeds_start": seeds[0] if seeds else "",
        "seeds_end": seeds[-1] if seeds else "",
        "n_episodes": n,
        "n_success": counts["success"],
        "n_failure": counts["failure"],
        "n_error": counts["error"],
        "n_cancelled": counts["cancelled"],
        "n_ok": entry["summary"]["completed_task_episodes"],
        "sugar_n": sugar_n, "sugar_pct": pct(sugar_n),
        "one_cup_n": one_cup_n, "one_cup_pct": pct(one_cup_n),
        "ready_n": ready_n, "ready_pct": pct(ready_n),
        "selection_n": selection_n, "selection_pct": pct(selection_n),
        "correct_n": correct_n, "correct_pct": pct(correct_n),
        "success_n": success_n, "success_pct": pct(success_n),
        "created_at": entry["manifest"].get("created_at", ""),
    }


def main() -> None:
    run_dirs = [Path(p) for p in sys.argv[1:]]
    if not run_dirs:
        run_dirs = sorted(Path("outputs/eval").glob("wa01-pi05-lora-*"))
    run_dirs = [d for d in run_dirs if d.is_dir()]

    rows: dict[str, dict] = {}
    if CSV_PATH.exists():
        with open(CSV_PATH, encoding="utf-8-sig", newline="") as f:
            rows = {r["run_id"]: r for r in csv.DictReader(f)}

    for run_dir in run_dirs:
        try:
            row = _metrics(_load(run_dir))
        except FileNotFoundError as exc:
            print(f"跳过（缺少文件）: {run_dir}: {exc}", file=sys.stderr)
            continue
        rows[row["run_id"]] = row
        print(f"汇总 {row['run_id']}: n={row['n_episodes']} "
              f"success={row['success_n']}/{row['n_episodes']} "
              f"sugar={row['sugar_n']} ready={row['ready_n']} "
              f"selection={row['selection_n']} correct={row['correct_n']}")

    ordered = sorted(rows.values(), key=lambda r: (r.get("created_at") or "", r["run_id"]))
    with open(CSV_PATH, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(ordered)
    print(f"已写入 {CSV_PATH}: {len(ordered)} 行")


if __name__ == "__main__":
    main()
