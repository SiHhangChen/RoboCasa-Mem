#!/usr/bin/env python3
"""统计 GR00T WA-01 各次评估的分阶段达成漏斗，输出 CSV 表格。

用法：
    python scripts/eval/summarize_gr00t_stages.py [--output outputs/eval/gr00t_wa01_stage_summary.csv]

扫描 outputs/eval 下所有 wa01-gr00t* 运行目录（含 8月8日 嵌套层级
outputs/eval/gr00t/<task>/<step>/...），对每个 run 统计五个阶段：
    糖入杯 → 两杯就位 → 执行选杯 → 选对糖杯 → 最终成功
新结果直接使用 diagnostics.stages.<name>.qualified；旧结果仍从
physical_events / selection / success 字段兼容推断。error/cancelled 的
episode 计入 n_other 但不进漏斗分母。nice/ 目录下的补录 episode 会并入统计。
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EVAL_ROOT = ROOT / "outputs" / "eval"

STAGES = ["sugar", "one_ready", "ready", "selection", "correct", "success"]
STAGE_LABELS = {
    "sugar": "糖入杯",
    "one_ready": "一杯就位",
    "ready": "两杯就位",
    "selection": "执行选杯",
    "correct": "选对糖杯",
    "success": "最终成功",
}


def collect_runs() -> list[Path]:
    """递归收集所有 GR00T WA-01 运行目录（含 outputs/eval/gr00t/<task>/<step>/... 嵌套）。"""
    runs: list[Path] = []

    def walk(base: Path, depth: int) -> None:
        if not base.is_dir():
            return
        if (base / "episodes").is_dir():
            runs.append(base)
        if depth == 0:
            return
        for d in sorted(base.iterdir()):
            if d.is_dir() and not d.name.startswith("."):
                walk(d, depth - 1)

    if EVAL_ROOT.is_dir():
        for d in sorted(EVAL_ROOT.iterdir()):
            if d.name.startswith("wa01-gr00t"):
                walk(d, 0)
            elif d.name == "gr00t":
                walk(d, 3)
    return runs


def load_episodes(run: Path) -> list[dict]:
    """读取 run 下所有 episode result.json（含 nice/ 补录）。"""
    ep = run / "episodes"
    dirs = [d for d in ep.iterdir() if d.name.startswith("seed_")]
    nice = ep / "nice"
    if nice.is_dir():
        dirs += [d for d in nice.iterdir() if d.name.startswith("seed_")]
    rows = []
    for d in dirs:
        p = d / "result.json"
        if p.exists():
            with open(p, encoding="utf-8") as f:
                rows.append(json.load(f))
    return rows


def stage_hits(r: dict) -> dict[str, bool]:
    d = r.get("diagnostics", {})
    authoritative = d.get("stages", {})
    if isinstance(authoritative, dict) and all(
        isinstance(authoritative.get(stage), dict) for stage in STAGES
    ):
        return {
            stage: bool(authoritative[stage].get("qualified"))
            for stage in STAGES
        }
    pe = d.get("physical_events", {})
    return {
        "sugar": pe.get("sugar_delivery") is not None,
        # 一杯就位：至少一个咖啡杯被搬到餐桌上（on_dining_counter）
        "one_ready": any(
            d.get(c, {}).get("on_dining_counter")
            for c in ("coffee_A", "coffee_B")
        ),
        "ready": bool(pe.get("cups_ready_latched")),
        "selection": pe.get("selection") is not None,
        "correct": bool(d.get("selection", {}).get("correct")),
        "success": bool(r.get("success")),
    }


def summarize_run(run: Path) -> dict:
    with open(run / "run_manifest.json", encoding="utf-8") as f:
        manifest = json.load(f)
    rows = load_episodes(run)
    n_ok = sum(1 for r in rows if r.get("status") in ("success", "failure"))
    counts = {"success": 0, "failure": 0, "error": 0, "cancelled": 0}
    hits = {s: 0 for s in STAGES}
    for r in rows:
        status = r.get("status")
        if status in counts:
            counts[status] += 1
        if status not in ("success", "failure"):
            continue
        for s, hit in stage_hits(r).items():
            if hit:
                hits[s] += 1

    seeds = manifest.get("seeds", [])
    ckpt_uri = manifest.get("policy_contract", {}).get("checkpoint", {}).get("uri", "")
    # 从 checkpoint URI 提取训练步数（ckpts/gr00t/WA01/<step>/...），
    # 不依赖 run_id/目录名——曾出现目录名与实际加载步数不一致的情况。
    uri_parts = Path(ckpt_uri.replace("file://", "")).parts
    step = ""
    for i, part in enumerate(uri_parts):
        if part.isdigit() and i > 0 and uri_parts[i - 1].upper() == "WA01":
            step = part
            break
    return {
        "run_id": run.name,
        "checkpoint_step": step,
        "replan_steps": manifest.get("rollout", {}).get("replan_steps", ""),
        "max_steps": manifest.get("task", {}).get("identity", {}).get(
            "effective_rollout_max_steps", ""
        ),
        "seeds_start": seeds[0] if seeds else "",
        "seeds_end": seeds[-1] if seeds else "",
        "created_at": manifest.get("created_at", ""),
        "n_episodes": len(rows),
        **{f"n_{k}": v for k, v in counts.items()},
        **{f"{s}_n": hits[s] for s in STAGES},
        **{f"{s}_pct": round(hits[s] / n_ok * 100, 1) if n_ok else 0.0 for s in STAGES},
        "n_ok": n_ok,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default=str(EVAL_ROOT / "gr00t_wa01_stage_summary.csv"),
        help="输出 CSV 路径（默认 outputs/eval/gr00t_wa01_stage_summary.csv）",
    )
    args = parser.parse_args()

    runs = collect_runs()
    if not runs:
        print(f"未找到运行目录: {EVAL_ROOT}", file=sys.stderr)
        return 1

    summary = [summarize_run(run) for run in runs]

    fields = [
        "run_id", "checkpoint_step", "replan_steps", "max_steps",
        "seeds_start", "seeds_end", "n_episodes",
        "n_success", "n_failure", "n_error", "n_cancelled",
    ]
    for s in STAGES:
        fields += [f"{s}_n", f"{s}_pct"]
    fields += ["created_at"]

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in summary:
            writer.writerow(row)

    print(f"已写入 {len(summary)} 个运行: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
