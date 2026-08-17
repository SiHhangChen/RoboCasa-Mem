#!/usr/bin/env python3
"""统计 GR00T WR-03 各次评估的分阶段达成漏斗，输出 CSV 表格。

用法：
    python scripts/eval/summarize_wr03_stages.py

默认只汇总正式全 horizon 评测（run 名以 ``-50seed-20action`` 结尾）。
旧版 smoke/实验 run 不会进入正式报表；使用 ``--include-legacy-runs``
才显式包含它们。脚本同时输出逐 seed 的最终判定，避免把“到达过某个中间
阶段”误读为最终成功。

扫描 outputs/eval/gr00t/WR03/<step>/<run_id>/ 和根级 outputs/eval/wr03-gr00t-*/
下的运行目录，对每个 run 统计 WR-03 记忆任务的达成漏斗（判定依据来自各
episode result.json 的 diagnostics.success_checks 字段，与
bench/tasks/wr03/domain/semantics.py 的 SuccessReport 一致）：
    存储入柜 → 双门关闭 → 记忆锁定 → 正确取回门 → 同物取出 → 最终成功

error/cancelled 的 episode 计入 n_error/n_cancelled 但不进漏斗分母。
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EVAL_ROOT = ROOT / "outputs" / "eval"
OFFICIAL_EVALUATOR_PROFILE = "wr03-50seed-20action"
OFFICIAL_RUN_SUFFIX = "-50seed-20action"

STAGES = [
    "storage",
    "doors_closed",
    "barrier",
    "correct_door",
    "retrieved",
    "success",
]
STAGE_LABELS = {
    "storage": "存储入柜",
    "doors_closed": "双门关闭",
    "barrier": "记忆锁定",
    "correct_door": "正确取回门",
    "retrieved": "同物取出",
    "success": "最终成功",
}


def _checks(d: dict) -> dict:
    """从 diagnostics 提取 success_checks（缺失时回退空字典）。"""
    return d.get("success_checks", {}) if isinstance(d, dict) else {}


def _evaluator_profile(run: Path) -> str:
    manifest_path = run / "run_manifest.json"
    if manifest_path.is_file():
        with open(manifest_path, encoding="utf-8") as f:
            manifest = json.load(f)
        profile = manifest.get("replay", {}).get("evaluator_profile")
        if profile:
            return str(profile)
    if run.name.endswith(OFFICIAL_RUN_SUFFIX):
        return OFFICIAL_EVALUATOR_PROFILE
    return "legacy-or-unversioned"


def collect_runs(*, include_legacy: bool = False) -> list[Path]:
    """递归收集 WR-03 评测运行目录。"""
    runs: list[Path] = []

    def walk(base: Path) -> None:
        if not base.is_dir():
            return
        if (base / "episodes").is_dir():
            runs.append(base)
            return
        for d in sorted(base.iterdir()):
            if d.is_dir() and not d.name.startswith("."):
                walk(d)

    root = EVAL_ROOT / "gr00t" / "WR03"
    if root.is_dir():
        walk(root)
    if EVAL_ROOT.is_dir():
        for d in sorted(EVAL_ROOT.iterdir()):
            if d.name.startswith("wr03-gr00t") and d.is_dir() and (d / "episodes").is_dir():
                runs.append(d)
    if include_legacy:
        return runs
    return [run for run in runs if _evaluator_profile(run) == OFFICIAL_EVALUATOR_PROFILE]


def load_episodes(run: Path) -> list[dict]:
    """读取 run 下所有 episode result.json。"""
    ep = run / "episodes"
    if not ep.is_dir():
        return []
    rows = []
    for d in sorted(ep.iterdir()):
        if not d.name.startswith("seed_"):
            continue
        p = d / "result.json"
        if p.exists():
            with open(p, encoding="utf-8") as f:
                rows.append(json.load(f))
    return rows


def stage_hits(r: dict) -> dict[str, bool]:
    checks = _checks(r.get("diagnostics", {}))
    d = r.get("diagnostics", {})
    hits = {
        "storage": bool(checks.get("can_storage_verified")),
        "doors_closed": bool(checks.get("both_doors_closed_after_storage")),
        "barrier": bool(checks.get("memory_barrier_latched")),
        "correct_door": bool(
            checks.get("correct_retrieval_door")
            and checks.get("no_wrong_retrieval_door")
        ),
        "retrieved": bool(checks.get("same_can_retrieved")),
    }
    # “最终成功”必须以 episode 的最终判定为准，不能因为历史上曾短暂满足
    # 某个条件就被计为成功。
    hits["success"] = bool(
        r.get("status") == "success"
        and r.get("success") is True
        and d.get("success") is True
        and not r.get("failure_reason")
        and not d.get("retrieval", {}).get("protocol_violation")
    )
    return hits


def episode_row(run: Path, r: dict) -> dict:
    diagnostics = r.get("diagnostics", {})
    hits = stage_hits(r)
    storage = diagnostics.get("storage", {})
    retrieval = diagnostics.get("retrieval", {})
    return {
        "run_id": run.name,
        "checkpoint_step": "",
        "evaluator_profile": _evaluator_profile(run),
        "seed": r.get("seed", ""),
        "status": r.get("status", ""),
        "success": hits["success"],
        "failure_reason": r.get("failure_reason") or diagnostics.get("failure_code") or "",
        "env_steps": r.get("env_steps", ""),
        "stored_level": storage.get("stored_level", ""),
        "storage_hidden": bool(storage.get("storage_hidden")),
        "opened_level": retrieval.get("opened_level", ""),
        "retrieved_item_id": retrieval.get("retrieved_item_id", ""),
        **{f"stage_{stage}": hits[stage] for stage in STAGES},
    }


def checkpoint_step(run: Path, manifest: dict) -> str:
    """从 checkpoint URI 提取训练步数（ckpts/gr00t/WR03/<step>/...）。"""
    uri = manifest.get("policy_contract", {}).get("checkpoint", {}).get("uri", "")
    parts = Path(uri.replace("file://", "")).parts
    for i, part in enumerate(parts):
        if part.isdigit() and i > 0 and parts[i - 1].upper() == "WR03":
            return part
    return ""


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
    pending = max(0, len(seeds) - len(rows))
    return {
        "run_id": run.name,
        "checkpoint_step": checkpoint_step(run, manifest),
        "evaluator_profile": _evaluator_profile(run),
        "replan_steps": manifest.get("rollout", {}).get("replan_steps", ""),
        "max_steps": manifest.get("task", {}).get("identity", {}).get(
            "effective_rollout_max_steps", ""
        ),
        "seeds_start": seeds[0] if seeds else "",
        "seeds_end": seeds[-1] if seeds else "",
        "n_planned": len(seeds),
        "created_at": manifest.get("created_at", ""),
        "n_episodes": len(rows),
        "n_pending": pending,
        "publishable": bool(
            seeds
            and pending == 0
            and counts["error"] == 0
            and counts["cancelled"] == 0
            and n_ok == len(rows)
        ),
        **{f"n_{k}": v for k, v in counts.items()},
        **{f"{s}_n": hits[s] for s in STAGES},
        **{f"{s}_pct": round(hits[s] / n_ok * 100, 1) if n_ok else 0.0 for s in STAGES},
        "n_ok": n_ok,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default=str(EVAL_ROOT / "gr00t_wr03_stage_summary.csv"),
        help="输出 CSV 路径（默认 outputs/eval/gr00t_wr03_stage_summary.csv）",
    )
    parser.add_argument(
        "--episodes-output",
        default=str(EVAL_ROOT / "gr00t_wr03_episode_results.csv"),
        help="逐 seed 输出 CSV（默认 outputs/eval/gr00t_wr03_episode_results.csv）",
    )
    parser.add_argument(
        "--include-legacy-runs",
        action="store_true",
        help="显式包含旧版和未标注 evaluator 的运行；正式汇总不要使用",
    )
    args = parser.parse_args()

    fields = [
        "run_id", "checkpoint_step", "evaluator_profile", "replan_steps", "max_steps",
        "seeds_start", "seeds_end", "n_planned", "n_episodes", "n_pending",
        "publishable",
        "n_success", "n_failure", "n_error", "n_cancelled",
    ]
    for s in STAGES:
        fields += [f"{s}_n", f"{s}_pct"]
    fields += ["created_at"]
    episode_fields = [
        "run_id", "checkpoint_step", "evaluator_profile", "seed", "status",
        "success", "failure_reason", "env_steps",
        "stored_level", "storage_hidden", "opened_level", "retrieved_item_id",
        *[f"stage_{stage}" for stage in STAGES],
    ]

    runs = collect_runs(include_legacy=args.include_legacy_runs)
    summary = [summarize_run(run) for run in runs]
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(summary)

    episode_rows = []
    for run in runs:
        with open(run / "run_manifest.json", encoding="utf-8") as f:
            manifest = json.load(f)
        step = checkpoint_step(run, manifest)
        for result in load_episodes(run):
            row = episode_row(run, result)
            row["checkpoint_step"] = step
            episode_rows.append(row)
    episodes_out = Path(args.episodes_output)
    episodes_out.parent.mkdir(parents=True, exist_ok=True)
    with open(episodes_out, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=episode_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(episode_rows)

    if not runs:
        print(
            f"未找到 {OFFICIAL_EVALUATOR_PROFILE} 运行；已清空旧版正式报表: "
            f"{EVAL_ROOT / 'gr00t' / 'WR03'}",
            file=sys.stderr,
        )
        return 1

    # 终端表格（含中文阶段标签）
    print(f"\nWR-03 分阶段达成率（error/cancelled 不进分母）\n")
    print(f"{'run':<44} {'ckpt':>6} {'n_ok':>4}  "
          + "  ".join(f"{STAGE_LABELS[s]:>6}" for s in STAGES))
    for row in summary:
        cells = "  ".join(f"{row[f'{s}_pct']:>6.1f}" for s in STAGES)
        print(f"{row['run_id'][:44]:<44} {row['checkpoint_step']:>6} "
              f"{row['n_ok']:>4}  {cells}")
    print(f"\n已写入 {len(summary)} 个正式运行: {out}")
    print(f"已写入 {len(episode_rows)} 个逐 seed 判定: {episodes_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
