#!/usr/bin/env bash
# One-shot launcher submitted to at(1) for 2026-08-29 22:30 UTC.

set -euo pipefail

REPO_ROOT="/home/chensihang/membench"
LAUNCH_LOG="$REPO_ROOT/outputs/gr00t_n1_7_wa02_v061_launch.log"

cd "$REPO_ROOT"
source /home/chensihang/miniconda3/etc/profile.d/conda.sh
conda activate membench

echo "[$(date -u '+%F %T UTC')] Scheduled WA02 launch started." | tee -a "$LAUNCH_LOG"
exec bash scripts/train/train_gr00t_n1_7_wa02_v061.sh >>"$LAUNCH_LOG" 2>&1
