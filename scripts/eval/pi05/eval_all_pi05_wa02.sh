#!/usr/bin/env bash
# Run the WA02 checkpoint set sequentially on one GPU.
#
# Usage:
#   GPU=7 PORT=55040 ./scripts/eval/pi05/eval_all_pi05_wa02.sh
#
# Environment overrides:
#   CKPT_STEPS    space-separated checkpoint steps (default: 45000 50000 55000 59999)
#   GPU           physical GPU used by the server (default: 7)
#   RENDER_GPU    physical GPU used by MuJoCo EGL (default: GPU)
#   PORT          reused policy server port (default: 55040)
#   OUTPUT_ROOT   evaluation output root (default: outputs/eval)
#   RETRY_ERRORS  rerun existing error episodes (default: 1)

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
CKPT_ROOT="${ROOT}/ckpts/pi05/wa02"
LAUNCHER="${ROOT}/scripts/eval/pi05/eval_pi05_wa02.sh"

CKPT_STEPS="${CKPT_STEPS:-45000 50000 55000 59999}"
GPU="${GPU:-7}"
RENDER_GPU="${RENDER_GPU:-${GPU}}"
PORT="${PORT:-55040}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/eval}"
RETRY_ERRORS="${RETRY_ERRORS:-1}"

for step in ${CKPT_STEPS}; do
    ckpt_dir="${CKPT_ROOT}/${step}"
    run_id="wa02-pi05-${step}-50seed-r50-actualbinding-v2"
    echo "[$(date '+%F %T')] Starting WA02 checkpoint ${step} on GPU ${GPU}, port ${PORT}"
    CKPT_DIR="${ckpt_dir}" \
    GPU="${GPU}" \
    RENDER_GPU="${RENDER_GPU}" \
    PORT="${PORT}" \
    RUN_ID="${run_id}" \
    OUTPUT_ROOT="${OUTPUT_ROOT}" \
    RETRY_ERRORS="${RETRY_ERRORS}" \
        "${LAUNCHER}" full
    echo "[$(date '+%F %T')] Finished WA02 checkpoint ${step}: ${OUTPUT_ROOT}/${run_id}"
done

echo "[$(date '+%F %T')] Completed all WA02 checkpoints: ${CKPT_STEPS}"
