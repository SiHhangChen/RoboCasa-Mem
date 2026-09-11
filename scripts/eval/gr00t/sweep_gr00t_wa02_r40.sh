#!/usr/bin/env bash
# Evaluate the WA-02 GR00T checkpoints concurrently, one GPU per checkpoint.
#
#   ./scripts/eval/gr00t/sweep_gr00t_wa02_r40.sh
#   STEPS="090000 100000" GPUS="2 3" ./scripts/eval/gr00t/sweep_gr00t_wa02_r40.sh

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
LAUNCHER="${ROOT}/scripts/eval/gr00t/eval_gr00t_wa02.sh"

STEPS="${STEPS:-080000 090000 100000}"
GPUS="${GPUS:-2 3 4}"
RENDER_GPUS="${RENDER_GPUS:-${GPUS}}"
BASE_PORT="${BASE_PORT:-57120}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/eval}"
RETRY_ERRORS="${RETRY_ERRORS:-1}"
LOG_DIR="${ROOT}/${OUTPUT_ROOT}/logs"

read -r -a STEP_ARR <<<"${STEPS}"
read -r -a GPU_ARR <<<"${GPUS}"
read -r -a RENDER_GPU_ARR <<<"${RENDER_GPUS}"

if (( ${#STEP_ARR[@]} > ${#GPU_ARR[@]} )); then
    echo "Need at least one policy GPU per checkpoint" >&2
    exit 1
fi
if (( ${#STEP_ARR[@]} > ${#RENDER_GPU_ARR[@]} )); then
    echo "Need at least one render GPU per checkpoint" >&2
    exit 1
fi

mkdir -p "${LOG_DIR}"

PORTS=()
for i in "${!STEP_ARR[@]}"; do
    PORTS+=( $((BASE_PORT + i)) )
done

cleanup() {
    echo
    echo "Stopping GR00T servers..."
    for i in "${!STEP_ARR[@]}"; do
        CKPT_STEP="${STEP_ARR[i]}" PORT="${PORTS[i]}" OUTPUT_ROOT="${OUTPUT_ROOT}" \
            "${LAUNCHER}" stop || true
    done
}
trap cleanup EXIT

echo "Clearing target ports..."
for i in "${!STEP_ARR[@]}"; do
    CKPT_STEP="${STEP_ARR[i]}" PORT="${PORTS[i]}" OUTPUT_ROOT="${OUTPUT_ROOT}" \
        "${LAUNCHER}" stop || true
done
sleep 2

for port in "${PORTS[@]}"; do
    if ss -ltn "sport = :${port}" | tail -n +2 | grep -q .; then
        echo "Port ${port} is already in use" >&2
        exit 1
    fi
done

echo
echo "Starting ${#STEP_ARR[@]} GR00T servers..."
for i in "${!STEP_ARR[@]}"; do
    CKPT_STEP="${STEP_ARR[i]}" GPU="${GPU_ARR[i]}" PORT="${PORTS[i]}" \
        OUTPUT_ROOT="${OUTPUT_ROOT}" "${LAUNCHER}" start
done

echo
echo "Waiting for servers to load..."
for i in "${!STEP_ARR[@]}"; do
    CKPT_STEP="${STEP_ARR[i]}" PORT="${PORTS[i]}" OUTPUT_ROOT="${OUTPUT_ROOT}" \
        "${LAUNCHER}" wait
done

echo
echo "Preflighting replan_steps=40..."
for i in "${!STEP_ARR[@]}"; do
    CKPT_STEP="${STEP_ARR[i]}" PORT="${PORTS[i]}" OUTPUT_ROOT="${OUTPUT_ROOT}" \
        "${LAUNCHER}" preflight
done

echo
echo "Launching ${#STEP_ARR[@]} evaluations in parallel..."
PIDS=()
for i in "${!STEP_ARR[@]}"; do
    step="${STEP_ARR[i]}"
    log="${LOG_DIR}/sweep_wa02_${step}_r40.log"
    CKPT_STEP="${step}" PORT="${PORTS[i]}" RENDER_GPU="${RENDER_GPU_ARR[i]}" \
        OUTPUT_ROOT="${OUTPUT_ROOT}" RETRY_ERRORS="${RETRY_ERRORS}" \
        nohup "${LAUNCHER}" eval >"${log}" 2>&1 &
    PIDS+=($!)
    echo "  ${step}: policy GPU ${GPU_ARR[i]}, render GPU ${RENDER_GPU_ARR[i]}, port ${PORTS[i]}, pid $!, log ${log}"
done

echo
echo "Waiting for all evaluations to finish..."
STATUS=0
for i in "${!PIDS[@]}"; do
    if wait "${PIDS[i]}"; then
        echo "  ${STEP_ARR[i]}: done"
    else
        echo "  ${STEP_ARR[i]}: FAILED (see ${LOG_DIR}/sweep_wa02_${STEP_ARR[i]}_r40.log)" >&2
        STATUS=1
    fi
done

echo
echo "Summaries:"
for i in "${!STEP_ARR[@]}"; do
    run_dir="${OUTPUT_ROOT}/wa02-gr00t-${STEP_ARR[i]}-50seeds-r40"
    if [[ -f "${ROOT}/${run_dir}/summary.json" ]]; then
        "${ROOT}/.venv-robocasa/bin/python" - "${ROOT}/${run_dir}" <<'PY'
import json
import sys
from pathlib import Path

run = Path(sys.argv[1])
summary = json.loads((run / "summary.json").read_text())
print(f"  {run.name}: success_rate={summary.get('success_rate')} counts={summary.get('counts')}")
PY
    else
        echo "  ${run_dir}: no summary.json"
    fi
done

exit "${STATUS}"
