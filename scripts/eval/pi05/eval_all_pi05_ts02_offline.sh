#!/usr/bin/env bash
# Validate every TS02 pi0.5 LoRA checkpoint with fixed offline-loss settings.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-${ROOT}/policy/openpi/checkpoints/pi05_membench_ts02_lora/pi05_lora_membench_ts02}"
DATASET_ROOT="${DATASET_ROOT:-${ROOT}/data/val/ts02_val_5seeds_v061}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT}/outputs/eval/offline}"
GPU_IDS="${GPU_IDS:-2,3}"
BATCH_SIZE="${BATCH_SIZE:-8}"
CONFIG_NAME="${CONFIG_NAME:-pi05_membench_ts02_lora}"
OPENPI_PYTHON="${ROOT}/policy/openpi/.venv/bin/python"
LOG_DIR="${OUTPUT_DIR}/logs"
LOSS_VALUES_DIR="${OUTPUT_DIR}/loss_values"

IFS=',' read -r -a GPUS <<< "${GPU_IDS}"
[[ ${#GPUS[@]} -gt 0 ]] || { echo "GPU_IDS is empty" >&2; exit 2; }
mapfile -t STEPS < <(
    find "${CHECKPOINT_ROOT}" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' \
        | awk '/^[0-9]+$/' \
        | sort -n
)
[[ ${#STEPS[@]} -gt 0 ]] || { echo "No checkpoints found: ${CHECKPOINT_ROOT}" >&2; exit 2; }
mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}" "${LOSS_VALUES_DIR}"

is_complete() {
    local result="$1"
    local loss_values="$2"
    [[ -s "${result}" ]] || return 1
    [[ -s "${loss_values}" ]] || return 1
    "${OPENPI_PYTHON}" -c \
        'import json,numpy as np,sys; p=json.load(open(sys.argv[1])); f=p.get("flow_matching_loss", {}); o=f.get("overall", {}); a=np.load(sys.argv[2])["loss"]; needed=("median","p90","p95","p99","variance"); sys.exit(0 if p.get("status") == "complete" and f.get("frames") == 6391 and o.get("count") == 319550 and all(k in o for k in needed) and a.shape == (319550,) else 1)' \
        "${result}" "${loss_values}"
}

run_worker() {
    local slot="$1"
    local gpu="${GPUS[slot]}"
    local index step checkpoint result loss_values log
    for ((index=slot; index<${#STEPS[@]}; index+=${#GPUS[@]})); do
        step="${STEPS[index]}"
        checkpoint="${CHECKPOINT_ROOT}/${step}"
        result="${OUTPUT_DIR}/ts02-pi05-lora-${step}-val5.json"
        loss_values="${LOSS_VALUES_DIR}/ts02-pi05-lora-${step}-val5-losses.npz"
        log="${LOG_DIR}/ts02-pi05-lora-${step}-val5.log"
        if is_complete "${result}" "${loss_values}"; then
            echo "[GPU ${gpu}] step ${step}: complete, skipping"
            continue
        fi
        echo "[GPU ${gpu}] step ${step}: starting"
        CUDA_VISIBLE_DEVICES="${gpu}" \
        XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.9}" \
        "${OPENPI_PYTHON}" "${ROOT}/scripts/eval/pi05/eval_openpi_checkpoint.py" \
            --config "${CONFIG_NAME}" \
            --checkpoint "${checkpoint}" \
            --dataset-root "${DATASET_ROOT}" \
            --output "${result}" \
            --loss-values-output "${loss_values}" \
            --batch-size "${BATCH_SIZE}" \
            --skip-action-metrics \
            > "${log}" 2>&1
        echo "[GPU ${gpu}] step ${step}: complete"
        tail -n 2 "${log}"
    done
}

pids=()
for slot in "${!GPUS[@]}"; do
    run_worker "${slot}" &
    pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
        failed=1
    fi
done
[[ "${failed}" == 0 ]] || { echo "One or more validation workers failed" >&2; exit 1; }

"${OPENPI_PYTHON}" "${ROOT}/scripts/eval/pi05/rank_pi05_ts02_offline_checkpoints.py" \
    --input-dir "${OUTPUT_DIR}"
