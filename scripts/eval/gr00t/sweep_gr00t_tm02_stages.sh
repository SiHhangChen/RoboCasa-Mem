#!/usr/bin/env bash
# Run the formal TM02 stage campaign for all four GR00T checkpoints.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
LAUNCHER="${ROOT}/scripts/eval/gr00t/eval_gr00t_tm02.sh"

STEPS="${STEPS:-070000 080000 090000 100000}"
REPLICAS="${REPLICAS:-3}"
GPU_SLOTS="${GPU_SLOTS:-1 1 2 2 3 3 4 4 5 5 6 6}"
BASE_PORT="${BASE_PORT:-57320}"
STAGE_SUITE="${STAGE_SUITE:-configs/eval/stages/tm02_gr00t_r40_formal_v1.yaml}"
LOG_DIR="${ROOT}/outputs/eval/logs"

read -r -a STEP_ARR <<<"${STEPS}"
read -r -a GPU_SLOT_ARR <<<"${GPU_SLOTS}"

worker_count=$((${#STEP_ARR[@]} * REPLICAS))
if (( worker_count > ${#GPU_SLOT_ARR[@]} )); then
    echo "Need ${worker_count} GPU slots, got ${#GPU_SLOT_ARR[@]}" >&2
    exit 1
fi

WORKER_STEPS=()
WORKER_SHARDS=()
for step in "${STEP_ARR[@]}"; do
    for ((shard = 0; shard < REPLICAS; shard++)); do
        WORKER_STEPS+=("${step}")
        WORKER_SHARDS+=("${shard}")
    done
done

mkdir -p "${LOG_DIR}"
PIDS=()
SERVERS=()

cleanup() {
    for pid in "${SERVERS[@]}"; do
        kill "${pid}" 2>/dev/null || true
    done
}
trap cleanup EXIT INT TERM

for i in "${!WORKER_STEPS[@]}"; do
    step="${WORKER_STEPS[i]}"
    shard="${WORKER_SHARDS[i]}"
    gpu="${GPU_SLOT_ARR[i]}"
    port="$((BASE_PORT + i))"
    serve_log="${LOG_DIR}/serve_gr00t_tm02_${step}_stage_shard${shard}_${port}.log"
    CKPT_STEP="${step}" GPU="${gpu}" PORT="${port}" \
        "${LAUNCHER}" serve-fg >"${serve_log}" 2>&1 &
    SERVERS+=("$!")
done

for i in "${!WORKER_STEPS[@]}"; do
    step="${WORKER_STEPS[i]}"
    shard="${WORKER_SHARDS[i]}"
    port="$((BASE_PORT + i))"
    serve_log="${LOG_DIR}/serve_gr00t_tm02_${step}_stage_shard${shard}_${port}.log"
    until rg -q "Serving gr00t-groot|server listening" "${serve_log}"; do
        if ! kill -0 "${SERVERS[i]}" 2>/dev/null; then
            wait "${SERVERS[i]}"
            exit 1
        fi
        sleep 2
    done
done

for i in "${!WORKER_STEPS[@]}"; do
    step="${WORKER_STEPS[i]}"
    shard="${WORKER_SHARDS[i]}"
    gpu="${GPU_SLOT_ARR[i]}"
    port="$((BASE_PORT + i))"
    eval_log="${LOG_DIR}/eval_tm02_gr00t_${step}_stage_r40_formal_v1_shard${shard}.log"
    CKPT_STEP="${step}" GPU="${gpu}" RENDER_GPU="${gpu}" PORT="${port}" \
        STAGE_SUITE="${STAGE_SUITE}" STAGE_SEED_SHARD_INDEX="${shard}" \
        STAGE_SEED_SHARD_COUNT="${REPLICAS}" \
        "${LAUNCHER}" stage-eval >"${eval_log}" 2>&1 &
    PIDS+=("$!")
done

status=0
for i in "${!PIDS[@]}"; do
    if ! wait "${PIDS[i]}"; then
        echo "${WORKER_STEPS[i]} shard ${WORKER_SHARDS[i]} stage evaluation failed" >&2
        status=1
    fi
done

for step in "${STEP_ARR[@]}"; do
    CKPT_STEP="${step}" STAGE_SUITE="${STAGE_SUITE}" \
        "${LAUNCHER}" stage-summary
done
exit "${status}"
