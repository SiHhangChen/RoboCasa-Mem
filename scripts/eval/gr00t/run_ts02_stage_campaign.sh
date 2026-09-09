#!/usr/bin/env bash
# Run one TS02 GR00T stage campaign with a managed policy server.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
CKPT_STEP="${CKPT_STEP:?CKPT_STEP is required}"
PORT="${PORT:?PORT is required}"
LOG_DIR="${ROOT}/outputs/eval/logs"
SERVE_LOG="${LOG_DIR}/serve_ts02_gr00t_${CKPT_STEP}_stage_${PORT}.log"
EVAL_LOG="${LOG_DIR}/eval_ts02_gr00t_${CKPT_STEP}_stage_r40_formal_v1.log"

mkdir -p "${LOG_DIR}"
cd "${ROOT}"

server_pid=""
cleanup() {
    if [[ -n "${server_pid}" ]] && kill -0 "${server_pid}" 2>/dev/null; then
        kill "${server_pid}" 2>/dev/null || true
        wait "${server_pid}" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

./scripts/eval/gr00t/eval_gr00t_ts02.sh serve-fg >"${SERVE_LOG}" 2>&1 &
server_pid=$!

until rg -q "Serving gr00t-groot" "${SERVE_LOG}"; do
    if ! kill -0 "${server_pid}" 2>/dev/null; then
        wait "${server_pid}"
        exit 1
    fi
    sleep 2
done

./scripts/eval/gr00t/eval_gr00t_ts02.sh stage-eval >"${EVAL_LOG}" 2>&1
