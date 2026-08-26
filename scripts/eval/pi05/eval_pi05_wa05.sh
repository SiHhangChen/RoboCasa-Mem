#!/usr/bin/env bash
# Evaluate one pi0.5 LoRA checkpoint on the WA-05 smoke suite.
#
#   GPU=2 PORT=55005 ./scripts/eval/pi05/eval_pi05_wa05.sh serve
#   PORT=55005 ./scripts/eval/pi05/eval_pi05_wa05.sh preflight
#   PORT=55005 ./scripts/eval/pi05/eval_pi05_wa05.sh eval
#   PORT=55005 ./scripts/eval/pi05/eval_pi05_wa05.sh summary
#   PORT=55005 ./scripts/eval/pi05/eval_pi05_wa05.sh stop
#   GPU=2 PORT=55005 ./scripts/eval/pi05/eval_pi05_wa05.sh full

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"

CKPT_DIR="${CKPT_DIR:-/data/shared_workspace/chensihang/ckpts/membench/gr00t/wa05/55000}"
CONFIG_NAME="${CONFIG_NAME:-pi05_membench_wa05_lora}"
PORT="${PORT:-55005}"
GPU="${GPU:-2}"
SUITE="${SUITE:-configs/eval/suites/wa05_v1.yaml}"
CKPT_STEP="$(basename "${CKPT_DIR}")"
RUN_ID="${RUN_ID:-wa05-pi05-lora-${CKPT_STEP}-smoke}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/eval}"
SERVER_HOST="${SERVER_HOST:-127.0.0.1}"
SERVER_INFERENCE_TIMEOUT_S="${SERVER_INFERENCE_TIMEOUT_S:-300}"
RETRY_ERRORS="${RETRY_ERRORS:-0}"
LOG_DIR="${OUTPUT_ROOT}/logs"

OPENPI_PYTHON="${ROOT}/policy/openpi/.venv/bin/python"
ROBOCASA_PYTHON="${ROOT}/.venv-robocasa/bin/python"
MPLIB_PYTHON="${ROOT}/.venv-mplib/bin/python"
MANISKILL_PANDA_DIR="${ROOT}/.venv-robocasa/lib/python3.11/site-packages/mani_skill/assets/robots/panda"
PID_FILE="${LOG_DIR}/serve_wa05_${PORT}.pid"
LOG_FILE="${LOG_DIR}/serve_wa05_${PORT}.log"

usage() {
    sed -n '2,10p' "${BASH_SOURCE[0]}" | sed -n 's/^# \{0,1\}//p'
}

validate_common() {
    [[ -x "${OPENPI_PYTHON}" ]] || { echo "OpenPI environment is missing: ${OPENPI_PYTHON}" >&2; exit 1; }
    [[ -x "${ROBOCASA_PYTHON}" ]] || { echo "RoboCasa environment is missing: ${ROBOCASA_PYTHON}" >&2; exit 1; }
    [[ -f "${ROOT}/${SUITE}" || -f "${SUITE}" ]] || { echo "Evaluation suite is missing: ${SUITE}" >&2; exit 1; }
}

start_server() {
    validate_common
    [[ -d "${CKPT_DIR}/params" ]] || { echo "Checkpoint params are missing: ${CKPT_DIR}/params" >&2; exit 1; }
    [[ -f "${CKPT_DIR}/assets/wa05_200seeds_v061/norm_stats.json" ]] || {
        echo "Checkpoint normalization stats are missing: ${CKPT_DIR}" >&2
        exit 1
    }
    mkdir -p "${LOG_DIR}"
    if [[ -f "${PID_FILE}" ]] && kill -0 "$(<"${PID_FILE}")" 2>/dev/null; then
        echo "Server ${PORT} is already running (pid $(<"${PID_FILE}"))"
        return 0
    fi
    cd "${ROOT}"
    export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.9}"
    CUDA_VISIBLE_DEVICES="${GPU}" \
        nohup "${OPENPI_PYTHON}" scripts/serve/serve_openpi.py \
            --config "${CONFIG_NAME}" \
            --checkpoint "${CKPT_DIR}" \
            --image-size 224 \
            --state-schema membench.robot_state37.v2 \
            --inference-timeout-s "${SERVER_INFERENCE_TIMEOUT_S}" \
            --host 0.0.0.0 \
            --port "${PORT}" \
            >"${LOG_FILE}" 2>&1 &
    echo $! >"${PID_FILE}"
    echo "Started server ${PORT} (pid $!, GPU ${GPU}, checkpoint ${CKPT_STEP})"
    echo "Log: ${LOG_FILE}"
}

stop_server() {
    if [[ -f "${PID_FILE}" ]] && kill -0 "$(<"${PID_FILE}")" 2>/dev/null; then
        kill "$(<"${PID_FILE}")"
        rm -f "${PID_FILE}"
        echo "Stopped server ${PORT}"
    else
        rm -f "${PID_FILE}" 2>/dev/null || true
        echo "Server ${PORT} is not running"
    fi
}

run_preflight() {
    validate_common
    cd "${ROOT}"
    export NO_PROXY="${NO_PROXY:+${NO_PROXY},}127.0.0.1,localhost"
    export no_proxy="${no_proxy:+${no_proxy},}127.0.0.1,localhost"
    "${ROBOCASA_PYTHON}" scripts/eval/run_remote.py \
        --endpoint "ws://${SERVER_HOST}:${PORT}" \
        --suite "${SUITE}" \
        --preflight-only
}

run_eval() {
    local retry_args=()
    validate_common
    [[ -x "${MPLIB_PYTHON}" ]] || { echo "MPlib environment is missing: ${MPLIB_PYTHON}" >&2; exit 1; }
    [[ -d "${MANISKILL_PANDA_DIR}" ]] || { echo "ManiSkill Panda assets are missing: ${MANISKILL_PANDA_DIR}" >&2; exit 1; }
    [[ "${RETRY_ERRORS}" == "1" ]] && retry_args=(--retry-errors)
    cd "${ROOT}"
    export PYTHONPATH=.
    export MUJOCO_GL="${MUJOCO_GL:-egl}"
    export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
    export MEMBENCH_MPLIB_PYTHON="${MEMBENCH_MPLIB_PYTHON:-${MPLIB_PYTHON}}"
    export MEMBENCH_MANISKILL_PANDA_DIR="${MEMBENCH_MANISKILL_PANDA_DIR:-${MANISKILL_PANDA_DIR}}"
    export NO_PROXY="${NO_PROXY:+${NO_PROXY},}127.0.0.1,localhost"
    export no_proxy="${no_proxy:+${no_proxy},}127.0.0.1,localhost"
    "${ROBOCASA_PYTHON}" scripts/eval/run_remote.py \
        --endpoint "ws://${SERVER_HOST}:${PORT}" \
        --suite "${SUITE}" \
        --output-root "${OUTPUT_ROOT}" \
        --run-id "${RUN_ID}" \
        "${retry_args[@]}"
}

run_summary() {
    validate_common
    cd "${ROOT}"
    "${ROBOCASA_PYTHON}" scripts/eval/summarize_run.py "${OUTPUT_ROOT}/${RUN_ID}"
}

status() {
    if [[ -f "${PID_FILE}" ]] && kill -0 "$(<"${PID_FILE}")" 2>/dev/null; then
        echo "Server ${PORT}: running (pid $(<"${PID_FILE}"), checkpoint ${CKPT_STEP})"
    else
        echo "Server ${PORT}: not running"
    fi
}

run_full() {
    local attempt
    start_server
    trap stop_server EXIT
    for attempt in {1..10}; do
        if run_preflight; then
            run_eval
            return
        fi
        if [[ ! -f "${PID_FILE}" ]] || ! kill -0 "$(<"${PID_FILE}")" 2>/dev/null; then
            echo "Policy server exited before preflight completed; see ${LOG_FILE}" >&2
            return 1
        fi
        echo "Policy server is still loading (preflight attempt ${attempt}/10)"
        sleep 5
    done
    echo "Policy server did not become ready; see ${LOG_FILE}" >&2
    return 1
}

case "${1:-}" in
    serve) start_server ;;
    preflight) run_preflight ;;
    eval) run_eval ;;
    summary) run_summary ;;
    stop) stop_server ;;
    status) status ;;
    full) run_full ;;
    help|-h|--help) usage ;;
    *)
        echo "Unknown command: ${1:-}" >&2
        usage >&2
        exit 2
        ;;
esac
