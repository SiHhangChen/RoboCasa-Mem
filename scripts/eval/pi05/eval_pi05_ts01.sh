#!/usr/bin/env bash
#
# Evaluate a pi0.5 LoRA checkpoint on the canonical TS01 suite.
#
# Usage:
#   GPU=0 PORT=55030 ./scripts/eval/pi05/eval_pi05_ts01.sh serve
#   PORT=55030 ./scripts/eval/pi05/eval_pi05_ts01.sh preflight
#   PORT=55030 ./scripts/eval/pi05/eval_pi05_ts01.sh eval
#   PORT=55030 ./scripts/eval/pi05/eval_pi05_ts01.sh summary
#   PORT=55030 ./scripts/eval/pi05/eval_pi05_ts01.sh status
#   PORT=55030 ./scripts/eval/pi05/eval_pi05_ts01.sh stop
#
# Environment overrides:
#   CKPT_DIR      checkpoint directory (default: ckpts/pi05/ts01/50000)
#   CONFIG_NAME   OpenPI training config (default: pi05_membench_ts01_lora)
#   PORT          policy server port (default: 55030)
#   GPU           physical GPU visible to the policy server (default: 0)
#   RENDER_GPU    physical GPU used by MuJoCo EGL in the rollout client (default: GPU)
#   REPLAN_STEPS  policy replan interval; supported presets: 20, 50 (default: 50)
#   SUITE         evaluation suite (default: selected by REPLAN_STEPS)
#   RUN_ID        output run ID (default: ts01-pi05-{checkpoint_step}-50seed-r{REPLAN_STEPS})
#   OUTPUT_ROOT   evaluation output root (default: outputs/eval)
#   SERVER_HOST   policy server host used by the client (default: 127.0.0.1)
#   SERVER_INFERENCE_TIMEOUT_S server-side inference timeout (default: 300)
#   XLA_PYTHON_CLIENT_MEM_FRACTION OpenPI server GPU memory fraction (default: 0.75)
#   RETRY_ERRORS  set to 1 to rerun episodes whose previous status is error

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
CKPT_ROOT="${ROOT}/ckpts/pi05/ts01"

CKPT_DIR="${CKPT_DIR:-${CKPT_ROOT}/50000}"
CONFIG_NAME="${CONFIG_NAME:-pi05_membench_ts01_lora}"
DATASET_REPO_ID="ts01_200seeds_v061"
PORT="${PORT:-55030}"
GPU="${GPU:-0}"
RENDER_GPU="${RENDER_GPU:-${GPU}}"
REPLAN_STEPS="${REPLAN_STEPS:-50}"
case "${REPLAN_STEPS}" in
    20) DEFAULT_SUITE="configs/eval/legacy_suites/ts01_r20_v1.yaml" ;;
    50) DEFAULT_SUITE="configs/eval/suites/ts01_v1.yaml" ;;
    *)
        echo "Unsupported REPLAN_STEPS=${REPLAN_STEPS}; expected 20 or 50" >&2
        exit 2
        ;;
esac
SUITE="${SUITE:-${DEFAULT_SUITE}}"
CKPT_STEP="$(basename "${CKPT_DIR}")"
RUN_ID="${RUN_ID:-ts01-pi05-${CKPT_STEP}-50seed-r${REPLAN_STEPS}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/eval}"
SERVER_HOST="${SERVER_HOST:-127.0.0.1}"
SERVER_INFERENCE_TIMEOUT_S="${SERVER_INFERENCE_TIMEOUT_S:-300}"
SERVER_STARTUP_TIMEOUT_S="${SERVER_STARTUP_TIMEOUT_S:-900}"
XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.75}"
RETRY_ERRORS="${RETRY_ERRORS:-0}"
LOG_DIR="${OUTPUT_ROOT}/logs"

OPENPI_PYTHON="${ROOT}/policy/openpi/.venv/bin/python"
ROBOCASA_PYTHON="${ROOT}/.venv-robocasa/bin/python"
MPLIB_PYTHON="${ROOT}/.venv-mplib/bin/python"
MANISKILL_PANDA_DIR="${ROOT}/.venv-robocasa/lib/python3.11/site-packages/mani_skill/assets/robots/panda"
PID_FILE="${LOG_DIR}/serve_ts01_${PORT}.pid"
LOG_FILE="${LOG_DIR}/serve_ts01_${PORT}.log"

usage() {
    sed -n '2,28p' "${BASH_SOURCE[0]}" | sed -n 's/^# \{0,1\}//p'
}

validate_common() {
    [[ -x "${OPENPI_PYTHON}" ]] || {
        echo "OpenPI environment is missing: ${OPENPI_PYTHON}" >&2
        exit 1
    }
    [[ -x "${ROBOCASA_PYTHON}" ]] || {
        echo "RoboCasa environment is missing: ${ROBOCASA_PYTHON}" >&2
        exit 1
    }
    [[ -f "${ROOT}/${SUITE}" || -f "${SUITE}" ]] || {
        echo "Evaluation suite is missing: ${SUITE}" >&2
        exit 1
    }
}

validate_checkpoint() {
    [[ -d "${CKPT_DIR}/params" ]] || {
        echo "Checkpoint params are missing: ${CKPT_DIR}/params" >&2
        exit 1
    }
    [[ -f "${CKPT_DIR}/assets/ts01_200seeds_v061/norm_stats.json" ]] || {
        echo "Checkpoint normalization stats are missing: ${CKPT_DIR}/assets/ts01_200seeds_v061/norm_stats.json" >&2
        exit 1
    }
}

start_server() {
    validate_common
    validate_checkpoint
    mkdir -p "${LOG_DIR}"
    if [[ -f "${PID_FILE}" ]] && kill -0 "$(<"${PID_FILE}")" 2>/dev/null; then
        echo "Server ${PORT} is already running (pid $(<"${PID_FILE}"))"
        return 0
    fi

    cd "${ROOT}"
    # The OpenPI config supports DATASET_REPO_ID overrides for other tasks. Do
    # not inherit a stale TS02/TS03 value from the calling shell for TS01.
    export DATASET_REPO_ID
    export XLA_PYTHON_CLIENT_MEM_FRACTION
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
    local server_pid=$!
    echo "${server_pid}" >"${PID_FILE}"
    wait_for_server "${server_pid}"
    echo "Started TS01 policy server ${PORT} (pid ${server_pid}, GPU ${GPU}, checkpoint ${CKPT_STEP}, XLA memory fraction ${XLA_PYTHON_CLIENT_MEM_FRACTION})"
    echo "Log: ${LOG_FILE}"
}

wait_for_server() {
    local server_pid="$1"
    local elapsed=0
    while (( elapsed < SERVER_STARTUP_TIMEOUT_S )); do
        if ! kill -0 "${server_pid}" 2>/dev/null; then
            echo "TS01 policy server exited during startup; inspect ${LOG_FILE}" >&2
            rm -f "${PID_FILE}"
            tail -n 35 "${LOG_FILE}" >&2 || true
            exit 1
        fi
        if grep -q "server listening on .*:${PORT}" "${LOG_FILE}" 2>/dev/null; then
            return 0
        fi
        if (( elapsed == 0 || elapsed % 10 == 0 )); then
            echo "Waiting for TS01 policy server ${PORT} to load (${elapsed}s/${SERVER_STARTUP_TIMEOUT_S}s)..."
        fi
        sleep 1
        ((elapsed += 1))
    done
    echo "TS01 policy server did not start within ${SERVER_STARTUP_TIMEOUT_S}s; inspect ${LOG_FILE}" >&2
    kill "${server_pid}" 2>/dev/null || true
    rm -f "${PID_FILE}"
    tail -n 35 "${LOG_FILE}" >&2 || true
    exit 1
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
    [[ -x "${MPLIB_PYTHON}" ]] || {
        echo "MPlib environment is missing: ${MPLIB_PYTHON}" >&2
        exit 1
    }
    [[ -d "${MANISKILL_PANDA_DIR}" ]] || {
        echo "ManiSkill Panda assets are missing: ${MANISKILL_PANDA_DIR}" >&2
        exit 1
    }
    [[ "${RETRY_ERRORS}" == "1" ]] && retry_args=(--retry-errors)
    cd "${ROOT}"
    export PYTHONPATH=.
    export MUJOCO_GL="${MUJOCO_GL:-egl}"
    export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
    export MUJOCO_EGL_DEVICE_ID="${RENDER_GPU}"
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

case "${1:-}" in
    serve) start_server ;;
    preflight) run_preflight ;;
    eval) run_eval ;;
    summary) run_summary ;;
    stop) stop_server ;;
    status) status ;;
    help|-h|--help) usage ;;
    *)
        echo "Unknown command: ${1:-}" >&2
        usage >&2
        exit 2
        ;;
esac
