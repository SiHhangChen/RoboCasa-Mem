#!/usr/bin/env bash
#
# Evaluate one pi0.5 LoRA checkpoint on the 50-seed WX-01 validation suite.
# Run each checkpoint on its own GPU and port by overriding CKPT_DIR, GPU, PORT,
# and RUN_ID. Completed seeds are skipped when the same RUN_ID is resumed.
#
# Examples:
#   GPU=0 PORT=56000 ./scripts/eval/pi05/eval_pi05_wx01.sh serve
#   GPU=0 PORT=56000 ./scripts/eval/pi05/eval_pi05_wx01.sh serve-fg
#   PORT=56000 ./scripts/eval/pi05/eval_pi05_wx01.sh preflight
#   PORT=56000 ./scripts/eval/pi05/eval_pi05_wx01.sh eval
#   PORT=56000 ./scripts/eval/pi05/eval_pi05_wx01.sh summary
#   PORT=56000 ./scripts/eval/pi05/eval_pi05_wx01.sh stop
#
# Environment overrides:
#   CKPT_DIR      checkpoint directory (default: WX-01 step 59999)
#   CONFIG_NAME   OpenPI training config (default: pi05_membench_wx01_lora)
#   PORT          policy server port (default: 56000)
#   GPU           physical GPU visible to the policy server (default: 0)
#   SUITE         evaluation suite (default: configs/eval/suites/wx01_v1.yaml)
#   RUN_ID        output run ID (default: wx01-pi05-{checkpoint_step}-50seed-r50)
#   OUTPUT_ROOT   evaluation output root (default: outputs/eval)
#   SERVER_HOST   policy server host used by the client (default: 127.0.0.1)
#   SERVER_INFERENCE_TIMEOUT_S server-side inference timeout (default: 300)
#   RETRY_ERRORS  set to 1 to rerun episodes whose previous status is error

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
CKPT_ROOT="${ROOT}/policy/openpi/checkpoints/pi05_membench_wx01_lora/pi05_lora_membench_wx01"

CKPT_DIR="${CKPT_DIR:-${CKPT_ROOT}/59999}"
CONFIG_NAME="${CONFIG_NAME:-pi05_membench_wx01_lora}"
PORT="${PORT:-56000}"
GPU="${GPU:-0}"
SUITE="${SUITE:-configs/eval/suites/wx01_v1.yaml}"
CKPT_STEP="$(basename "${CKPT_DIR}")"
RUN_ID="${RUN_ID:-wx01-pi05-${CKPT_STEP}-50seed-r50}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/eval}"
SERVER_HOST="${SERVER_HOST:-127.0.0.1}"
SERVER_INFERENCE_TIMEOUT_S="${SERVER_INFERENCE_TIMEOUT_S:-300}"
RETRY_ERRORS="${RETRY_ERRORS:-0}"
LOG_DIR="${OUTPUT_ROOT}/logs"

OPENPI_PYTHON="${ROOT}/policy/openpi/.venv/bin/python"
ROBOCASA_PYTHON="${ROOT}/.venv-robocasa/bin/python"
MPLIB_PYTHON="${ROOT}/.venv-mplib/bin/python"
MANISKILL_PANDA_DIR="${ROOT}/.venv-robocasa/lib/python3.11/site-packages/mani_skill/assets/robots/panda"

usage() {
    sed -n '2,25p' "${BASH_SOURCE[0]}" | sed -n 's/^# \{0,1\}//p'
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

start_server() {
    local pid_file="${LOG_DIR}/serve_wx01_${PORT}.pid"
    local log_file="${LOG_DIR}/serve_wx01_${PORT}.log"

    validate_common
    [[ -d "${CKPT_DIR}/params" ]] || {
        echo "Checkpoint params are missing: ${CKPT_DIR}/params" >&2
        exit 1
    }
    [[ -f "${CKPT_DIR}/assets/wx01_200seeds_v061/norm_stats.json" ]] || {
        echo "Checkpoint normalization stats are missing: ${CKPT_DIR}" >&2
        exit 1
    }
    mkdir -p "${LOG_DIR}"
    if [[ -f "${pid_file}" ]] && kill -0 "$(<"${pid_file}")" 2>/dev/null; then
        echo "Server ${PORT} is already running (pid $(<"${pid_file}"))"
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
            >"${log_file}" 2>&1 &
    echo $! >"${pid_file}"
    echo "Started server ${PORT} (pid $!, GPU ${GPU}, checkpoint ${CKPT_STEP})"
    echo "Log: ${log_file}"
}

serve_foreground() {
    local pid_file="${LOG_DIR}/serve_wx01_${PORT}.pid"

    validate_common
    [[ -d "${CKPT_DIR}/params" ]] || {
        echo "Checkpoint params are missing: ${CKPT_DIR}/params" >&2
        exit 1
    }
    [[ -f "${CKPT_DIR}/assets/wx01_200seeds_v061/norm_stats.json" ]] || {
        echo "Checkpoint normalization stats are missing: ${CKPT_DIR}" >&2
        exit 1
    }
    mkdir -p "${LOG_DIR}"
    echo $$ >"${pid_file}"
    cd "${ROOT}"
    export CUDA_VISIBLE_DEVICES="${GPU}"
    export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.9}"
    exec "${OPENPI_PYTHON}" scripts/serve/serve_openpi.py \
        --config "${CONFIG_NAME}" \
        --checkpoint "${CKPT_DIR}" \
        --image-size 224 \
        --state-schema membench.robot_state37.v2 \
        --inference-timeout-s "${SERVER_INFERENCE_TIMEOUT_S}" \
        --host 0.0.0.0 \
        --port "${PORT}"
}

stop_server() {
    local pid_file="${LOG_DIR}/serve_wx01_${PORT}.pid"
    if [[ -f "${pid_file}" ]] && kill -0 "$(<"${pid_file}")" 2>/dev/null; then
        kill "$(<"${pid_file}")"
        rm -f "${pid_file}"
        echo "Stopped server ${PORT}"
    else
        rm -f "${pid_file}" 2>/dev/null || true
        echo "Server ${PORT} is not running"
    fi
}

run_preflight() {
    validate_common
    cd "${ROOT}"
    "${ROBOCASA_PYTHON}" scripts/eval/run_remote.py \
        --endpoint "ws://${SERVER_HOST}:${PORT}" \
        --suite "${SUITE}" \
        --preflight-only
}

run_eval() {
    local retry_args=()
    validate_common
    cd "${ROOT}"
    export PYTHONPATH=.
    export MUJOCO_GL="${MUJOCO_GL:-egl}"
    export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
    export MEMBENCH_MPLIB_PYTHON="${MEMBENCH_MPLIB_PYTHON:-${MPLIB_PYTHON}}"
    export MEMBENCH_MANISKILL_PANDA_DIR="${MEMBENCH_MANISKILL_PANDA_DIR:-${MANISKILL_PANDA_DIR}}"
    [[ -x "${MEMBENCH_MPLIB_PYTHON}" ]] || {
        echo "MPlib environment is missing: ${MEMBENCH_MPLIB_PYTHON}" >&2
        exit 1
    }
    [[ -d "${MEMBENCH_MANISKILL_PANDA_DIR}" ]] || {
        echo "ManiSkill Panda assets are missing: ${MEMBENCH_MANISKILL_PANDA_DIR}" >&2
        exit 1
    }
    if [[ "${RETRY_ERRORS}" == "1" ]]; then
        retry_args=(--retry-errors)
    fi
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
    local pid_file="${LOG_DIR}/serve_wx01_${PORT}.pid"
    if [[ -f "${pid_file}" ]] && kill -0 "$(<"${pid_file}")" 2>/dev/null; then
        echo "Server ${PORT}: running (pid $(<"${pid_file}"), checkpoint ${CKPT_STEP})"
    else
        echo "Server ${PORT}: not running"
    fi
}

case "${1:-}" in
    serve) start_server ;;
    serve-fg) serve_foreground ;;
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
