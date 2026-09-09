#!/usr/bin/env bash
#
# Evaluate a pi0.5 LoRA checkpoint on the canonical WA02 suite.
#
# Usage:
#   GPU=7 PORT=55040 ./scripts/eval/pi05/eval_pi05_wa02.sh serve
#   PORT=55040 ./scripts/eval/pi05/eval_pi05_wa02.sh preflight
#   PORT=55040 ./scripts/eval/pi05/eval_pi05_wa02.sh eval
#   PORT=55040 ./scripts/eval/pi05/eval_pi05_wa02.sh summary
#   ./scripts/eval/pi05/eval_pi05_wa02.sh stage-prefix
#   PORT=55040 ./scripts/eval/pi05/eval_pi05_wa02.sh stage-eval
#   ./scripts/eval/pi05/eval_pi05_wa02.sh stage-summary
#   PORT=55040 ./scripts/eval/pi05/eval_pi05_wa02.sh status
#   PORT=55040 ./scripts/eval/pi05/eval_pi05_wa02.sh stop
#   GPU=7 PORT=55040 ./scripts/eval/pi05/eval_pi05_wa02.sh full
#
# Environment overrides:
#   CKPT_DIR      checkpoint directory (default: ckpts/pi05/wa02/45000)
#   CONFIG_NAME   OpenPI training config (default: pi05_membench_wa02_lora)
#   PORT          policy server port (default: 55040)
#   GPU           physical GPU visible to the policy server (default: 0)
#   RENDER_GPU    physical GPU used by MuJoCo EGL in the rollout client (default: GPU)
#   SUITE         evaluation suite (default: configs/eval/suites/wa02_v1.yaml)
#   RUN_ID        output run ID (default: wa02-pi05-{checkpoint_step}-50seed-r20)
#   OUTPUT_ROOT   evaluation output root (default: outputs/eval)
#   SERVER_HOST   policy server host used by the client (default: 127.0.0.1)
#   SERVER_INFERENCE_TIMEOUT_S server-side inference timeout (default: 300)
#   XLA_PYTHON_CLIENT_MEM_FRACTION OpenPI server GPU memory fraction (default: 0.75)
#   RETRY_ERRORS  set to 1 to rerun episodes whose previous status is error

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
CKPT_ROOT="${CKPT_ROOT:-${HOME}/ckpt/pi05/wa02}"

CKPT_DIR="${CKPT_DIR:-${CKPT_ROOT}/45000}"
CONFIG_NAME="${CONFIG_NAME:-pi05_membench_wa02_lora}"
PORT="${PORT:-55040}"
GPU="${GPU:-0}"
RENDER_GPU="${RENDER_GPU:-${GPU}}"
SUITE="${SUITE:-configs/eval/suites/wa02_v1.yaml}"
CKPT_STEP="$(basename "${CKPT_DIR}")"
RUN_ID="${RUN_ID:-wa02-pi05-${CKPT_STEP}-50seed-r20}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/eval}"
SERVER_HOST="${SERVER_HOST:-127.0.0.1}"
SERVER_INFERENCE_TIMEOUT_S="${SERVER_INFERENCE_TIMEOUT_S:-300}"
SERVER_STARTUP_TIMEOUT_S="${SERVER_STARTUP_TIMEOUT_S:-900}"
XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.75}"
RETRY_ERRORS="${RETRY_ERRORS:-0}"
STAGE_SUITE="${STAGE_SUITE:-configs/eval/stages/wa02_pi05_45000_r20_pilot_v1.yaml}"
PREFIX_ROOT="${PREFIX_ROOT:-outputs/stage_prefixes/wa02_pi05_45000_r20_pilot_v1}"
STAGE_OUTPUT_ROOT="${STAGE_OUTPUT_ROOT:-outputs/stage_eval}"
STAGE_RUN_ID="${STAGE_RUN_ID:-wa02-pi05-${CKPT_STEP}-stage-50seed-r20-pilot-v1}"
STAGE_RECORD_VIDEO="${STAGE_RECORD_VIDEO:-1}"
MAIN_ROLLOUT_RUN="${MAIN_ROLLOUT_RUN:-${OUTPUT_ROOT}/wa02-pi05-${CKPT_STEP}-50seed-r20-dynamictemp-v3}"
LOG_DIR="${OUTPUT_ROOT}/logs"

OPENPI_PYTHON="${ROOT}/policy/openpi/.venv/bin/python"
ROBOCASA_PYTHON="${ROOT}/.venv-robocasa/bin/python"
MPLIB_PYTHON="${ROOT}/.venv-mplib/bin/python"
MANISKILL_PANDA_DIR="${ROOT}/.venv-robocasa/lib/python3.11/site-packages/mani_skill/assets/robots/panda"
PID_FILE="${LOG_DIR}/serve_wa02_${PORT}.pid"
LOG_FILE="${LOG_DIR}/serve_wa02_${PORT}.log"

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
    [[ -f "${CKPT_DIR}/assets/wa02_200seeds_v061/norm_stats.json" ]] || {
        echo "Checkpoint normalization stats are missing: ${CKPT_DIR}/assets/wa02_200seeds_v061/norm_stats.json" >&2
        exit 1
    }
}

wait_for_server() {
    local server_pid="$1"
    local elapsed=0
    while (( elapsed < SERVER_STARTUP_TIMEOUT_S )); do
        if ! kill -0 "${server_pid}" 2>/dev/null; then
            echo "WA02 policy server exited during startup; inspect ${LOG_FILE}" >&2
            rm -f "${PID_FILE}"
            tail -n 35 "${LOG_FILE}" >&2 || true
            exit 1
        fi
        if grep -q "server listening on .*:${PORT}" "${LOG_FILE}" 2>/dev/null; then
            return 0
        fi
        if (( elapsed == 0 || elapsed % 10 == 0 )); then
            echo "Waiting for WA02 policy server ${PORT} to load (${elapsed}s/${SERVER_STARTUP_TIMEOUT_S}s)..."
        fi
        sleep 1
        ((elapsed += 1))
    done
    echo "WA02 policy server did not start within ${SERVER_STARTUP_TIMEOUT_S}s; inspect ${LOG_FILE}" >&2
    kill "${server_pid}" 2>/dev/null || true
    rm -f "${PID_FILE}"
    tail -n 35 "${LOG_FILE}" >&2 || true
    exit 1
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
    echo "Started WA02 policy server ${PORT} (pid ${server_pid}, GPU ${GPU}, checkpoint ${CKPT_STEP}, XLA memory fraction ${XLA_PYTHON_CLIENT_MEM_FRACTION})"
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

stage_environment() {
    [[ -x "${MPLIB_PYTHON}" ]] || {
        echo "MPlib environment is missing: ${MPLIB_PYTHON}" >&2
        exit 1
    }
    [[ -d "${MANISKILL_PANDA_DIR}" ]] || {
        echo "ManiSkill Panda assets are missing: ${MANISKILL_PANDA_DIR}" >&2
        exit 1
    }
    export PYTHONPATH=.
    export MUJOCO_GL="${MUJOCO_GL:-egl}"
    export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
    export MUJOCO_EGL_DEVICE_ID="${RENDER_GPU}"
    export MEMBENCH_MPLIB_PYTHON="${MEMBENCH_MPLIB_PYTHON:-${MPLIB_PYTHON}}"
    export MEMBENCH_MANISKILL_PANDA_DIR="${MEMBENCH_MANISKILL_PANDA_DIR:-${MANISKILL_PANDA_DIR}}"
    export NO_PROXY="${NO_PROXY:+${NO_PROXY},}127.0.0.1,localhost"
    export no_proxy="${no_proxy:+${no_proxy},}127.0.0.1,localhost"
}

run_stage_prefix() {
    validate_common
    [[ -f "${ROOT}/${STAGE_SUITE}" || -f "${STAGE_SUITE}" ]] || {
        echo "Stage suite is missing: ${STAGE_SUITE}" >&2
        exit 1
    }
    cd "${ROOT}"
    stage_environment
    "${ROBOCASA_PYTHON}" scripts/eval/build_expert_prefixes.py \
        --stage-suite "${STAGE_SUITE}" \
        --output-root "${PREFIX_ROOT}"
}

run_stage_eval() {
    local video_args=()
    validate_common
    [[ -d "${PREFIX_ROOT}" ]] || {
        echo "Expert prefix root is missing: ${PREFIX_ROOT}" >&2
        exit 1
    }
    [[ -f "${MAIN_ROLLOUT_RUN}/summary.json" ]] || {
        echo "Completed main rollout is required: ${MAIN_ROLLOUT_RUN}" >&2
        exit 1
    }
    if [[ "${STAGE_RECORD_VIDEO}" == "1" ]]; then
        video_args=(--record-video)
    elif [[ "${STAGE_RECORD_VIDEO}" == "0" ]]; then
        video_args=(--no-record-video)
    else
        echo "STAGE_RECORD_VIDEO must be 0 or 1" >&2
        exit 2
    fi
    cd "${ROOT}"
    stage_environment
    "${ROBOCASA_PYTHON}" scripts/eval/run_stage_remote.py \
        --endpoint "ws://${SERVER_HOST}:${PORT}" \
        --stage-suite "${STAGE_SUITE}" \
        --prefix-root "${PREFIX_ROOT}" \
        --main-rollout-run "${MAIN_ROLLOUT_RUN}" \
        --output-root "${STAGE_OUTPUT_ROOT}" \
        --run-id "${STAGE_RUN_ID}" \
        "${video_args[@]}"
}

run_stage_summary() {
    validate_common
    cd "${ROOT}"
    "${ROBOCASA_PYTHON}" scripts/eval/summarize_stage_run.py \
        "${STAGE_OUTPUT_ROOT}/${STAGE_RUN_ID}"
}

status() {
    if [[ -f "${PID_FILE}" ]] && kill -0 "$(<"${PID_FILE}")" 2>/dev/null; then
        echo "Server ${PORT}: running (pid $(<"${PID_FILE}"), checkpoint ${CKPT_STEP})"
    else
        echo "Server ${PORT}: not running"
    fi
}

run_full() {
    start_server
    trap stop_server EXIT
    run_preflight
    run_eval
}

case "${1:-}" in
    serve) start_server ;;
    preflight) run_preflight ;;
    eval) run_eval ;;
    summary) run_summary ;;
    stage-prefix) run_stage_prefix ;;
    stage-eval) run_stage_eval ;;
    stage-summary) run_stage_summary ;;
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
