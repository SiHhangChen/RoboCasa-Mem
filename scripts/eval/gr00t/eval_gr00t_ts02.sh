#!/usr/bin/env bash
# Evaluate one GR00T N1.7 LoRA checkpoint on the 50-seed TS-02 suite.
#
#   GPU=2 PORT=57010 ./scripts/eval/gr00t/eval_gr00t_ts02.sh serve-fg
#   PORT=57010 ./scripts/eval/gr00t/eval_gr00t_ts02.sh preflight
#   PORT=57010 ./scripts/eval/gr00t/eval_gr00t_ts02.sh eval
#   PORT=57010 ./scripts/eval/gr00t/eval_gr00t_ts02.sh summary
#   PORT=57010 ./scripts/eval/gr00t/eval_gr00t_ts02.sh stop
#
# Override CKPT_DIR, GPU, PORT, RUN_ID, BASE_MODEL_DIR, VLM_MODEL_DIR,
# GROOT_PYTHON, SUITE, OUTPUT_ROOT, and SERVER_INFERENCE_TIMEOUT_S as needed.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
CKPT_ROOT="${ROOT}/ckpts/gr00t/ts02"

CKPT_DIR="${CKPT_DIR:-${CKPT_ROOT}/100000/pretrained_model}"
BASE_MODEL_DIR="${BASE_MODEL_DIR:-/data/shared_workspace/chensihang/pretrained/GR00T-N1.7-3B}"
VLM_MODEL_DIR="${VLM_MODEL_DIR:-/data/shared_workspace/chensihang/pretrained/Cosmos-Reason2-2B}"
SUITE="${SUITE:-configs/eval/suites/ts02_v1.yaml}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/eval}"
SERVER_HOST="${SERVER_HOST:-127.0.0.1}"
SERVER_INFERENCE_TIMEOUT_S="${SERVER_INFERENCE_TIMEOUT_S:-300}"
PORT="${PORT:-57010}"
GPU="${GPU:-2}"
RETRY_ERRORS="${RETRY_ERRORS:-0}"

CKPT_STEP="$(basename "$(dirname "${CKPT_DIR}")")"
RUN_ID="${RUN_ID:-ts02-gr00t-${CKPT_STEP}-50seeds-r50}"
LOG_DIR="${OUTPUT_ROOT}/logs"

GROOT_PYTHON="${GROOT_PYTHON:-${ROOT}/.venv-gr00t/bin/python}"
ROBOCASA_PYTHON="${ROBOCASA_PYTHON:-${ROOT}/.venv-robocasa/bin/python}"
MPLIB_PYTHON="${MPLIB_PYTHON:-${ROOT}/.venv-mplib/bin/python}"
MANISKILL_PANDA_DIR="${MANISKILL_PANDA_DIR:-${ROOT}/.venv-robocasa/lib/python3.11/site-packages/mani_skill/assets/robots/panda}"

usage() {
    sed -n '2,12p' "${BASH_SOURCE[0]}" | sed -n 's/^# \{0,1\}//p'
}

validate_common() {
    [[ -x "${ROBOCASA_PYTHON}" ]] || {
        echo "RoboCasa environment is missing: ${ROBOCASA_PYTHON}" >&2
        exit 1
    }
    [[ -f "${ROOT}/${SUITE}" || -f "${SUITE}" ]] || {
        echo "Evaluation suite is missing: ${SUITE}" >&2
        exit 1
    }
}

validate_server() {
    validate_common
    [[ -x "${GROOT_PYTHON}" ]] || {
        echo "GR00T environment is missing: ${GROOT_PYTHON}" >&2
        exit 1
    }
    [[ -f "${CKPT_DIR}/train_config.json" ]] || {
        echo "GR00T checkpoint is missing: ${CKPT_DIR}" >&2
        exit 1
    }
    [[ -f "${BASE_MODEL_DIR}/config.json" ]] || {
        echo "GR00T base model is missing: ${BASE_MODEL_DIR}" >&2
        exit 1
    }
    [[ -f "${VLM_MODEL_DIR}/tokenizer.json" ]] || {
        echo "Cosmos tokenizer is missing: ${VLM_MODEL_DIR}" >&2
        exit 1
    }
}

serve_foreground() {
    local pid_file="${LOG_DIR}/serve_gr00t_ts02_${PORT}.pid"

    validate_server
    mkdir -p "${LOG_DIR}"
    echo $$ >"${pid_file}"
    cd "${ROOT}"
    export CUDA_VISIBLE_DEVICES="${GPU}"
    exec "${GROOT_PYTHON}" scripts/serve/serve_gr00t.py \
        --checkpoint "${CKPT_DIR}" \
        --base-model-dir "${BASE_MODEL_DIR}" \
        --vlm-model-dir "${VLM_MODEL_DIR}" \
        --image-size 224 \
        --state-schema membench.robot_state37.v2 \
        --inference-timeout-s "${SERVER_INFERENCE_TIMEOUT_S}" \
        --host 0.0.0.0 \
        --port "${PORT}"
}

stop_server() {
    local pid_file="${LOG_DIR}/serve_gr00t_ts02_${PORT}.pid"
    if [[ -f "${pid_file}" ]] && kill -0 "$(<"${pid_file}")" 2>/dev/null; then
        kill "$(<"${pid_file}")"
        rm -f "${pid_file}"
        echo "Stopped GR00T server ${PORT}"
    else
        rm -f "${pid_file}" 2>/dev/null || true
        echo "GR00T server ${PORT} is not running"
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
    [[ -x "${MPLIB_PYTHON}" ]] || {
        echo "MPlib environment is missing: ${MPLIB_PYTHON}" >&2
        exit 1
    }
    [[ -d "${MANISKILL_PANDA_DIR}" ]] || {
        echo "ManiSkill Panda assets are missing: ${MANISKILL_PANDA_DIR}" >&2
        exit 1
    }
    if [[ "${RETRY_ERRORS}" == "1" ]]; then
        retry_args=(--retry-errors)
    fi

    cd "${ROOT}"
    export PYTHONPATH=.
    export MUJOCO_GL="${MUJOCO_GL:-egl}"
    export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
    export MEMBENCH_MPLIB_PYTHON="${MEMBENCH_MPLIB_PYTHON:-${MPLIB_PYTHON}}"
    export MEMBENCH_MANISKILL_PANDA_DIR="${MEMBENCH_MANISKILL_PANDA_DIR:-${MANISKILL_PANDA_DIR}}"
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
    local pid_file="${LOG_DIR}/serve_gr00t_ts02_${PORT}.pid"
    if [[ -f "${pid_file}" ]] && kill -0 "$(<"${pid_file}")" 2>/dev/null; then
        echo "GR00T server ${PORT}: running (pid $(<"${pid_file}"), checkpoint ${CKPT_STEP})"
    else
        echo "GR00T server ${PORT}: not running"
    fi
}

case "${1:-}" in
    serve-fg|serve) serve_foreground ;;
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
