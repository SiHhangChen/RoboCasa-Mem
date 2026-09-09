#!/usr/bin/env bash
# Evaluate one GR00T N1.7 LoRA checkpoint on the 50-seed WA-03 r40 suite.
#
#   GPU=4 PORT=57103 CKPT_STEP=080000 ./scripts/eval/gr00t/eval_gr00t_wa03.sh start
#   PORT=57103 CKPT_STEP=080000 ./scripts/eval/gr00t/eval_gr00t_wa03.sh wait
#   PORT=57103 CKPT_STEP=080000 ./scripts/eval/gr00t/eval_gr00t_wa03.sh eval
#   CKPT_STEP=080000 ./scripts/eval/gr00t/eval_gr00t_wa03.sh summary
#   ./scripts/eval/gr00t/eval_gr00t_wa03.sh prefixes
#   PORT=57103 CKPT_STEP=080000 ./scripts/eval/gr00t/eval_gr00t_wa03.sh stage-eval
#   CKPT_STEP=080000 ./scripts/eval/gr00t/eval_gr00t_wa03.sh stage-summary
#   PORT=57103 ./scripts/eval/gr00t/eval_gr00t_wa03.sh stop

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
CKPT_ROOT="${CKPT_ROOT:-/data/chensihang/ckpts/gr00t/wa03}"
CKPT_STEP="${CKPT_STEP:-100000}"
CKPT_DIR="${CKPT_DIR:-${CKPT_ROOT}/${CKPT_STEP}/pretrained_model}"
CKPT_STEP="$(basename "$(dirname "${CKPT_DIR}")")"
BASE_MODEL_DIR="${BASE_MODEL_DIR:-/data/shared_workspace/chensihang/pretrained/GR00T-N1.7-3B}"
VLM_MODEL_DIR="${VLM_MODEL_DIR:-/data/shared_workspace/chensihang/pretrained/Cosmos-Reason2-2B}"
SUITE="${SUITE:-configs/eval/legacy_suites/wa03_gr00t_r40_v1.yaml}"
STAGE_SUITE="${STAGE_SUITE:-configs/eval/stages/wa03_gr00t_r40_formal_v1.yaml}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/eval}"
STAGE_OUTPUT_ROOT="${STAGE_OUTPUT_ROOT:-outputs/stage_eval}"
PREFIX_ROOT="${PREFIX_ROOT:-outputs/stage_prefixes/wa03_gr00t_r40_v1}"
RUN_ID="${RUN_ID:-wa03-gr00t-${CKPT_STEP}-50seeds-r40}"
STAGE_RUN_ID="${STAGE_RUN_ID:-wa03-gr00t-${CKPT_STEP}-stage-50seeds-r40-formal-v1}"
MAIN_ROLLOUT_RUN="${MAIN_ROLLOUT_RUN:-${OUTPUT_ROOT}/${RUN_ID}}"
SERVER_HOST="${SERVER_HOST:-127.0.0.1}"
SERVER_INFERENCE_TIMEOUT_S="${SERVER_INFERENCE_TIMEOUT_S:-300}"
PORT="${PORT:-57103}"
GPU="${GPU:-4}"
RENDER_GPU="${RENDER_GPU:-${GPU}}"
RETRY_ERRORS="${RETRY_ERRORS:-0}"
STAGE_RECORD_VIDEO="${STAGE_RECORD_VIDEO:-0}"

LOG_DIR="${OUTPUT_ROOT}/logs"
PID_FILE="${LOG_DIR}/serve_gr00t_wa03_${PORT}.pid"
LOG_FILE="${LOG_DIR}/serve_gr00t_wa03_${PORT}.log"
GROOT_PYTHON="${GROOT_PYTHON:-${ROOT}/.venv-gr00t/bin/python}"
ROBOCASA_PYTHON="${ROBOCASA_PYTHON:-${ROOT}/.venv-robocasa/bin/python}"
MPLIB_PYTHON="${MPLIB_PYTHON:-${ROOT}/.venv-mplib/bin/python}"
MANISKILL_PANDA_DIR="${MANISKILL_PANDA_DIR:-${ROOT}/.venv-robocasa/lib/python3.11/site-packages/mani_skill/assets/robots/panda}"

usage() { sed -n '2,13p' "${BASH_SOURCE[0]}" | sed -n 's/^# \{0,1\}//p'; }

validate_common() {
    [[ -x "${ROBOCASA_PYTHON}" ]] || { echo "RoboCasa environment is missing: ${ROBOCASA_PYTHON}" >&2; exit 1; }
    [[ -f "${ROOT}/${SUITE}" || -f "${SUITE}" ]] || { echo "Evaluation suite is missing: ${SUITE}" >&2; exit 1; }
}

validate_server() {
    validate_common
    [[ -x "${GROOT_PYTHON}" ]] || { echo "GR00T environment is missing: ${GROOT_PYTHON}" >&2; exit 1; }
    [[ -f "${CKPT_DIR}/train_config.json" ]] || { echo "GR00T checkpoint is missing: ${CKPT_DIR}" >&2; exit 1; }
    [[ -f "${BASE_MODEL_DIR}/config.json" ]] || { echo "GR00T base model is missing: ${BASE_MODEL_DIR}" >&2; exit 1; }
    [[ -f "${VLM_MODEL_DIR}/tokenizer.json" ]] || { echo "Cosmos tokenizer is missing: ${VLM_MODEL_DIR}" >&2; exit 1; }
}

configure_runtime() {
    [[ -x "${MPLIB_PYTHON}" ]] || { echo "MPlib environment is missing: ${MPLIB_PYTHON}" >&2; exit 1; }
    [[ -d "${MANISKILL_PANDA_DIR}" ]] || { echo "ManiSkill Panda assets are missing: ${MANISKILL_PANDA_DIR}" >&2; exit 1; }
    export PYTHONPATH=.
    export MUJOCO_GL="${MUJOCO_GL:-egl}"
    export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
    export MUJOCO_EGL_DEVICE_ID="${RENDER_GPU}"
    export MEMBENCH_MPLIB_PYTHON="${MEMBENCH_MPLIB_PYTHON:-${MPLIB_PYTHON}}"
    export MEMBENCH_MANISKILL_PANDA_DIR="${MEMBENCH_MANISKILL_PANDA_DIR:-${MANISKILL_PANDA_DIR}}"
    export NO_PROXY="${NO_PROXY:+${NO_PROXY},}127.0.0.1,localhost"
    export no_proxy="${no_proxy:+${no_proxy},}127.0.0.1,localhost"
}

start_server() {
    validate_server; mkdir -p "${LOG_DIR}"
    if [[ -f "${PID_FILE}" ]] && kill -0 "$(<"${PID_FILE}")" 2>/dev/null; then
        echo "GR00T server ${PORT} is already running (pid $(<"${PID_FILE}"))"; return 0
    fi
    cd "${ROOT}"
    CUDA_VISIBLE_DEVICES="${GPU}" nohup "${GROOT_PYTHON}" scripts/serve/serve_gr00t.py \
        --checkpoint "${CKPT_DIR}" --base-model-dir "${BASE_MODEL_DIR}" \
        --vlm-model-dir "${VLM_MODEL_DIR}" --image-size 224 \
        --state-schema membench.robot_state37.v2 --inference-timeout-s "${SERVER_INFERENCE_TIMEOUT_S}" \
        --host 0.0.0.0 --port "${PORT}" >"${LOG_FILE}" 2>&1 &
    echo $! >"${PID_FILE}"
    echo "Started GR00T server ${PORT} (pid $!, GPU ${GPU}, checkpoint ${CKPT_STEP})"
    echo "Log: ${LOG_FILE}"
}

stop_server() {
    if [[ -f "${PID_FILE}" ]] && kill -0 "$(<"${PID_FILE}")" 2>/dev/null; then
        kill "$(<"${PID_FILE}")"; rm -f "${PID_FILE}"; echo "Stopped GR00T server ${PORT}"
    else rm -f "${PID_FILE}" 2>/dev/null || true; echo "GR00T server ${PORT} is not running"; fi
}

wait_for_server() {
    local attempts="${WAIT_ATTEMPTS:-120}" delay="${WAIT_DELAY_S:-10}"
    for ((i=1; i<=attempts; i++)); do
        grep -q "server listening" "${LOG_FILE}" 2>/dev/null && { echo "GR00T server ${PORT} is ready"; return 0; }
        [[ -f "${PID_FILE}" ]] && ! kill -0 "$(<"${PID_FILE}")" 2>/dev/null && { echo "GR00T server died; see ${LOG_FILE}" >&2; return 1; }
        sleep "${delay}"
    done
    echo "GR00T server did not become ready; see ${LOG_FILE}" >&2; return 1
}

run_preflight() { validate_common; cd "${ROOT}"; "${ROBOCASA_PYTHON}" scripts/eval/run_remote.py --endpoint "ws://${SERVER_HOST}:${PORT}" --suite "${SUITE}" --preflight-only; }
run_eval() {
    validate_common; configure_runtime; cd "${ROOT}"; local retry_args=(); [[ "${RETRY_ERRORS}" == "1" ]] && retry_args=(--retry-errors)
    "${ROBOCASA_PYTHON}" scripts/eval/run_remote.py --endpoint "ws://${SERVER_HOST}:${PORT}" --suite "${SUITE}" --output-root "${OUTPUT_ROOT}" --run-id "${RUN_ID}" "${retry_args[@]}"
}
run_summary() { validate_common; cd "${ROOT}"; "${ROBOCASA_PYTHON}" scripts/eval/summarize_run.py "${OUTPUT_ROOT}/${RUN_ID}"; }
run_prefixes() { validate_common; configure_runtime; cd "${ROOT}"; "${ROBOCASA_PYTHON}" scripts/eval/build_expert_prefixes.py --stage-suite "${STAGE_SUITE}" --output-root "${PREFIX_ROOT}" "${@}"; }
run_stage_preflight() { validate_common; cd "${ROOT}"; "${ROBOCASA_PYTHON}" scripts/eval/run_stage_remote.py --endpoint "ws://${SERVER_HOST}:${PORT}" --stage-suite "${STAGE_SUITE}" --preflight-only; }
run_stage_eval() {
    [[ -d "${PREFIX_ROOT}" ]] || { echo "Expert prefix root is missing: ${PREFIX_ROOT}" >&2; exit 1; }
    [[ -f "${MAIN_ROLLOUT_RUN}/summary.json" ]] || { echo "Completed main rollout is required: ${MAIN_ROLLOUT_RUN}" >&2; exit 1; }
    validate_common; configure_runtime; cd "${ROOT}"
    local video_args=(); [[ "${STAGE_RECORD_VIDEO}" == "1" ]] && video_args=(--record-video); [[ "${STAGE_RECORD_VIDEO}" == "0" ]] && video_args=(--no-record-video)
    "${ROBOCASA_PYTHON}" scripts/eval/run_stage_remote.py --endpoint "ws://${SERVER_HOST}:${PORT}" --stage-suite "${STAGE_SUITE}" --prefix-root "${PREFIX_ROOT}" --main-rollout-run "${MAIN_ROLLOUT_RUN}" --output-root "${STAGE_OUTPUT_ROOT}" --run-id "${STAGE_RUN_ID}" "${video_args[@]}"
}
run_stage_summary() { validate_common; cd "${ROOT}"; "${ROBOCASA_PYTHON}" scripts/eval/summarize_stage_run.py "${STAGE_OUTPUT_ROOT}/${STAGE_RUN_ID}"; }
status() { if [[ -f "${PID_FILE}" ]] && kill -0 "$(<"${PID_FILE}")" 2>/dev/null; then echo "GR00T server ${PORT}: running (pid $(<"${PID_FILE}"), checkpoint ${CKPT_STEP})"; else echo "GR00T server ${PORT}: not running"; fi; }

case "${1:-}" in
    start) start_server;; serve-fg|serve) validate_server; mkdir -p "${LOG_DIR}"; echo $$ >"${PID_FILE}"; cd "${ROOT}"; export CUDA_VISIBLE_DEVICES="${GPU}"; exec "${GROOT_PYTHON}" scripts/serve/serve_gr00t.py --checkpoint "${CKPT_DIR}" --base-model-dir "${BASE_MODEL_DIR}" --vlm-model-dir "${VLM_MODEL_DIR}" --image-size 224 --state-schema membench.robot_state37.v2 --inference-timeout-s "${SERVER_INFERENCE_TIMEOUT_S}" --host 0.0.0.0 --port "${PORT}";;
    wait) wait_for_server;; preflight) run_preflight;; eval) run_eval;; summary) run_summary;; prefixes) shift; run_prefixes "${@}";; stage-preflight) run_stage_preflight;; stage-eval) run_stage_eval;; stage-summary) run_stage_summary;; stop) stop_server;; status) status;; help|-h|--help) usage;; *) echo "Unknown command: ${1:-}" >&2; usage >&2; exit 2;;
esac
