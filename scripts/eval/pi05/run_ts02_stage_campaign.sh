#!/usr/bin/env bash
# Run the TS02 Pi05 15000 r50 stage campaign with a managed policy server.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
CKPT_DIR="${CKPT_DIR:-/data/ext_workspace/chensihang/ckpts/pi05/ts02/15000}"
CONFIG_NAME="${CONFIG_NAME:-pi05_membench_ts02_lora}"
DATASET_REPO_ID="${DATASET_REPO_ID:-ts02_compact_return_arm_200seeds_v061}"
PORT="${PORT:-55020}"
GPU="${GPU:-1}"
RENDER_GPU="${RENDER_GPU:-${GPU}}"
STAGE_SUITE="${STAGE_SUITE:-configs/eval/stages/ts02_pi05_r50_formal_v1.yaml}"
MAIN_ROLLOUT_RUN="${MAIN_ROLLOUT_RUN:-outputs/eval/ts02-pi05-lora-15000-50seeds}"
STAGE_RUN_ID="${STAGE_RUN_ID:-ts02-pi05-15000-stage-50seeds-r50-formal-v1}"
PREFIX_ROOT="${PREFIX_ROOT:-outputs/stage_prefixes/ts02_gr00t_r20_pilot_v1}"
LOG_DIR="${ROOT}/outputs/eval/logs"
SERVE_LOG="${LOG_DIR}/serve_ts02_pi05_15000_stage_${PORT}.log"
EVAL_LOG="${LOG_DIR}/eval_ts02_pi05_15000_stage_r50_formal_v1.log"
OPENPI_PYTHON="${ROOT}/policy/openpi/.venv/bin/python"
ROBOCASA_PYTHON="${ROOT}/.venv-robocasa/bin/python"
MPLIB_PYTHON="${ROOT}/.venv-mplib/bin/python"
MANISKILL_PANDA_DIR="${ROOT}/.venv-robocasa/lib/python3.11/site-packages/mani_skill/assets/robots/panda"

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

export OPENPI_PRETRAIN_MODEL_DIR="${OPENPI_PRETRAIN_MODEL_DIR:-${HOME}/openpi/pretrained}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.9}"
export DATASET_REPO_ID
CUDA_VISIBLE_DEVICES="${GPU}" "${OPENPI_PYTHON}" scripts/serve/serve_openpi.py \
    --config "${CONFIG_NAME}" \
    --checkpoint "${CKPT_DIR}" \
    --image-size 224 \
    --state-schema membench.robot_state37.v2 \
    --inference-timeout-s 300 \
    --host 0.0.0.0 \
    --port "${PORT}" \
    >"${SERVE_LOG}" 2>&1 &
server_pid=$!

until rg -q "server listening" "${SERVE_LOG}"; do
    if ! kill -0 "${server_pid}" 2>/dev/null; then
        wait "${server_pid}"
        exit 1
    fi
    sleep 2
done

export PYTHONPATH=.
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export MUJOCO_EGL_DEVICE_ID="${RENDER_GPU}"
export MEMBENCH_MPLIB_PYTHON="${MEMBENCH_MPLIB_PYTHON:-${MPLIB_PYTHON}}"
export MEMBENCH_MANISKILL_PANDA_DIR="${MEMBENCH_MANISKILL_PANDA_DIR:-${MANISKILL_PANDA_DIR}}"
export NO_PROXY="${NO_PROXY:+${NO_PROXY},}127.0.0.1,localhost"
export no_proxy="${no_proxy:+${no_proxy},}127.0.0.1,localhost"
"${ROBOCASA_PYTHON}" scripts/eval/run_stage_remote.py \
    --endpoint "ws://127.0.0.1:${PORT}" \
    --stage-suite "${STAGE_SUITE}" \
    --prefix-root "${PREFIX_ROOT}" \
    --main-rollout-run "${MAIN_ROLLOUT_RUN}" \
    --output-root outputs/stage_eval \
    --run-id "${STAGE_RUN_ID}" \
    --no-record-video \
    >"${EVAL_LOG}" 2>&1
