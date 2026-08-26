#!/usr/bin/env bash
# Evaluate a GR00T N1.7 LoRA checkpoint on WR-06.
#
#   ./scripts/eval/gr00t/eval_gr00t_wr06.sh serve
#   ./scripts/eval/gr00t/eval_gr00t_wr06.sh preflight
#   ./scripts/eval/gr00t/eval_gr00t_wr06.sh eval
#   ./scripts/eval/gr00t/eval_gr00t_wr06.sh summary
#
# Override paths with CKPT_DIR, BASE_MODEL_DIR, VLM_MODEL_DIR and GROOT_PYTHON.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
CKPT_DIR="${CKPT_DIR:-/data/shared_workspace/chensihang/ckpts/membench/gr00t/wr06/100000/pretrained_model}"
BASE_MODEL_DIR="${BASE_MODEL_DIR:-/data/shared_workspace/chensihang/pretrained/GR00T-N1.7-3B}"
VLM_MODEL_DIR="${VLM_MODEL_DIR:-/data/shared_workspace/chensihang/pretrained/Cosmos-Reason2-2B}"
SUITE="${SUITE:-configs/eval/suites/wr06_v1.yaml}"
RUN_ID="${RUN_ID:-wr06-gr00t-lora-100000-smoke}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/eval}"
SERVER_HOST="${SERVER_HOST:-127.0.0.1}"
PORT="${PORT:-8000}"

if [[ -z "${GROOT_PYTHON:-}" ]]; then
    for candidate in \
        "${ROOT}/.venv-gr00t/bin/python" \
        /home/user/anaconda3/envs/lerobot061/bin/python \
        /home/chensihang/anaconda3/envs/lerobot061/bin/python \
        /opt/anaconda3/envs/lerobot061/bin/python \
        "${ROOT}/policy/lerobot-0.6.1/.venv/bin/python"; do
        if [[ -x "${candidate}" ]]; then
            GROOT_PYTHON="${candidate}"
            break
        fi
    done
fi
GROOT_PYTHON="${GROOT_PYTHON:-}"
ROBOCASA_PYTHON="${ROBOCASA_PYTHON:-${ROOT}/.venv-robocasa/bin/python}"
if [[ ! -x "${ROBOCASA_PYTHON}" ]]; then
    ROBOCASA_RUN=(conda run -n robocasa python)
else
    ROBOCASA_RUN=("${ROBOCASA_PYTHON}")
fi
export MEMBENCH_MPLIB_PYTHON="${MEMBENCH_MPLIB_PYTHON:-${ROBOCASA_PYTHON}}"

usage() {
    sed -n '2,12p' "${BASH_SOURCE[0]}" | sed 's/^# //'
}

case "${1:-}" in
    serve)
        [[ -n "${GROOT_PYTHON}" && -x "${GROOT_PYTHON}" ]] || {
            echo "找不到 GR00T Python。请设置 GROOT_PYTHON=/path/to/lerobot061/bin/python" >&2
            exit 1
        }
        GROOT_PYTHON_VERSION="$("${GROOT_PYTHON}" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
        if [[ "${GROOT_PYTHON_VERSION}" != 3.1[2-9] && "${GROOT_PYTHON_VERSION}" != 3.[2-9][0-9] ]]; then
            echo "GR00T LeRobot 环境需要 Python >= 3.12，当前为 ${GROOT_PYTHON_VERSION}: ${GROOT_PYTHON}" >&2
            exit 1
        fi
        [[ -f "${CKPT_DIR}/train_config.json" ]] || { echo "checkpoint 不存在: ${CKPT_DIR}" >&2; exit 1; }
        [[ -f "${BASE_MODEL_DIR}/config.json" ]] || { echo "GR00T 基座不存在: ${BASE_MODEL_DIR}" >&2; exit 1; }
        [[ -f "${VLM_MODEL_DIR}/tokenizer.json" ]] || { echo "Cosmos tokenizer 不存在: ${VLM_MODEL_DIR}" >&2; exit 1; }
        cd "${ROOT}"
        exec "${GROOT_PYTHON}" scripts/serve/serve_gr00t.py \
            --checkpoint "${CKPT_DIR}" \
            --base-model-dir "${BASE_MODEL_DIR}" \
            --vlm-model-dir "${VLM_MODEL_DIR}" \
            --image-size 224 \
            --state-schema membench.robot_state37.v2 \
            --host 0.0.0.0 \
            --port "${PORT}"
        ;;
    preflight)
        cd "${ROOT}"
        "${ROBOCASA_RUN[@]}" scripts/eval/run_remote.py \
            --endpoint "ws://${SERVER_HOST}:${PORT}" \
            --suite "${SUITE}" \
            --preflight-only
        ;;
    eval)
        cd "${ROOT}"
        export PYTHONPATH=.
        export MUJOCO_GL="${MUJOCO_GL:-egl}"
        export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
        "${ROBOCASA_RUN[@]}" scripts/eval/run_remote.py \
            --endpoint "ws://${SERVER_HOST}:${PORT}" \
            --suite "${SUITE}" \
            --output-root "${OUTPUT_ROOT}" \
            --run-id "${RUN_ID}"
        ;;
    summary)
        cd "${ROOT}"
        "${ROBOCASA_RUN[@]}" python scripts/eval/summarize_run.py "${OUTPUT_ROOT}/${RUN_ID}"
        ;;
    help|-h|--help)
        usage
        ;;
    *)
        echo "未知子命令: ${1:-}" >&2
        usage >&2
        exit 2
        ;;
esac
