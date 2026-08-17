#!/usr/bin/env bash
#
# 评测 GR00T N1.7 LoRA checkpoint（WR-03）的可复用入口。
#
# 用法（各子命令在对应终端执行）：
#   ./scripts/eval/eval_gr00t_wr03.sh serve      终端1：启动 GR00T 策略服务（常驻）
#   ./scripts/eval/eval_gr00t_wr03.sh preflight  终端2：协议 preflight 检查
#   ./scripts/eval/eval_gr00t_wr03.sh eval       终端2：闭环评测（正式 50 seeds）
#   ./scripts/eval/eval_gr00t_wr03.sh summary    任意终端：重建评测汇总
#
# 可通过环境变量覆盖（保持默认即可复用）：
#   CKPT_DIR          checkpoint 目录，默认 ckpts/gr00t/WR03/90000/pretrained_model
#   BASE_MODEL_DIR    NVIDIA 原版 GR00T-N1.7-3B 目录
#   VLM_MODEL_DIR     Cosmos-Reason2-2B 目录（vlm_encode tokenizer）
#   SUITE             评测 suite，默认 configs/eval/suites/wr03_gr00t_50seeds_v1.yaml
#   RUN_ID            评测运行 ID，默认 wr03-gr00t-lora-90000-50seed-20action
#   OUTPUT_ROOT       评测输出根目录，默认 outputs/eval
#   SERVER_HOST       客户端连接的服务端地址，默认 127.0.0.1（同机）
#   PORT              服务端端口，默认 8000

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

CKPT_DIR="${CKPT_DIR:-${ROOT}/ckpts/gr00t/WR03/90000/pretrained_model}"
BASE_MODEL_DIR="${BASE_MODEL_DIR:-/home/user/chensihang/pretrained/GR00T-N1.7-3B}"
VLM_MODEL_DIR="${VLM_MODEL_DIR:-/home/user/chensihang/pretrained/Cosmos-Reason2-2B}"
SUITE="${SUITE:-configs/eval/suites/wr03_gr00t_50seeds_v1.yaml}"
RUN_ID="${RUN_ID:-wr03-gr00t-lora-90000-50seed-20action}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/eval}"
SERVER_HOST="${SERVER_HOST:-127.0.0.1}"
PORT="${PORT:-8000}"

GROOT_PYTHON="/home/user/anaconda3/envs/lerobot061/bin/python"
[[ -x "${GROOT_PYTHON}" ]] || GROOT_PYTHON="${ROOT}/policy/lerobot-0.6.1/.venv/bin/python"
ROBOCASA_RUN=(conda run -n robocasa)

usage() {
    sed -n '2,40p' "${BASH_SOURCE[0]}" | grep -E '^#' | sed 's/^# \{0,1\}//'
    exit 0
}

case "${1:-}" in
    serve)
        [[ -x "${GROOT_PYTHON}" ]] || { echo "GR00T python 不存在: ${GROOT_PYTHON}" >&2; exit 1; }
        [[ -f "${CKPT_DIR}/train_config.json" ]] || { echo "checkpoint 不存在: ${CKPT_DIR}" >&2; exit 1; }
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
        "${ROBOCASA_RUN[@]}" python scripts/eval/run_remote.py \
            --endpoint "ws://${SERVER_HOST}:${PORT}" \
            --suite "${SUITE}" \
            --preflight-only
        ;;
    eval)
        cd "${ROOT}"
        export PYTHONPATH=.
        export MUJOCO_GL="${MUJOCO_GL:-egl}"
        export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
        "${ROBOCASA_RUN[@]}" python scripts/eval/run_remote.py \
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
        usage
        ;;
esac
