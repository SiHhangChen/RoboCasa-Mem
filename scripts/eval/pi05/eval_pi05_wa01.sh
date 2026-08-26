#!/usr/bin/env bash
#
# 评测 pi05 LoRA checkpoint（WA-01）的可复用入口。
#
# 用法（各子命令在对应终端执行）：
#   ./scripts/eval/pi05/eval_pi05_wa01.sh serve      终端1：启动 OpenPI 策略服务（常驻）
#   ./scripts/eval/pi05/eval_pi05_wa01.sh preflight  终端2：协议 preflight 检查
#   ./scripts/eval/pi05/eval_pi05_wa01.sh eval       终端2：闭环评测
#   ./scripts/eval/pi05/eval_pi05_wa01.sh summary    任意终端：重建评测汇总
#
# 可通过环境变量覆盖（保持默认即可复用）：
#   CKPT_DIR      checkpoint 目录，默认 ckpts/pi05/.../15000
#   CONFIG_NAME   OpenPI 配置名，默认 pi05_membench_wa01_lora
#   SUITE         评测 suite，默认 configs/eval/suites/wa01_v1.yaml
#   RUN_ID        评测运行 ID，默认 wa01-pi05-lora-15000
#   OUTPUT_ROOT   评测输出根目录，默认 outputs/eval
#   SERVER_HOST   客户端连接的服务端地址，默认 127.0.0.1（同机）
#   PORT          服务端端口，默认 8000
#   RETRY_ERRORS  设为 1 时续跑 status=error 的 episode（断网中断后恢复用），默认 0

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"

CKPT_DIR="${CKPT_DIR:-${ROOT}/ckpts/pi05/pi05_membench_wa01_lora/pi05_lora_membench_wa01/45000}"
CONFIG_NAME="${CONFIG_NAME:-pi05_membench_wa01_lora}"
SUITE="${SUITE:-configs/eval/suites/wa01_v1.yaml}"
RUN_ID="${RUN_ID:-wa01-pi05-lora-45000}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/eval}"
SERVER_HOST="${SERVER_HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
RETRY_ERRORS="${RETRY_ERRORS:-0}"

OPENPI_PYTHON="${ROOT}/policy/openpi/.venv/bin/python"
ROBOCASA_RUN=(conda run -n robocasa)

usage() {
    sed -n '2,40p' "${BASH_SOURCE[0]}" | grep -E '^#' | sed 's/^# \{0,1\}//'
    exit 0
}

case "${1:-}" in
    serve)
        [[ -x "${OPENPI_PYTHON}" ]] || { echo "OpenPI .venv 不存在: ${OPENPI_PYTHON}" >&2; exit 1; }
        [[ -d "${CKPT_DIR}/params" ]] || { echo "checkpoint 不存在: ${CKPT_DIR}" >&2; exit 1; }
        cd "${ROOT}"
        export OPENPI_PRETRAIN_MODEL_DIR="${OPENPI_PRETRAIN_MODEL_DIR:-${HOME}/openpi/pretrained}"
        export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.9}"
        exec "${OPENPI_PYTHON}" scripts/serve/serve_openpi.py \
            --config "${CONFIG_NAME}" \
            --checkpoint "${CKPT_DIR}" \
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
        retry_args=()
        if [[ "${RETRY_ERRORS}" == "1" ]]; then
            retry_args=(--retry-errors)
        fi
        "${ROBOCASA_RUN[@]}" python scripts/eval/run_remote.py \
            --endpoint "ws://${SERVER_HOST}:${PORT}" \
            --suite "${SUITE}" \
            --output-root "${OUTPUT_ROOT}" \
            --run-id "${RUN_ID}" \
            "${retry_args[@]}"
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
