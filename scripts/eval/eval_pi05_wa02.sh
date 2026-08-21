#!/usr/bin/env bash
#
# 评测 pi05 LoRA checkpoint（WA-02）的可复用入口。单个 server，ckpt/端口/GPU
# 全部通过环境变量指定，测任意 step 无需改脚本。
#
# 用法（各子命令在对应终端执行）：
#   ./scripts/eval/eval_pi05_wa02.sh serve      终端1：后台启动 OpenPI 策略服务（常驻）
#   ./scripts/eval/eval_pi05_wa02.sh preflight  终端2：协议 preflight 检查
#   ./scripts/eval/eval_pi05_wa02.sh eval       终端2：闭环评测
#   ./scripts/eval/eval_pi05_wa02.sh summary    任意终端：重建评测汇总
#   ./scripts/eval/eval_pi05_wa02.sh stop       停止 server
#   ./scripts/eval/eval_pi05_wa02.sh status     查看 server 状态
#
# 测其他 step（如 55000）或换端口/GPU：
#   CKPT_DIR=.../pi05_lora_membench_wa02/55000 PORT=55000 GPU=1 \
#     ./scripts/eval/eval_pi05_wa02.sh serve
#
# 同时评测两个 ckpt（各占一块 GPU / 端口，例如 55000 与 59999）：
#   CKPT_DIR=.../55000 PORT=55000 GPU=0 ./scripts/eval/eval_pi05_wa02.sh serve
#   CKPT_DIR=.../59999 PORT=59999 GPU=1 ./scripts/eval/eval_pi05_wa02.sh serve
#   然后分别 eval：
#   CKPT_DIR=.../55000 PORT=55000 ./scripts/eval/eval_pi05_wa02.sh eval
#   CKPT_DIR=.../59999 PORT=59999 ./scripts/eval/eval_pi05_wa02.sh eval
#
# 可通过环境变量覆盖：
#   CKPT_DIR      checkpoint 目录，默认 .../pi05_lora_membench_wa02/59999
#   CONFIG_NAME   OpenPI 配置名，默认 pi05_membench_wa02_lora
#   PORT          服务端端口，默认 59999（同时用于区分 pid/日志文件）
#   GPU           挂载的物理卡，默认 0
#   SUITE         评测 suite，默认 configs/eval/suites/wa02_smoke_v1.yaml
#   RUN_ID        评测运行 ID，默认 wa02-pi05-lora-{ckpt目录名}（如 -59999/-55000）
#   OUTPUT_ROOT   评测输出根目录，默认 outputs/eval
#   SERVER_HOST   客户端连接的服务端地址，默认 127.0.0.1（同机）
#   RETRY_ERRORS  设为 1 时续跑 status=error 的 episode（断网中断后恢复用），默认 0

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

CKPT_DIR="${CKPT_DIR:-${ROOT}/policy/openpi/checkpoints/pi05_membench_wa02_lora/pi05_lora_membench_wa02/59999}"
CONFIG_NAME="${CONFIG_NAME:-pi05_membench_wa02_lora}"
PORT="${PORT:-59999}"
GPU="${GPU:-0}"
SUITE="${SUITE:-configs/eval/suites/wa02_smoke_v1.yaml}"
CKPT_STEP="$(basename "${CKPT_DIR}")"
RUN_ID="${RUN_ID:-wa02-pi05-lora-${CKPT_STEP}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/eval}"
SERVER_HOST="${SERVER_HOST:-127.0.0.1}"
RETRY_ERRORS="${RETRY_ERRORS:-0}"
LOG_DIR="${OUTPUT_ROOT}/logs"

OPENPI_PYTHON="${ROOT}/policy/openpi/.venv/bin/python"
ROBOCASA_RUN=(conda run -n robocasa)

usage() {
    sed -n '2,40p' "${BASH_SOURCE[0]}" | grep -E '^#' | sed 's/^# \{0,1\}//'
    exit 0
}

start_server() {
    local pid_file
    pid_file="${LOG_DIR}/serve_${PORT}.pid"
    mkdir -p "${LOG_DIR}"
    [[ -x "${OPENPI_PYTHON}" ]] || { echo "OpenPI .venv 不存在: ${OPENPI_PYTHON}" >&2; exit 1; }
    [[ -d "${CKPT_DIR}/params" ]] || { echo "checkpoint 不存在: ${CKPT_DIR}" >&2; exit 1; }
    if [[ -f "${pid_file}" ]] && kill -0 "$(cat "${pid_file}")" 2>/dev/null; then
        echo "server ${PORT} 已在运行 (pid $(cat "${pid_file}"))，跳过"
        return 0
    fi
    cd "${ROOT}"
    export OPENPI_PRETRAIN_MODEL_DIR="${OPENPI_PRETRAIN_MODEL_DIR:-${HOME}/openpi/pretrained}"
    export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.9}"
    CUDA_VISIBLE_DEVICES="${GPU}" \
    nohup "${OPENPI_PYTHON}" scripts/serve/serve_openpi.py \
        --config "${CONFIG_NAME}" \
        --checkpoint "${CKPT_DIR}" \
        --image-size 224 \
        --state-schema membench.robot_state37.v2 \
        --host 0.0.0.0 \
        --port "${PORT}" \
        --pytorch-device cuda:0 \
        > "${LOG_DIR}/serve_${PORT}.log" 2>&1 &
    echo $! > "${pid_file}"
    echo "server ${PORT} 已启动 (pid $!, GPU ${GPU}, ckpt ${CKPT_DIR})"
    echo "  log: ${LOG_DIR}/serve_${PORT}.log"
}

stop_server() {
    local pid_file
    pid_file="${LOG_DIR}/serve_${PORT}.pid"
    if [[ -f "${pid_file}" ]] && kill -0 "$(cat "${pid_file}")" 2>/dev/null; then
        kill "$(cat "${pid_file}")"
        rm -f "${pid_file}"
        echo "server ${PORT} 已停止"
    else
        rm -f "${pid_file}" 2>/dev/null || true
        echo "server ${PORT} 未在运行"
    fi
}

run_preflight() {
    cd "${ROOT}"
    "${ROBOCASA_RUN[@]}" python scripts/eval/run_remote.py \
        --endpoint "ws://${SERVER_HOST}:${PORT}" \
        --suite "${SUITE}" \
        --preflight-only
}

run_eval() {
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
}

run_summary() {
    cd "${ROOT}"
    "${ROBOCASA_RUN[@]}" python scripts/eval/summarize_run.py "${OUTPUT_ROOT}/${RUN_ID}"
}

status() {
    local pid_files pid_file port
    mkdir -p "${LOG_DIR}"
    shopt -s nullglob
    pid_files=("${LOG_DIR}"/serve_*.pid)
    shopt -u nullglob
    if [[ ${#pid_files[@]} -eq 0 ]]; then
        echo "没有已记录的 server (${LOG_DIR})"
        return 0
    fi
    for pid_file in "${pid_files[@]}"; do
        port="$(basename "${pid_file}" | sed 's/serve_//; s/\.pid//')"
        if kill -0 "$(cat "${pid_file}")" 2>/dev/null; then
            echo "server ${port}: 运行中 (pid $(cat "${pid_file}"))"
        else
            echo "server ${port}: 未运行"
        fi
    done
}

case "${1:-}" in
    serve)
        start_server
        ;;
    preflight)
        run_preflight
        ;;
    eval)
        run_eval
        ;;
    summary)
        run_summary
        ;;
    stop)
        stop_server
        ;;
    status)
        status
        ;;
    help|-h|--help)
        usage
        ;;
    *)
        echo "未知子命令: ${1:-}" >&2
        usage
        ;;
esac
