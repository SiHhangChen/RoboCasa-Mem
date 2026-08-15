#!/usr/bin/env bash
#
# 评测 MemoryVLA wr03 checkpoint(WR-03)的可复用入口。
#
# 推理全部在服务器进行;client 只负责创建仿真、发送 observation、
# 执行 action chunk 和记录产物,本地只需一张 GPU 跑 MuJoCo 仿真渲染。
#
# 用法(各子命令在对应终端执行):
#   ./scripts/eval/eval_memvla_wr03.sh serve      终端1(服务器):启动 MemoryVLA 策略服务(常驻)
#   ./scripts/eval/eval_memvla_wr03.sh preflight  终端2(本地):协议 preflight 检查
#   ./scripts/eval/eval_memvla_wr03.sh eval       终端2(本地):闭环评测
#   ./scripts/eval/eval_memvla_wr03.sh summary    任意终端:重建评测汇总
#
# 可通过环境变量覆盖(保持默认即可复用):
#   CKPT           MemoryVLA checkpoint .pt,默认 step-050000-epoch-14
#   MEMVLA_PYTHON  serve 使用的 Python 解释器(服务器上 memvla 环境),默认 ~/.conda/envs/memvla/bin/python
#   SERVER_GPUS    serve 使用的 GPU,默认 0
#   DEVICE         serve 推理设备,默认 cuda
#   USE_BF16       1 启用 bf16 推理(默认开启),0 关闭
#   NUM_DDIM_STEPS DDIM 采样步数,默认 10
#   ROBOCASA_RUN   client 使用的运行环境,默认 conda run -n robocasa(本地)
#   SUITE          评测 suite,默认 configs/eval/suites/wr03_memvla_v1.yaml
#   RUN_ID         评测运行 ID,默认 wr03-memvla-v1
#   OUTPUT_ROOT    评测输出根目录,默认 outputs/eval
#   SERVER_HOST    客户端连接的服务端地址,默认 127.0.0.1(同机)
#   PORT           服务端端口,默认 8000

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

CKPT="${CKPT:-${ROOT}/policy/MemoryVLA/runs/memvla_wr03/checkpoints/step-050000-epoch-14-loss=0.0032.pt}"
MEMVLA_PYTHON="${MEMVLA_PYTHON:-${HOME}/.conda/envs/memvla/bin/python}"
SERVER_GPUS="${SERVER_GPUS:-0}"
DEVICE="${DEVICE:-cuda}"
USE_BF16="${USE_BF16:-1}"
NUM_DDIM_STEPS="${NUM_DDIM_STEPS:-10}"
ROBOCASA_RUN=(conda run -n robocasa)
SUITE="${SUITE:-configs/eval/suites/wr03_memvla_v1.yaml}"
RUN_ID="${RUN_ID:-wr03-memvla-v1}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/eval}"
SERVER_HOST="${SERVER_HOST:-127.0.0.1}"
PORT="${PORT:-8000}"

usage() {
    sed -n '2,40p' "${BASH_SOURCE[0]}" | grep -E '^#' | sed 's/^# \{0,1\}//'
    exit 0
}

case "${1:-}" in
    serve)
        [[ -x "${MEMVLA_PYTHON}" ]] || { echo "memvla Python 不存在: ${MEMVLA_PYTHON}" >&2; exit 1; }
        [[ -f "${CKPT}" ]] || { echo "checkpoint 不存在: ${CKPT}" >&2; exit 1; }
        cd "${ROOT}"
        export PYTHONPATH=.
        bf16_flag=()
        if [[ "${USE_BF16}" == "1" ]]; then
            bf16_flag=(--use-bf16)
        fi
        exec env CUDA_VISIBLE_DEVICES="${SERVER_GPUS}" \
            "${MEMVLA_PYTHON}" scripts/serve/serve_memvla.py \
            --checkpoint "${CKPT}" \
            --device "${DEVICE}" \
            --host 0.0.0.0 \
            --port "${PORT}" \
            "${bf16_flag[@]}" \
            --num-ddim-steps "${NUM_DDIM_STEPS}"
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
