#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
MODEL="${MODEL:-full}"
PORT="${PORT:-59217}"
BIND_HOST="${BIND_HOST:-127.0.0.1}"
SERVER_HOST="${SERVER_HOST:-127.0.0.1}"
OPENPI_PYTHON="${OPENPI_PYTHON:-${ROOT}/policy/openpi/.venv/bin/python}"
REPLAY_PYTHON="${REPLAY_PYTHON:-${ROOT}/.venv-gr00t/bin/python}"
DATASET_ROOT="${DATASET_ROOT:-${ROOT}/data/real_robot/wr07_lerobot_v3}"
ANNOTATION_ROOT="${ANNOTATION_ROOT:-${ROOT}/data/real_robot/wr07_real_robot_v3}"
PROMPT_SOURCE="${PROMPT_SOURCE:-auto}"

case "${MODEL}" in
  full) CONFIG_NAME=pi05_real_robot_wr07_v3 ;;
  subgoal) CONFIG_NAME=pi05_real_robot_wr07_v3_subgoal ;;
  *) echo "MODEL must be full or subgoal" >&2; exit 2 ;;
esac

case "${1:-}" in
  serve)
    : "${CKPT_DIR:?Set CKPT_DIR to an OpenPI checkpoint directory}"
    [[ -x "${OPENPI_PYTHON}" ]] || { echo "Missing OpenPI Python: ${OPENPI_PYTHON}" >&2; exit 1; }
    [[ -d "${CKPT_DIR}/params" || -f "${CKPT_DIR}/model.safetensors" ]] || {
      echo "Missing model weights under ${CKPT_DIR}" >&2; exit 1;
    }
    cd "${ROOT}"
    exec "${OPENPI_PYTHON}" scripts/serve/serve_openpi.py \
      --config "${CONFIG_NAME}" --checkpoint "${CKPT_DIR}" \
      --image-size 224 --runtime-profile membench.cobot_magic_wr07.v1 \
      --state-schema membench.cobot_magic_state17.v1 \
      --host "${BIND_HOST}" --port "${PORT}"
    ;;
  replay)
    [[ -x "${REPLAY_PYTHON}" ]] || { echo "Missing replay Python: ${REPLAY_PYTHON}" >&2; exit 1; }
    output="${OUTPUT:-${ROOT}/outputs/eval/real/wr07_${MODEL}_replay.json}"
    annotation_args=()
    if [[ "${PROMPT_SOURCE}" == "annotation" ]]; then
      annotation_args=(--annotation-root "${ANNOTATION_ROOT}")
    fi
    cd "${ROOT}"
    exec "${REPLAY_PYTHON}" scripts/eval/real/replay_remote_dataset.py \
      --endpoint "ws://${SERVER_HOST}:${PORT}" \
      --dataset-root "${DATASET_ROOT}" --repo-id wr07-real-v3 \
      --expected-config "${CONFIG_NAME}" \
      --prompt-source "${PROMPT_SOURCE}" "${annotation_args[@]}" \
      --max-frames "${MAX_FRAMES:-100}" --stride "${STRIDE:-30}" \
      --output "${output}"
    ;;
  *)
    echo "Usage: MODEL=full|subgoal CKPT_DIR=... $0 serve" >&2
    echo "       MODEL=full|subgoal $0 replay" >&2
    exit 2
    ;;
esac
