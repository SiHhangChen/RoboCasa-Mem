#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PORT="${PORT:-59219}"
BIND_HOST="${BIND_HOST:-127.0.0.1}"
SERVER_HOST="${SERVER_HOST:-127.0.0.1}"
BASELINE_PYTHON="${BASELINE_PYTHON:-python3}"
REPLAY_PYTHON="${REPLAY_PYTHON:-${ROOT}/.venv-gr00t/bin/python}"
DATASET_ROOT="${DATASET_ROOT:-${ROOT}/data/real_robot/wr07_lerobot_v3}"
ANNOTATION_ROOT="${ANNOTATION_ROOT:-${ROOT}/data/annotation_json_20260921/wr07_real_robot_v3}"

case "${1:-}" in
  serve)
    : "${CKPT_DIR:?Set CKPT_DIR to the trained baseline checkpoint}"
    : "${OMEGA_REPO:?Set OMEGA_REPO to the VGGT-Omega source directory}"
    : "${OMEGA_CHECKPOINT:?Set OMEGA_CHECKPOINT to the training Omega weights}"
    cd "${ROOT}"
    exec "${BASELINE_PYTHON}" scripts/serve/serve_baseline_wr07.py \
      --checkpoint "${CKPT_DIR}" --dataset-root "${DATASET_ROOT}" \
      --omega-repo "${OMEGA_REPO}" --omega-checkpoint "${OMEGA_CHECKPOINT}" \
      --omega-device "${OMEGA_DEVICE:-cuda:0}" --host "${BIND_HOST}" --port "${PORT}"
    ;;
  replay)
    cd "${ROOT}"
    exec "${REPLAY_PYTHON}" scripts/eval/real/replay_remote_dataset.py \
      --endpoint "ws://${SERVER_HOST}:${PORT}" --dataset-root "${DATASET_ROOT}" \
      --repo-id wr07-real-v3 --expected-config pi05_real_robot_wr07_v3 \
      --prompt-source annotation --annotation-root "${ANNOTATION_ROOT}" \
      --max-frames "${MAX_FRAMES:-100}" --stride 10 \
      --output "${OUTPUT:-${ROOT}/outputs/eval/real/wr07_baseline_omega_replay.json}"
    ;;
  *)
    echo "Usage: CKPT_DIR=... OMEGA_REPO=... OMEGA_CHECKPOINT=... $0 serve" >&2
    echo "       $0 replay" >&2
    exit 2
    ;;
esac
