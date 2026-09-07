#!/usr/bin/env bash
# Train pi05 (LoRA) on the WR05 per-frame subtask-prompt dataset.
#
# Usage:
#   scripts/train/pi05/train_pi05_wr05.sh              # fresh run
#   scripts/train/pi05/train_pi05_wr05.sh --overwrite  # overwrite existing run
#   scripts/train/pi05/train_pi05_wr05.sh --resume     # resume latest checkpoint

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${ROOT}"

CONFIG_NAME="${CONFIG_NAME:-pi05_membench_wr05_lora}"
EXP_NAME="${EXP_NAME:-pi05_lora_membench_wr05}"
DATASET_REPO_ID="${DATASET_REPO_ID:-pi05/wr05_200seeds_v061_subtasks_v3}"
LEROBOT_HOME_DIR="${LEROBOT_HOME_DIR:-/home/chensihang/membench/data}"
OPENPI_PRETRAIN_MODEL_DIR="${OPENPI_PRETRAIN_MODEL_DIR:-/home/chensihang/membench/pretrain_model}"
CAMERA_KEYS="${CAMERA_KEYS:-observation.images.robot0_agentview_right,observation.images.robot0_eye_in_hand}"
GPU_IDS="${GPU_IDS:-4,5,6,7}"
BATCH_SIZE="${BATCH_SIZE:-48}"
NUM_TRAIN_STEPS="${NUM_TRAIN_STEPS:-60000}"
LOG_INTERVAL="${LOG_INTERVAL:-10}"
COMPUTE_NORM_STATS="${COMPUTE_NORM_STATS:-false}"

RESUME_FLAG=()
for arg in "$@"; do
  case "$arg" in
    --resume|--overwrite) RESUME_FLAG+=("$arg") ;;
    *) echo "unknown arg: $arg (only --resume / --overwrite are supported)" >&2; exit 1 ;;
  esac
done

if pgrep -f "train.py ${CONFIG_NAME}" >/dev/null 2>&1; then
  echo "WARNING: a training for '${CONFIG_NAME}' appears to already be running." >&2
  echo "Stop it first or choose a different EXP_NAME / GPU_IDS." >&2
fi

echo "=========================================================="
echo " pi05 LoRA training on ${DATASET_REPO_ID}"
echo "  config : ${CONFIG_NAME}"
echo "  exp    : ${EXP_NAME}"
echo "  GPUs   : ${GPU_IDS}   batch=${BATCH_SIZE}"
echo "  steps  : ${NUM_TRAIN_STEPS}   log-interval=${LOG_INTERVAL}"
echo "  norm   : ${COMPUTE_NORM_STATS}"
echo "  flags  : ${RESUME_FLAG[@]:-none}"
echo "=========================================================="

CONFIG_NAME="${CONFIG_NAME}" \
EXP_NAME="${EXP_NAME}" \
DATASET_REPO_ID="${DATASET_REPO_ID}" \
LEROBOT_HOME_DIR="${LEROBOT_HOME_DIR}" \
OPENPI_PRETRAIN_MODEL_DIR="${OPENPI_PRETRAIN_MODEL_DIR}" \
OPENPI_MEMBENCH_CAMERA_KEYS="${CAMERA_KEYS}" \
CUDA_VISIBLE_DEVICES="${GPU_IDS}" \
BATCH_SIZE="${BATCH_SIZE}" \
COMPUTE_NORM_STATS="${COMPUTE_NORM_STATS}" \
bash "${ROOT}/scripts/train/pi05/train_membench.sh" \
  --num-train-steps="${NUM_TRAIN_STEPS}" \
  --log-interval="${LOG_INTERVAL}" \
  "${RESUME_FLAG[@]}"
