#!/usr/bin/env bash
# Train pi05 (LoRA) on the wr03_200seeds_v061 dataset.
#
# Usage:
#   scripts/train/pi05/train_pi05_wr03.sh              # fresh run
#   scripts/train/pi05/train_pi05_wr03.sh --overwrite  # overwrite existing run
#   scripts/train/pi05/train_pi05_wr03.sh --resume     # resume latest checkpoint
#
# Overridable via env:
#   EXP_NAME, NUM_TRAIN_STEPS, LOG_INTERVAL, GPU_IDS, BATCH_SIZE, CONFIG_NAME,
#   COMPUTE_NORM_STATS ...
#   (batch must be divisible by the number of GPUs)
#
# NOTE: dataset lives on the shared workspace; GPUs 0-3 are other users' memory
# holders, so this defaults to 4-7 (adjust GPU_IDS if a serve is running).
# If a previous run for the same CONFIG_NAME is still active on the same GPUs,
# stop it first, or give a different EXP_NAME / GPU_IDS.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${ROOT}"

# ---- defaults (override via env) ----
CONFIG_NAME="${CONFIG_NAME:-pi05_membench_wr03_lora}"
EXP_NAME="${EXP_NAME:-pi05_lora_membench_wr03}"
DATASET_REPO_ID="${DATASET_REPO_ID:-wr03_200seeds_v061}"
LEROBOT_HOME_DIR="${LEROBOT_HOME_DIR:-/mnt/shared_workspace/chensihang/dataset/membench}"
OPENPI_PRETRAIN_MODEL_DIR="${OPENPI_PRETRAIN_MODEL_DIR:-/home/chensihang/membench/pretrain_model}"
CAMERA_KEYS="${CAMERA_KEYS:-observation.images.robot0_agentview_right,observation.images.robot0_eye_in_hand}"
GPU_IDS="${GPU_IDS:-4,5,6,7}"
BATCH_SIZE="${BATCH_SIZE:-48}"
NUM_TRAIN_STEPS="${NUM_TRAIN_STEPS:-60000}"
LOG_INTERVAL="${LOG_INTERVAL:-10}"
# First run on a new dataset must compute its own norm stats (they live in
# assets/{CONFIG_NAME}/{repo_id}). Override with COMPUTE_NORM_STATS=false once done.
COMPUTE_NORM_STATS="${COMPUTE_NORM_STATS:-true}"

RESUME_FLAG=()
for arg in "$@"; do
  case "$arg" in
    --resume|--overwrite) RESUME_FLAG+=("$arg") ;;
    *) echo "unknown arg: $arg (only --resume / --overwrite are supported)" >&2; exit 1 ;;
  esac
done

if pgrep -f "train.py ${CONFIG_NAME}" >/dev/null 2>&1; then
  echo "WARNING: a training for '${CONFIG_NAME}' appears to already be running." >&2
  echo "  - to take over: stop it first, e.g.  pkill -f 'train.py ${CONFIG_NAME}'" >&2
  echo "  - or run with a different EXP_NAME / GPU_IDS" >&2
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
