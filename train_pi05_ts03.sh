#!/usr/bin/env bash
# Train pi05 (LoRA) on the ts03_200seeds_v061 dataset.
#
# Usage:
#   ./train_pi05_ts03.sh              # fresh run: compute norm stats, 60k steps
#   ./train_pi05_ts03.sh --overwrite  # start fresh even if the checkpoint dir exists
#   ./train_pi05_ts03.sh --resume     # resume from latest checkpoint in the exp dir
#
# Overridable via env:
#   EXP_NAME, NUM_TRAIN_STEPS, LOG_INTERVAL, GPU_IDS, BATCH_SIZE, CONFIG_NAME,
#   COMPUTE_NORM_STATS ...
#   (batch must be divisible by the number of GPUs)
#
# NOTE: GPU 0 is occupied by another process, so this uses GPUs 2-7 by default.
# If a previous run for the same CONFIG_NAME is still active on the same GPUs,
# stop it first, or give a different EXP_NAME / GPU_IDS.

set -euo pipefail

cd "$(dirname "$0")"

# ---- defaults (override via env) ----
CONFIG_NAME="${CONFIG_NAME:-pi05_membench_ts03_lora}"
EXP_NAME="${EXP_NAME:-pi05_lora_membench_ts03}"
DATASET_REPO_ID="${DATASET_REPO_ID:-ts03_200seeds_v061}"
LEROBOT_HOME_DIR="${LEROBOT_HOME_DIR:-/data/shared_workspace/chensihang/dataset/membench}"
OPENPI_PRETRAIN_MODEL_DIR="${OPENPI_PRETRAIN_MODEL_DIR:-/data/shared_workspace/tangzhipeng/ckpts/openpi/openpi_assets}"
CAMERA_KEYS="${CAMERA_KEYS:-observation.images.robot0_agentview_right,observation.images.robot0_eye_in_hand}"
GPU_IDS="${GPU_IDS:-2,3,4,5,6,7}"
BATCH_SIZE="${BATCH_SIZE:-48}"
NUM_TRAIN_STEPS="${NUM_TRAIN_STEPS:-60000}"
LOG_INTERVAL="${LOG_INTERVAL:-10}"
# First run on a new dataset must compute its own norm stats (they live in
# assets/{CONFIG_NAME}/{repo_id}). Override with COMPUTE_NORM_STATS=false once done.
COMPUTE_NORM_STATS="${COMPUTE_NORM_STATS:-true}"
# Video decode backend for the v3 pyarrow loader. "torchcodec" is the default but
# needs a system FFmpeg; fall back to PyAV when torchcodec can't load (see the
# libtorchcodec loading traceback). Override via VIDEO_BACKEND.
VIDEO_BACKEND="${VIDEO_BACKEND:-pyav}"

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
echo "  video  : ${VIDEO_BACKEND}"
echo "  flags  : ${RESUME_FLAG[@]:-none}"
echo "=========================================================="

# PYTHONUNBUFFERED: train.py logs Step loss via pbar.write -> sys.stdout, which
# is block-buffered when redirected to a file. Unbuffering makes loss lines show
# up in the log as they are produced instead of every ~8KB of buffered output.
PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}" \
CONFIG_NAME="${CONFIG_NAME}" \
EXP_NAME="${EXP_NAME}" \
DATASET_REPO_ID="${DATASET_REPO_ID}" \
LEROBOT_HOME_DIR="${LEROBOT_HOME_DIR}" \
OPENPI_PRETRAIN_MODEL_DIR="${OPENPI_PRETRAIN_MODEL_DIR}" \
OPENPI_MEMBENCH_CAMERA_KEYS="${CAMERA_KEYS}" \
OPENPI_MEMBENCH_VIDEO_BACKEND="${VIDEO_BACKEND}" \
CUDA_VISIBLE_DEVICES="${GPU_IDS}" \
BATCH_SIZE="${BATCH_SIZE}" \
COMPUTE_NORM_STATS="${COMPUTE_NORM_STATS}" \
bash policy/openpi/scripts/train_membench_hs_wx_01.sh \
  --num-train-steps="${NUM_TRAIN_STEPS}" \
  --log-interval="${LOG_INTERVAL}" \
  "${RESUME_FLAG[@]}"
