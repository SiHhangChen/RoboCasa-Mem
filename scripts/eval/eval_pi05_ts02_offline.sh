#!/usr/bin/env bash
# Evaluate a pi0.5 LoRA TS02 checkpoint on the held-out LeRobot dataset.
#
# Usage:
#   ./scripts/eval/eval_pi05_ts02_offline.sh
#
# Override with CKPT_DIR, DATASET_ROOT, CONFIG_NAME, GPU, OUTPUT_ROOT,
# BATCH_SIZE, ACTION_STRIDE, DIFFUSION_STEPS, or OFFLINE_OUTPUT.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

CKPT_DIR="${CKPT_DIR:-${ROOT}/policy/openpi/checkpoints/pi05_membench_ts02_lora/pi05_lora_membench_ts02/55000}"
DATASET_ROOT="${DATASET_ROOT:-${ROOT}/data/val/ts02_val_5seeds_v061}"
CONFIG_NAME="${CONFIG_NAME:-pi05_membench_ts02_lora}"
GPU="${GPU:-2}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/eval}"
BATCH_SIZE="${BATCH_SIZE:-8}"
ACTION_STRIDE="${ACTION_STRIDE:-50}"
DIFFUSION_STEPS="${DIFFUSION_STEPS:-10}"
CKPT_STEP="$(basename "${CKPT_DIR}")"
OFFLINE_OUTPUT="${OFFLINE_OUTPUT:-${OUTPUT_ROOT}/offline/ts02-pi05-lora-${CKPT_STEP}-val5.json}"
LOSS_VALUES_OUTPUT="${LOSS_VALUES_OUTPUT:-${OUTPUT_ROOT}/offline/loss_values/ts02-pi05-lora-${CKPT_STEP}-val5-losses.npz}"

OPENPI_PYTHON="${ROOT}/policy/openpi/.venv/bin/python"

[[ -x "${OPENPI_PYTHON}" ]] || { echo "OpenPI Python missing: ${OPENPI_PYTHON}" >&2; exit 1; }
[[ -d "${CKPT_DIR}/params" ]] || { echo "checkpoint missing: ${CKPT_DIR}" >&2; exit 1; }
[[ -d "${DATASET_ROOT}" ]] || { echo "validation dataset missing: ${DATASET_ROOT}" >&2; exit 1; }

cd "${ROOT}"
CUDA_VISIBLE_DEVICES="${GPU}" \
XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.9}" \
"${OPENPI_PYTHON}" scripts/eval/eval_openpi_checkpoint.py \
    --config "${CONFIG_NAME}" \
    --checkpoint "${CKPT_DIR}" \
    --dataset-root "${DATASET_ROOT}" \
    --output "${OFFLINE_OUTPUT}" \
    --loss-values-output "${LOSS_VALUES_OUTPUT}" \
    --batch-size "${BATCH_SIZE}" \
    --action-stride "${ACTION_STRIDE}" \
    --diffusion-steps "${DIFFUSION_STEPS}"
