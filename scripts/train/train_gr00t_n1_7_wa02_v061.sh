#!/usr/bin/env bash
# GR00T N1.7-3B LoRA fine-tuning on the local WA02 LeRobot v0.6.1 dataset.
#
# Usage:
#   conda activate membench
#   bash scripts/train/train_gr00t_n1_7_wa02_v061.sh --check-only
#   bash scripts/train/train_gr00t_n1_7_wa02_v061.sh
#   bash scripts/train/train_gr00t_n1_7_wa02_v061.sh --resume 060000

set -euo pipefail

MODE="train"
CHECKPOINT_STEP=""
case "${1:-}" in
    "") ;;
    --check-only) MODE="check" ;;
    --resume)
        MODE="resume"
        CHECKPOINT_STEP="${2:-}"
        [[ "$CHECKPOINT_STEP" =~ ^[0-9]{6}$ ]] || {
            echo "Checkpoint step must be six digits, for example 060000." >&2
            exit 2
        }
        ;;
    *)
        echo "Usage: $0 [--check-only | --resume CHECKPOINT_STEP]" >&2
        exit 2
        ;;
esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

# Paths can be overridden without editing this file.
DATASET_PATH="${DATASET_PATH:-$REPO_ROOT/data/wa02_200seeds_v061}"
MODEL_PATH="${MODEL_PATH:-/home/chensihang/pretrained_models/GR00T-N1.7-3B}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/outputs/gr00t_n1_7_wa02_v061_lora}"
TRAIN_ENTRY="${TRAIN_ENTRY:-$REPO_ROOT/scripts/train/lerobot_train_entry.py}"
VALIDATOR="${VALIDATOR:-$REPO_ROOT/scripts/train/validate_lerobot_v3_dataset.py}"
PYTHON_BIN="${PYTHON_BIN:-python}"
ACCELERATE_BIN="${ACCELERATE_BIN:-accelerate}"

# Use a writable local cache even in non-interactive or restricted launch environments.
export HF_HOME="${HF_HOME:-$REPO_ROOT/outputs/.cache/huggingface}"
mkdir -p "$HF_HOME"

# This configuration was previously measured at about 18.6-20.4 GiB per RTX 4090.
NUM_GPUS="${NUM_GPUS:-8}"
BATCH_SIZE="${BATCH_SIZE:-8}"
NUM_WORKERS="${NUM_WORKERS:-8}"
MAX_STEPS="${MAX_STEPS:-100000}"
LR="${LR:-2e-4}"
WARMUP_RATIO="${WARMUP_RATIO:-0.03}"
SAVE_EVERY="${SAVE_EVERY:-10000}"
LOG_EVERY="${LOG_EVERY:-10}"
MIN_FREE_GPU_GIB="${MIN_FREE_GPU_GIB:-21}"
MIN_FREE_DISK_GIB="${MIN_FREE_DISK_GIB:-60}"
WANDB_ENABLE="${WANDB_ENABLE:-true}"
WANDB_PROJECT="${WANDB_PROJECT:-gr00t-n1-7-wa02-v061}"

PEFT_TARGET_MODULES='["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj","fc1","fc2","to_q","to_k","to_v","to_out.0","net.0.proj","net.2","proj_out_1","proj_out_2"]'
PEFT_FULL_TRAINING_MODULES='["state_encoder","action_encoder","action_decoder"]'
PEFT_RANK=32
PEFT_LORA_ALPHA=64
PEFT_METHOD="lora"

TUNE_LLM=false
TUNE_VISUAL=false
TUNE_PROJECTOR=true
TUNE_DIFFUSION=true
TUNE_VLLN=true

fail() {
    echo "Preflight failed: $*" >&2
    exit 1
}

command -v "$PYTHON_BIN" >/dev/null 2>&1 || fail "Python executable not found: $PYTHON_BIN"
command -v "$ACCELERATE_BIN" >/dev/null 2>&1 || fail "Accelerate not found; activate the membench conda environment."
[[ -f "$DATASET_PATH/meta/info.json" ]] || fail "dataset not found or incomplete: $DATASET_PATH"
[[ -s "$MODEL_PATH/config.json" ]] || fail "base model config not found: $MODEL_PATH/config.json"
[[ -s "$MODEL_PATH/model.safetensors.index.json" ]] || fail "base model weight index is missing"
[[ -s "$MODEL_PATH/model-00001-of-00002.safetensors" ]] || fail "base model shard 1 is missing"
[[ -s "$MODEL_PATH/model-00002-of-00002.safetensors" ]] || fail "base model shard 2 is missing"
[[ -f "$TRAIN_ENTRY" ]] || fail "training entrypoint not found: $TRAIN_ENTRY"
[[ -f "$VALIDATOR" ]] || fail "dataset validator not found: $VALIDATOR"
if [[ "$MODE" == "resume" ]]; then
    RESUME_CONFIG="$OUTPUT_DIR/checkpoints/$CHECKPOINT_STEP/pretrained_model/train_config.json"
    [[ -f "$RESUME_CONFIG" ]] || fail "checkpoint config not found: $RESUME_CONFIG"
    RESUME_ARGS=(--resume=true "--config_path=$RESUME_CONFIG")
else
    [[ ! -e "$OUTPUT_DIR" ]] || fail "output already exists: $OUTPUT_DIR (use the resume script)"
    RESUME_ARGS=(--resume=false)
fi

"$PYTHON_BIN" - <<'PY'
import importlib
import sys
from importlib.metadata import version

required = ("torch", "accelerate", "lerobot", "pyarrow", "av", "transformers", "peft", "wandb")
for module_name in required:
    importlib.import_module(module_name)
if version("lerobot") != "0.6.1":
    raise SystemExit(f"LeRobot 0.6.1 is required, got {version('lerobot')}")
print(f"Python: {sys.executable}")
print(f"LeRobot: {version('lerobot')} | PyTorch: {version('torch')} | Accelerate: {version('accelerate')}")
PY

"$PYTHON_BIN" "$VALIDATOR" "$DATASET_PATH" \
    --repo-id wa02_200seeds_v061 \
    --expected-episodes 200 \
    --expected-fps 20 \
    --expected-state-dim 37 \
    --expected-action-dim 13 \
    --expected-cameras 3 \
    --action-horizon 40

mkdir -p "$(dirname "$OUTPUT_DIR")"
AVAILABLE_DISK_KIB="$(df -Pk "$(dirname "$OUTPUT_DIR")" | awk 'NR == 2 {print $4}')"
REQUIRED_DISK_KIB="$((MIN_FREE_DISK_GIB * 1024 * 1024))"
if (( AVAILABLE_DISK_KIB < REQUIRED_DISK_KIB )); then
    fail "less than ${MIN_FREE_DISK_GIB} GiB is free under $(dirname "$OUTPUT_DIR")"
fi

"$PYTHON_BIN" - "$NUM_GPUS" "$MIN_FREE_GPU_GIB" <<'PY'
import sys
import torch

required_gpus = int(sys.argv[1])
min_free_gib = float(sys.argv[2])
available_gpus = torch.cuda.device_count()
if available_gpus != required_gpus:
    raise SystemExit(f"expected {required_gpus} visible GPUs, found {available_gpus}")

for index in range(available_gpus):
    properties = torch.cuda.get_device_properties(index)
    free_bytes, total_bytes = torch.cuda.mem_get_info(index)
    free_gib = free_bytes / 2**30
    total_gib = total_bytes / 2**30
    print(f"GPU {index}: {properties.name}, free {free_gib:.1f}/{total_gib:.1f} GiB")
    if not torch.cuda.is_bf16_supported():
        raise SystemExit(f"GPU {index} does not support bf16")
    if free_gib < min_free_gib:
        raise SystemExit(
            f"GPU {index} has only {free_gib:.1f} GiB free; require at least {min_free_gib:.1f} GiB"
        )
PY

if [[ "$MODE" == "check" ]]; then
    echo "Preflight passed. No training was started."
    exit 0
fi

if [[ "$MODE" == "resume" ]]; then
    TRAIN_LOG="$(dirname "$OUTPUT_DIR")/train_gr00t_wa02_v061_resume_$(date +%Y%m%d_%H%M%S).log"
else
    TRAIN_LOG="$(dirname "$OUTPUT_DIR")/train_gr00t_wa02_v061_$(date +%Y%m%d_%H%M%S).log"
fi

echo "=========================================="
echo "GR00T N1.7-3B LoRA Training on WA02 v061"
echo "=========================================="
echo "Dataset: $DATASET_PATH"
echo "Model: $MODEL_PATH"
echo "Output: $OUTPUT_DIR"
echo "Log: $TRAIN_LOG"
if [[ "$MODE" == "resume" ]]; then
    echo "Resume from: $RESUME_CONFIG"
fi
echo "GPUs: $NUM_GPUS"
echo "Per-GPU batch: $BATCH_SIZE | Effective batch: $((BATCH_SIZE * NUM_GPUS))"
echo "Max steps: $MAX_STEPS | LR: $LR"
echo "PEFT: $PEFT_METHOD r=$PEFT_RANK alpha=$PEFT_LORA_ALPHA"
echo "=========================================="

"$ACCELERATE_BIN" launch \
    --num_processes="$NUM_GPUS" \
    --mixed_precision=bf16 \
    --multi_gpu \
    "$TRAIN_ENTRY" \
    \
    --dataset.repo_id="wa02_200seeds_v061" \
    --dataset.root="$DATASET_PATH" \
    --dataset.video_backend="pyav" \
    --dataset.use_imagenet_stats=false \
    \
    --policy.type="groot" \
    --policy.base_model_path="$MODEL_PATH" \
    --policy.pretrained_path="$MODEL_PATH" \
    --policy.embodiment_tag="robocasa_panda_omron" \
    \
    --policy.tune_llm=$TUNE_LLM \
    --policy.tune_visual=$TUNE_VISUAL \
    --policy.tune_projector=$TUNE_PROJECTOR \
    --policy.tune_diffusion_model=$TUNE_DIFFUSION \
    --policy.tune_vlln=$TUNE_VLLN \
    \
    --policy.max_state_dim=132 \
    --policy.max_action_dim=132 \
    --policy.chunk_size=40 \
    --policy.n_action_steps=40 \
    --policy.use_bf16=true \
    --policy.model_params_fp32=false \
    --policy.use_flash_attention=true \
    --policy.optimizer_lr="$LR" \
    --policy.warmup_ratio="$WARMUP_RATIO" \
    --policy.push_to_hub=false \
    \
    --peft.method_type="$PEFT_METHOD" \
    --peft.target_modules="$PEFT_TARGET_MODULES" \
    --peft.full_training_modules="$PEFT_FULL_TRAINING_MODULES" \
    --peft.r="$PEFT_RANK" \
    --peft.lora_alpha="$PEFT_LORA_ALPHA" \
    \
    --output_dir="$OUTPUT_DIR" \
    --batch_size="$BATCH_SIZE" \
    --num_workers="$NUM_WORKERS" \
    --steps="$MAX_STEPS" \
    --log_freq="$LOG_EVERY" \
    --save_freq="$SAVE_EVERY" \
    --save_checkpoint=true \
    "${RESUME_ARGS[@]}" \
    --wandb.enable="$WANDB_ENABLE" \
    --wandb.project="$WANDB_PROJECT" \
    --use_policy_training_preset=true \
    2>&1 | tee "$TRAIN_LOG"
