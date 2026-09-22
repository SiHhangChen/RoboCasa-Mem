#!/bin/bash
# GR00T N1.7-3B LoRA fine-tuning on the WA05 v0.6.1 dataset.
#
# Usage:
#   conda activate membench
#   bash scripts/train/train_gr00t_n1_7_wa05_v061.sh

set -euo pipefail

DATASET_PATH="/home/chensihang/membench/data/wa05_200seeds_v061"
MODEL_PATH="/home/chensihang/pretrained_models/GR00T-N1.7-3B"
OUTPUT_DIR="./outputs/gr00t_n1_7_wa05_v061_lora"

# 8 x RTX 4090 configuration validated on the other MemBench tasks.
BATCH_SIZE=8
NUM_WORKERS=8
MAX_STEPS=100000
LR=2e-4
WARMUP_RATIO=0.03
SAVE_EVERY=10000
LOG_EVERY=10

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

if [[ ! -f "$DATASET_PATH/meta/info.json" ]]; then
    echo "Dataset not found: $DATASET_PATH" >&2
    exit 1
fi
if [[ ! -f "$MODEL_PATH/config.json" ]]; then
    echo "Base model not found: $MODEL_PATH" >&2
    exit 1
fi
if [[ -e "$OUTPUT_DIR" ]]; then
    echo "Output already exists: $OUTPUT_DIR" >&2
    echo "Use the resume script or choose a new output directory." >&2
    exit 1
fi

mkdir -p "$(dirname "$OUTPUT_DIR")"
TRAIN_LOG="$(dirname "$OUTPUT_DIR")/train_gr00t_wa05_v061_$(date +%Y%m%d_%H%M%S).log"

echo "=========================================="
echo "GR00T N1.7-3B LoRA Training on WA05 v061"
echo "=========================================="
echo "Dataset: $DATASET_PATH"
echo "Model: $MODEL_PATH"
echo "Output: $OUTPUT_DIR"
echo "GPUs: 8 x RTX 4090"
echo "Per-GPU batch: $BATCH_SIZE | Effective batch: $((BATCH_SIZE * 8))"
echo "Max steps: $MAX_STEPS | LR: $LR"
echo "Log: $TRAIN_LOG"
echo "=========================================="

accelerate launch \
    --num_processes=8 \
    --mixed_precision=bf16 \
    --multi_gpu \
    /home/chensihang/membench/scripts/train/lerobot_train_entry.py \
    \
    --dataset.repo_id="wa05_200seeds_v061" \
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
    --policy.optimizer_lr=$LR \
    --policy.warmup_ratio=$WARMUP_RATIO \
    --policy.push_to_hub=false \
    \
    --peft.method_type="$PEFT_METHOD" \
    --peft.target_modules="$PEFT_TARGET_MODULES" \
    --peft.full_training_modules="$PEFT_FULL_TRAINING_MODULES" \
    --peft.r=$PEFT_RANK \
    --peft.lora_alpha=$PEFT_LORA_ALPHA \
    \
    --output_dir="$OUTPUT_DIR" \
    --batch_size=$BATCH_SIZE \
    --num_workers=$NUM_WORKERS \
    --steps=$MAX_STEPS \
    --log_freq=$LOG_EVERY \
    --save_freq=$SAVE_EVERY \
    --save_checkpoint=true \
    --resume=false \
    --wandb.enable=true \
    --wandb.project="gr00t-n1-7-wa05-v061" \
    --use_policy_training_preset=true \
    2>&1 | tee "$TRAIN_LOG"
