#!/bin/bash
# Resume TS02 compact-return-arm training from a saved checkpoint.
#
# Usage:
#   conda activate membench
#   bash scripts/train/train_gr00t_n1_7_ts02_compact_return_arm_v061_resume.sh 060000

set -euo pipefail

CHECKPOINT_STEP="${1:-060000}"
if [[ ! "$CHECKPOINT_STEP" =~ ^[0-9]{6}$ ]]; then
    echo "Checkpoint step must be six digits, for example 060000." >&2
    exit 1
fi

DATASET_PATH="/home/chensihang/membench/data/ts02_compact_return_arm_200seeds_v061"
MODEL_PATH="/home/chensihang/pretrained_models/GR00T-N1.7-3B"
OUTPUT_DIR="./outputs/gr00t_n1_7_ts02_compact_return_arm_v061_lora"
RESUME_CONFIG="$OUTPUT_DIR/checkpoints/$CHECKPOINT_STEP/pretrained_model/train_config.json"

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
if [[ ! -f "$RESUME_CONFIG" ]]; then
    echo "Checkpoint config not found: $RESUME_CONFIG" >&2
    exit 1
fi

TRAIN_LOG="$(dirname "$OUTPUT_DIR")/train_gr00t_ts02_compact_return_arm_resume_$(date +%Y%m%d_%H%M%S).log"

echo "=========================================="
echo "GR00T N1.7-3B LoRA: TS02 resume"
echo "=========================================="
echo "Dataset: $DATASET_PATH"
echo "Output: $OUTPUT_DIR"
echo "Resume from: $RESUME_CONFIG"
echo "Max steps: $MAX_STEPS"
echo "Log: $TRAIN_LOG"
echo "=========================================="

accelerate launch \
    --num_processes=8 \
    --mixed_precision=bf16 \
    --multi_gpu \
    /home/chensihang/membench/scripts/train/lerobot_train_entry.py \
    \
    --dataset.repo_id="ts02_compact_return_arm_200seeds_v061" \
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
    --resume=true \
    --config_path="$RESUME_CONFIG" \
    --wandb.enable=true \
    --wandb.project="gr00t-n1-7-ts02-compact-return-arm-v061" \
    --use_policy_training_preset=true \
    2>&1 | tee "$TRAIN_LOG"
