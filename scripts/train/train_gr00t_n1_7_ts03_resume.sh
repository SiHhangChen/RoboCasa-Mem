#!/bin/bash
# ============================================================
# GR00T N1.7-3B LoRA 微调 on ts03 —— 断点续训 (8×RTX 4090)
# ============================================================
# 用法:
#   conda activate membench
#   bash scripts/train/train_gr00t_n1_7_ts03_resume.sh
#
# 说明: 从 outputs/gr00t_n1_7_ts03_lora/checkpoints/010000 续训
#   (原训练中断于 ~11840 步, 010000 为最近已保存的 checkpoint)。
#   数据集: 共享盘纯 RGB 版本 (已剔除 depth), 与首次训练完全一致。
#   警告: 续训期间不要再移动/修改共享盘数据, 否则会再次中断!
# ============================================================

set -e

# ==================== 路径配置 ====================
DATASET_PATH="/mnt/shared_workspace/chensihang/dataset/membench/ts03_200seeds_v061"
MODEL_PATH="/home/chensihang/pretrained_models/GR00T-N1.7-3B"
OUTPUT_DIR="./outputs/gr00t_n1_7_ts03_lora"
RESUME_CONFIG="$OUTPUT_DIR/checkpoints/010000/pretrained_model/train_config.json"

# ==================== 训练超参数 ====================
BATCH_SIZE=8           # 每GPU batch size (实测 18.93GB/卡)
NUM_WORKERS=8          # DataLoader 工作进程数
MAX_STEPS=100000        # 总训练步数 (resume 从当前步续到该值)
LR=2e-4                # LoRA 学习率
WARMUP_RATIO=0.03      # warmup 比例
SAVE_EVERY=10000        # 每 N 步保存一次 checkpoint
LOG_EVERY=10           # 每 N 步打印日志

# ==================== LoRA 配置 ====================
PEFT_TARGET_MODULES='["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj","fc1","fc2","to_q","to_k","to_v","to_out.0","net.0.proj","net.2","proj_out_1","proj_out_2"]'
PEFT_FULL_TRAINING_MODULES='["state_encoder","action_encoder","action_decoder"]'
PEFT_RANK=32
PEFT_LORA_ALPHA=64
PEFT_METHOD="lora"

# ==================== 微调控制 ====================
TUNE_LLM=false
TUNE_VISUAL=false
TUNE_PROJECTOR=true
TUNE_DIFFUSION=true
TUNE_VLLN=true

# ==================== 设置 ====================
# 续训: OUTPUT_DIR 已存在, 直接使用; 日志写到父目录
TRAIN_LOG="$(dirname "$OUTPUT_DIR")/train_gr00t_resume_$(date +%Y%m%d_%H%M%S).log"

echo "=========================================="
echo "GR00T N1.7-3B LoRA Training on ts03 (RESUME)"
echo "=========================================="
echo "Dataset: $DATASET_PATH"
echo "Output: $OUTPUT_DIR"
echo "Resume from: $RESUME_CONFIG"
echo "Max steps: $MAX_STEPS"
echo "=========================================="

# ==================== 训练命令 ====================
accelerate launch \
    --num_processes=8 \
    --mixed_precision=bf16 \
    --multi_gpu \
    /home/chensihang/membench/scripts/train/lerobot_train_entry.py \
    \
    --dataset.repo_id="ts03_200seeds_v061" \
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
    --wandb.project="gr00t-n1-7-ts03" \
    --use_policy_training_preset=true \
    2>&1 | tee "$TRAIN_LOG"
