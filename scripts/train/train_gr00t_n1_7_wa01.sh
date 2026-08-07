#!/bin/bash
# ============================================================
# GR00T N1.7-3B LoRA 微调 on wa01 数据集 (8×RTX 4090)
# ============================================================
# 用法:
#   conda activate membench
#   bash scripts/train/train_gr00t_n1_7_wa01.sh
#
# 依赖: policy/lerobot @ 64b23178d5348609c266250d3e1f511eba4c33ff
#       (注意: 比 README §10.2 固定的 d4a2294 新 299 个上游提交)
#
# 说明: 本脚本所有参数均经过 8 卡实测验证
#   - batch=8/卡, 有效 batch=64, 显存 18.93GB/卡 (24GB 安全)
#   - 吞吐 ~46 smp/s, 50000 步约 19 小时
#   - loss 200 步内从 1.33 降至 0.21
# ============================================================

set -e

# ==================== 路径配置 ====================
DATASET_PATH="/home/chensihang/membench/data/wa01_200seeds_v061"
MODEL_PATH="/home/chensihang/pretrained_models/GR00T-N1.7-3B"
OUTPUT_DIR="./outputs/gr00t_n1_7_wa01_lora"

# ==================== 训练超参数 ====================
BATCH_SIZE=8           # 每GPU batch size (实测 18.93GB/卡)
NUM_WORKERS=8          # DataLoader 工作进程数
MAX_STEPS=100000        # 最大训练步数
LR=2e-4                # LoRA 学习率
WARMUP_RATIO=0.03      # warmup 比例
SAVE_EVERY=10000        # 每 N 步保存一次 checkpoint
LOG_EVERY=10           # 每 N 步打印日志

# ==================== LoRA 配置 ====================
# LoRA 应用于 (r=32, alpha=64):
#   backbone (Qwen3-VL): q/k/v/o_proj (attention), gate/up/down_proj (MLP), fc1/fc2 (SigLIP)
#   DiT (action head):   to_q/to_k/to_v/to_out.0 (attention), net.0.proj/net.2 (FFN), proj_out_1/2
# 全量微调 (full_training_modules):
#   state/action encoder/decoder (CategorySpecificMLP, 自定义层 PEFT 不支持 LoRA)
# 实测: 可训练 379M 参数
PEFT_TARGET_MODULES='["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj","fc1","fc2","to_q","to_k","to_v","to_out.0","net.0.proj","net.2","proj_out_1","proj_out_2"]'
PEFT_FULL_TRAINING_MODULES='["state_encoder","action_encoder","action_decoder"]'
PEFT_RANK=32
PEFT_LORA_ALPHA=64
PEFT_METHOD="lora"

# ==================== 微调控制 ====================
TUNE_LLM=false          # LoRA 接管 backbone 微调
TUNE_VISUAL=false
TUNE_PROJECTOR=true
TUNE_DIFFUSION=true
TUNE_VLLN=true

# ==================== 设置 ====================
# 注意: 不要预先创建 OUTPUT_DIR! lerobot-train 要求输出目录不存在
# (resume=false 时), 目录由训练器自己创建。
mkdir -p "$(dirname "$OUTPUT_DIR")"
TRAIN_LOG="$OUTPUT_DIR/../train_gr00t_$(date +%Y%m%d_%H%M%S).log"

echo "=========================================="
echo "GR00T N1.7-3B LoRA Training on wa01"
echo "=========================================="
echo "Dataset: $DATASET_PATH"
echo "Model: $MODEL_PATH"
echo "Output: $OUTPUT_DIR"
echo "GPUs: 8 × RTX 4090"
echo "Per-GPU batch: $BATCH_SIZE | Effective batch: $((BATCH_SIZE * 8))"
echo "Max steps: $MAX_STEPS | LR: $LR"
echo "PEFT: $PEFT_METHOD r=$PEFT_RANK alpha=$PEFT_LORA_ALPHA"
echo "Full training modules: $PEFT_FULL_TRAINING_MODULES"
echo "=========================================="

# ==================== 训练命令 ====================
# accelerate launch (支持多卡, 不支持 -m 模块方式, 使用入口脚本)
accelerate launch \
    --num_processes=8 \
    --mixed_precision=bf16 \
    --multi_gpu \
    /home/chensihang/membench/scripts/train/lerobot_train_entry.py \
    \
    --dataset.repo_id="wa01_200seeds_v061" \
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
    --wandb.project="gr00t-n1-7-wa01" \
    --use_policy_training_preset=true \
    2>&1 | tee "$TRAIN_LOG"
