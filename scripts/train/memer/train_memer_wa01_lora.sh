#!/usr/bin/env bash
set -euo pipefail

# Single RTX 4090 recipe. This trains LoRA adapters on the language model;
# the Qwen vision tower and multimodal projector remain frozen.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
QWEN_ROOT="${QWEN_ROOT:-${ROOT}/policy/Qwen3-VL/qwen-vl-finetune}"
DATA_ROOT="${DATA_ROOT:-${ROOT}/outputs/memer/wa01_sft_50_stride10_v2}"
MODEL_PATH="${MODEL_PATH:-${ROOT}/ckpts/Qwen3-VL-4B-Instruct}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT}/outputs/memer/wa01_qwen3vl_4b_lora}"

: "${CUDA_VISIBLE_DEVICES:=0}"
: "${MEMER_QWEN_ENV:=memer-qwen}"
: "${MAX_STEPS:=1500}"
: "${BATCH_SIZE:=1}"
: "${GRAD_ACCUM:=64}"
: "${DATALOADER_WORKERS:=2}"

test -f "${DATA_ROOT}/train.json"
test -d "${DATA_ROOT}/media"
test -f "${MODEL_PATH}/config.json"
test -f "${QWEN_ROOT}/qwenvl/train/train_qwen.py"

export MEMER_WA01_ANNOTATION_PATH="${DATA_ROOT}/train.json"
export MEMER_WA01_DATA_PATH="${DATA_ROOT}"
export QWEN_VL_ATTN_IMPL=flash_attention_2
export PYTHONUNBUFFERED=1

mkdir -p "${OUTPUT_ROOT}"
cd "${QWEN_ROOT}"

conda run --no-capture-output -n "${MEMER_QWEN_ENV}" \
  env CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
  MEMER_WA01_ANNOTATION_PATH="${MEMER_WA01_ANNOTATION_PATH}" \
  MEMER_WA01_DATA_PATH="${MEMER_WA01_DATA_PATH}" \
  QWEN_VL_ATTN_IMPL="${QWEN_VL_ATTN_IMPL}" \
  PYTHONUNBUFFERED="${PYTHONUNBUFFERED}" \
  torchrun --standalone --nproc_per_node=1 \
    qwenvl/train/train_qwen.py \
    --model_name_or_path "${MODEL_PATH}" \
    --dataset_use memer_wa01 \
    --data_flatten True \
    --tune_mm_vision False \
    --tune_mm_mlp False \
    --tune_mm_llm True \
    --lora_enable True \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_dropout 0.05 \
    --bf16 \
    --output_dir "${OUTPUT_ROOT}" \
    --max_steps "${MAX_STEPS}" \
    --per_device_train_batch_size "${BATCH_SIZE}" \
    --gradient_accumulation_steps "${GRAD_ACCUM}" \
    --optim adamw_torch \
    --max_pixels 115200 \
    --min_pixels 50176 \
    --eval_strategy no \
    --save_strategy steps \
    --save_steps 250 \
    --save_total_limit 2 \
    --learning_rate 6e-5 \
    --weight_decay 0 \
    --warmup_ratio 0.05 \
    --max_grad_norm 1 \
    --lr_scheduler_type cosine \
    --logging_steps 1 \
    --logging_nan_inf_filter False \
    --model_max_length 8192 \
    --gradient_checkpointing True \
    --dataloader_num_workers "${DATALOADER_WORKERS}" \
    --report_to none
