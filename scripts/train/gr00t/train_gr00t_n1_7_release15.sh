#!/usr/bin/env bash
# Joint GR00T N1.7-3B LoRA training for the 15-task MemBench release dataset.
#
# Usage:
#   conda activate membench
#   python scripts/train/gr00t/prepare_joint_dataset.py aggregate
#   bash scripts/train/gr00t/train_gr00t_n1_7_release15.sh --check-only
#   bash scripts/train/gr00t/train_gr00t_n1_7_release15.sh
#   bash scripts/train/gr00t/train_gr00t_n1_7_release15.sh --resume 250000

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
            echo "Checkpoint step must be six digits, for example 500000." >&2
            exit 2
        }
        ;;
    *)
        echo "Usage: $0 [--check-only | --resume CHECKPOINT_STEP]" >&2
        exit 2
        ;;
esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"

DATASET_REPO_ID="${DATASET_REPO_ID:-membench_release15_v1}"
DATASET_PATH="${DATASET_PATH:-$REPO_ROOT/data/$DATASET_REPO_ID}"
DATA_ROOT="${DATA_ROOT:-$REPO_ROOT/data}"
MANIFEST="${MANIFEST:-$REPO_ROOT/configs/train/gr00t/release15_v1.json}"
MODEL_PATH="${MODEL_PATH:-/home/chensihang/pretrained_models/GR00T-N1.7-3B}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/outputs/gr00t_n1_7_release15_v1_lora}"
TRAIN_ENTRY="${TRAIN_ENTRY:-$REPO_ROOT/scripts/train/gr00t/lerobot_train_entry.py}"
DATASET_TOOL="${DATASET_TOOL:-$REPO_ROOT/scripts/train/gr00t/prepare_joint_dataset.py}"
PYTHON_BIN="${PYTHON_BIN:-python}"
ACCELERATE_BIN="${ACCELERATE_BIN:-accelerate}"

export HF_HOME="${HF_HOME:-$REPO_ROOT/outputs/.cache/huggingface}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$REPO_ROOT/outputs/.cache/xdg}"
mkdir -p "$HF_HOME" "$XDG_CACHE_HOME"

# This matches the existing 8 x RTX 4090 LoRA recipe. The 500k baseline is
# approximately 5.9 passes over all valid 40-step training windows.
NUM_GPUS="${NUM_GPUS:-8}"
BATCH_SIZE="${BATCH_SIZE:-8}"
NUM_WORKERS="${NUM_WORKERS:-8}"
MAX_STEPS="${MAX_STEPS:-200000}"
LR="${LR:-2e-4}"
WARMUP_RATIO="${WARMUP_RATIO:-0.03}"
SAVE_EVERY="${SAVE_EVERY:-20000}"
LOG_EVERY="${LOG_EVERY:-10}"
TRAIN_SEED="${TRAIN_SEED:-1000}"
MIN_FREE_GPU_GIB="${MIN_FREE_GPU_GIB:-21}"
MIN_FREE_DISK_GIB="${MIN_FREE_DISK_GIB:-100}"
WANDB_ENABLE="${WANDB_ENABLE:-true}"
WANDB_PROJECT="${WANDB_PROJECT:-gr00t-n1-7-membench-release15}"
SKIP_GPU_CHECK="${SKIP_GPU_CHECK:-false}"
SKIP_DECODE_CHECK="${SKIP_DECODE_CHECK:-false}"

PEFT_TARGET_MODULES='["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj",'
PEFT_TARGET_MODULES+='"fc1","fc2","to_q","to_k","to_v","to_out.0","net.0.proj","net.2",'
PEFT_TARGET_MODULES+='"proj_out_1","proj_out_2"]'
PEFT_FULL_TRAINING_MODULES='["state_encoder","action_encoder","action_decoder"]'
PEFT_RANK="${PEFT_RANK:-32}"
PEFT_LORA_ALPHA="${PEFT_LORA_ALPHA:-64}"

fail() {
    echo "Preflight failed: $*" >&2
    exit 1
}

command -v "$PYTHON_BIN" >/dev/null 2>&1 || fail "Python executable not found: $PYTHON_BIN"
command -v "$ACCELERATE_BIN" >/dev/null 2>&1 || fail "Accelerate not found; activate membench."
[[ -f "$MANIFEST" ]] || fail "joint-training manifest not found: $MANIFEST"
[[ -f "$DATASET_TOOL" ]] || fail "joint dataset tool not found: $DATASET_TOOL"
[[ -f "$TRAIN_ENTRY" ]] || fail "training entrypoint not found: $TRAIN_ENTRY"
[[ -f "$DATASET_PATH/meta/info.json" ]] || {
    fail "aggregate dataset is missing; run: $PYTHON_BIN $DATASET_TOOL aggregate"
}
[[ -s "$MODEL_PATH/config.json" ]] || fail "base model config not found: $MODEL_PATH/config.json"
[[ -s "$MODEL_PATH/model.safetensors.index.json" ]] || fail "base model weight index is missing"
[[ -s "$MODEL_PATH/model-00001-of-00002.safetensors" ]] || fail "base model shard 1 is missing"
[[ -s "$MODEL_PATH/model-00002-of-00002.safetensors" ]] || fail "base model shard 2 is missing"

if [[ "$MODE" == "resume" ]]; then
    RESUME_CONFIG="$OUTPUT_DIR/checkpoints/$CHECKPOINT_STEP/pretrained_model/train_config.json"
    [[ -f "$RESUME_CONFIG" ]] || fail "checkpoint config not found: $RESUME_CONFIG"
    RESUME_ARGS=(--resume=true "--config_path=$RESUME_CONFIG")
elif [[ "$MODE" == "train" ]]; then
    [[ ! -e "$OUTPUT_DIR" ]] || fail "output already exists: $OUTPUT_DIR (use --resume)"
    RESUME_ARGS=(--resume=false)
else
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
print(
    f"LeRobot: {version('lerobot')} | PyTorch: {version('torch')} | "
    f"Accelerate: {version('accelerate')}"
)
PY

VALIDATE_ARGS=(
    validate
    --manifest "$MANIFEST"
    --data-root "$DATA_ROOT"
    --dataset-root "$DATASET_PATH"
)
if [[ "$SKIP_DECODE_CHECK" != "true" ]]; then
    VALIDATE_ARGS+=(--decode-samples)
fi
"$PYTHON_BIN" "$DATASET_TOOL" "${VALIDATE_ARGS[@]}"

mkdir -p "$(dirname "$OUTPUT_DIR")"
AVAILABLE_DISK_KIB="$(df -Pk "$(dirname "$OUTPUT_DIR")" | awk 'NR == 2 {print $4}')"
REQUIRED_DISK_KIB="$((MIN_FREE_DISK_GIB * 1024 * 1024))"
if (( AVAILABLE_DISK_KIB < REQUIRED_DISK_KIB )); then
    fail "less than ${MIN_FREE_DISK_GIB} GiB is free under $(dirname "$OUTPUT_DIR")"
fi

if [[ "$SKIP_GPU_CHECK" != "true" ]]; then
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
fi

if [[ "$MODE" == "check" ]]; then
    echo "Preflight passed. No training was started."
    exit 0
fi

TRAIN_LOG_DIR="$REPO_ROOT/outputs/train/gr00t"
mkdir -p "$TRAIN_LOG_DIR"
if [[ "$MODE" == "resume" ]]; then
    TRAIN_LOG="$TRAIN_LOG_DIR/release15_v1_resume_$(date +%Y%m%d_%H%M%S).log"
else
    TRAIN_LOG="$TRAIN_LOG_DIR/release15_v1_$(date +%Y%m%d_%H%M%S).log"
fi

echo "=========================================="
echo "GR00T N1.7-3B LoRA: MemBench release15 v1"
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
echo "Max steps: $MAX_STEPS | LR: $LR | Seed: $TRAIN_SEED"
echo "Sampling: frame-uniform | Save every: $SAVE_EVERY"
echo "=========================================="

"$ACCELERATE_BIN" launch \
    --num_processes="$NUM_GPUS" \
    --mixed_precision=bf16 \
    --multi_gpu \
    "$TRAIN_ENTRY" \
    \
    --dataset.repo_id="$DATASET_REPO_ID" \
    --dataset.root="$DATASET_PATH" \
    --dataset.video_backend="pyav" \
    --dataset.use_imagenet_stats=false \
    --dataset.eval_split=0.0 \
    \
    --policy.type="groot" \
    --policy.base_model_path="$MODEL_PATH" \
    --policy.pretrained_path="$MODEL_PATH" \
    --policy.embodiment_tag="robocasa_panda_omron" \
    --policy.tune_llm=false \
    --policy.tune_visual=false \
    --policy.tune_projector=true \
    --policy.tune_diffusion_model=true \
    --policy.tune_vlln=true \
    --policy.max_state_dim=132 \
    --policy.max_action_dim=132 \
    --policy.chunk_size=40 \
    --policy.n_action_steps=40 \
    --policy.use_bf16=true \
    --policy.model_params_fp32=false \
    --policy.use_flash_attention=true \
    --policy.optimizer_lr="$LR" \
    --policy.warmup_ratio="$WARMUP_RATIO" \
    --policy.max_steps="$MAX_STEPS" \
    --policy.push_to_hub=false \
    \
    --peft.method_type="lora" \
    --peft.target_modules="$PEFT_TARGET_MODULES" \
    --peft.full_training_modules="$PEFT_FULL_TRAINING_MODULES" \
    --peft.r="$PEFT_RANK" \
    --peft.lora_alpha="$PEFT_LORA_ALPHA" \
    \
    --job_name="membench-release15-v1" \
    --output_dir="$OUTPUT_DIR" \
    --seed="$TRAIN_SEED" \
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
