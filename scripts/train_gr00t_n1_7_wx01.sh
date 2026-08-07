#!/usr/bin/env bash
# Train GR00T N1.7 (LeRobot 0.6.1) on MemBench wx01 dataset.
# Usage:
#   bash scripts/train_gr00t_n1_7_wx01.sh          # smoke run: 200 steps
#   bash scripts/train_gr00t_n1_7_wx01.sh full     # full run: 5000 steps
set -euo pipefail

LEROBOT_ENV=/home/user/chensihang/membench/policy/lerobot-0.6.1/.venv
TRAIN="$LEROBOT_ENV/bin/lerobot-train"

MODE="${1:-smoke}"
if [ "$MODE" = "full" ]; then
    STEPS=5000
    SAVE_FREQ=500
    BATCH=1
else
    STEPS=200
    SAVE_FREQ=100
    BATCH=1
fi

# base model path: local dir to avoid re-downloading every run
BASE_MODEL=/home/user/models/gr00t-n1-7-3b
if [ ! -d "$BASE_MODEL" ]; then
    echo "base model not found: $BASE_MODEL (run huggingface-cli download first)" >&2
    exit 1
fi

OUTPUT_DIR=outputs/train/gr00t_n1_7_wx01_${MODE}

"$TRAIN" \
  --dataset.root=/home/user/chensihang/membench/lerobot_data \
  --dataset.repo_id=wx01_200seeds_v061 \
  --dataset.image_transforms.enable=true \
  --policy.type=groot \
  --policy.device=cuda \
  --policy.base_model_path="$BASE_MODEL" \
  --policy.embodiment_tag=new_embodiment \
  --policy.chunk_size=16 \
  --policy.n_action_steps=16 \
  --policy.use_relative_actions=true \
  --policy.relative_exclude_joints='["gripper"]' \
  --policy.use_bf16=true \
  --policy.push_to_hub=false \
  --seed=42 \
  --batch_size="$BATCH" \
  --steps="$STEPS" \
  --save_checkpoint=true \
  --save_freq="$SAVE_FREQ" \
  --env_eval_freq=0 \
  --eval_steps=0 \
  --log_freq=10 \
  --output_dir="$OUTPUT_DIR" \
  --job_name="gr00t_n1_7_wx01_${MODE}"
