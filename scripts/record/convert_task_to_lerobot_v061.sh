#!/bin/bash
# ============================================================
# 通用: 原生 MemBench episode → LeRobot 0.6.1 数据集
# ============================================================
# 用法:
#   conda activate lerobot061
#   cd ~/chensihang/membench
#   bash scripts/record/convert_task_to_lerobot_v061.sh
#
# 前置:
#   - conda 环境 lerobot061 (包含 policy/lerobot-0.6.1/src)
#   - 原生 episode 目录结构: <shard>/train/<TASK_NAME>/seed_*_attempt_*
#
# 注意:
#   - 转换器 create 模式要求输出根目录不存在 (不要预先创建)
#   - 追加新分片: WRITE_MODE=append (按 seed 去重, 不会重复写入)
#   - 推倒重建: OVERWRITE=1 (create 模式下)
#   - 单目录数据集 (如 WR04) 的 DATASET_SHARDS 只填一个元素
# ============================================================

set -euo pipefail

# ==================== 路径配置 ====================
DATASET_SHARDS=(
  outputs/datasets/WA01/wa01_unmarked_200seeds_part1
  outputs/datasets/WA01/wa01_unmarked_200seeds_part2
  outputs/datasets/WA01/wa01_unmarked_200seeds_part3
  outputs/datasets/WA01/wa01_unmarked_200seeds_part4
  outputs/datasets/WA01/wa01_unmarked_200seeds_part5
)
TASK_NAME="WA-01"          # 任务子目录名 (train/<TASK_NAME>/)
EXPECTED_EPISODES=200      # 0 = 跳过数量校验
OUTPUT_ROOT="lerobot_data/wa01_unmarked_200seeds_v061"
REPO_ID="local/wa01_unmarked_200seeds_v061"

# ==================== 转换参数 ====================
FPS=20
ROBOT_TYPE="mobile_manipulator"
WRITE_MODE="create"        # create | append
OVERWRITE=0                # create 模式下是否强制重建
DEPTH=0                    # 1 = 同时转换深度图, 0 = 仅转换 RGB (默认)

# ==================== 收集 episode ====================
episode_dirs=()
for shard in "${DATASET_SHARDS[@]}"; do
  task_dir="$shard/train/$TASK_NAME"
  if [[ ! -d "$task_dir" ]]; then
    echo "错误: 任务目录不存在: $task_dir"
    exit 1
  fi
  while IFS= read -r -d '' episode_dir; do
    episode_dirs+=("$episode_dir")
  done < <(
    find "$task_dir" \
      -mindepth 1 -maxdepth 1 -type d \
      -name 'seed_*_attempt_*' -print0 |
      sort -z
  )
done

# ==================== 数量校验 ====================
if (( EXPECTED_EPISODES > 0 )) && (( ${#episode_dirs[@]} != EXPECTED_EPISODES )); then
  echo "错误: 预期 ${EXPECTED_EPISODES} 个 episode, 实际找到 ${#episode_dirs[@]} 个"
  exit 1
fi

echo "找到 ${#episode_dirs[@]} 个 ${TASK_NAME} episode, 开始转换……"

# ==================== 输出目录预检 ====================
overwrite_arg=()
if [[ "$WRITE_MODE" == "create" ]] && [[ -e "$OUTPUT_ROOT" ]]; then
  if (( OVERWRITE )); then
    echo "警告: 输出目录已存在, 使用 --overwrite 重建: $OUTPUT_ROOT"
    overwrite_arg=(--overwrite)
  else
    echo "错误: 输出目录已存在, 请删除后重试, 或改用 WRITE_MODE=append 追加: $OUTPUT_ROOT"
    exit 1
  fi
fi

# ==================== 转换 ====================
episode_args=()
for episode_dir in "${episode_dirs[@]}"; do
  episode_args+=(--episode-dir "$episode_dir")
done

depth_arg=()
if (( DEPTH )); then
  depth_arg=(--depth)
fi

MEMBENCH_LEROBOT_SRC="$PWD/policy/lerobot-0.6.1/src" \
python scripts/record/convert_bench_episodes_to_lerobot_v061.py \
  "${episode_args[@]}" \
  --output-root "$OUTPUT_ROOT" \
  --repo-id "$REPO_ID" \
  --fps "$FPS" \
  --robot-type "$ROBOT_TYPE" \
  --write-mode "$WRITE_MODE" \
  "${depth_arg[@]}" \
  "${overwrite_arg[@]}"
