#!/usr/bin/env bash
# 导出 wa03 unmarked 200 seed → LeRobot 0.6.1 格式 (RGB-only, fps=20)
# 用法: 先 conda activate lerobot061, 然后在仓库根目录执行:
#   bash export_wa03_lerobot.sh
set -euo pipefail

# 定位仓库根目录 (脚本放在仓库根目录下)
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

OUTPUT_ROOT="lerobot_data/wa03_unmarked_200seeds_v061"
REPO_ID="local/wa03_unmarked_200seeds_v061"
EXPECTED=200

# 0) 环境预检
if ! python -c 'import lerobot; assert lerobot.__version__ == "0.6.1", lerobot.__version__' >/dev/null 2>&1; then
  echo "错误: 需要 lerobot061 环境 (lerobot==0.6.1), 请先 conda activate lerobot061"
  exit 1
fi

# 1) 收集 10 个分片下的 episode 目录 (单条 find; 脚本文件内换行安全, 不依赖粘贴)
mapfile -t EPISODES < <(
  find outputs/datasets/WA03/wa03_unmarked_200seeds_part*/train/WA-03 \
    -mindepth 1 -maxdepth 1 -type d -name 'seed_*_attempt_00' \
    | sort
)

echo "episode 数量: ${#EPISODES[@]} / $EXPECTED"
if (( ${#EPISODES[@]} != EXPECTED )); then
  echo "错误: episode 数量不符, 中止 (未做任何写入)"
  exit 1
fi

# 2) 输出目录预检 (create 模式要求目标不存在)
if [[ -e "$OUTPUT_ROOT" ]]; then
  echo "错误: 输出目录已存在, 请先确认是否需删除: $OUTPUT_ROOT"
  exit 1
fi

# 3) 装配参数并执行转换
episode_args=()
for ep in "${EPISODES[@]}"; do
  episode_args+=(--episode-dir "$ep")
done

echo "开始转换 ${#EPISODES[@]} 个 episode → $OUTPUT_ROOT (耗时较长, 请保持终端开启) ..."
MEMBENCH_LEROBOT_SRC="$PWD/policy/lerobot-0.6.1/src" \
python scripts/record/convert_bench_episodes_to_lerobot_v061.py \
  "${episode_args[@]}" \
  --output-root "$OUTPUT_ROOT" \
  --repo-id "$REPO_ID" \
  --fps 20 \
  --robot-type mobile_manipulator \
  --write-mode create

echo "完成: $OUTPUT_ROOT"
echo "校验请执行: wc -l ${OUTPUT_ROOT}/meta/seeds.jsonl   (应输出 200)"