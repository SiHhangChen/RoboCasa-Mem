#!/bin/bash
# ============================================================
# 看门狗: ts02 v061 训练完成后, 自动启动 wr06 v061 训练
# ============================================================
# 供后台常驻 (nohup) 运行, 事件驱动 (等待训练真正完成), 而非 at 定时。
#
# 完成判据 (缺一不可):
#   1. ts02 训练进程 (python lerobot_train_entry.py + dataset.root=ts02) 已退出
#   2. 最终 checkpoint outputs/gr00t_n1_7_ts02_v061_lora/checkpoints/100000/ 已出现
#      (lerobot 保证最终步必保存, 见 should_save_checkpoint)
#
# 若 ts02 进程消失但未产出最终 checkpoint (异常退出), 视为失败:
#   记录警告并退出, 不启动 wr06 (留待人工排查)。
#
# 用法:
#   nohup bash scripts/train/run_wr06_v061_scheduled.sh \
#       > outputs/run_wr06_v061_scheduled.log 2>&1 &
# ============================================================

set -e

cd /home/chensihang/membench

# 显式初始化 conda (后台 nohup 环境 PATH 精简, 不会自动激活 conda)
export PATH="/home/chensihang/miniconda3/bin:$PATH"
source /home/chensihang/miniconda3/etc/profile.d/conda.sh
conda activate membench

TS02_OUTPUT="./outputs/gr00t_n1_7_ts02_v061_lora"
FINAL_CKPT="$TS02_OUTPUT/checkpoints/100000"
# 匹配 ts02 训练进程: python lerobot_train_entry.py 且 dataset.root 指向 ts02
TS02_PATTERN="lerobot_train_entry.py.*data/ts02_200seeds_v061"
WR06_SCRIPT="scripts/train/train_gr00t_n1_7_wr06_v061.sh"
LAUNCH_LOG="outputs/gr00t_n1_7_wr06_v061_launch.log"
POLL_SEC=120            # 轮询间隔
GRACE_SEC=60            # 进程消失后等待最终 checkpoint 写入的宽限时间

log() { echo "[$(date '+%F %T')] $*"; }

log "看门狗启动, 等待 ts02 训练完成 (最终 checkpoint: $FINAL_CKPT)..."

# 完成判据 (缺一不可):
#   1. ts02 训练进程已完全退出 (进程消失, 否则 checkpoint 保存后 GPU 尚未释放, wr06 会抢跑 OOM)
#   2. 最终 checkpoint 已生成
# 任一条件不满足则继续轮询。
while pgrep -f "$TS02_PATTERN" > /dev/null 2>&1 || [ ! -d "$FINAL_CKPT" ]; do
    if ! pgrep -f "$TS02_PATTERN" > /dev/null 2>&1 && [ ! -d "$FINAL_CKPT" ]; then
        # ts02 进程已退出, 但最终 checkpoint 尚未出现 -> 可能是正在写盘或异常退出
        log "检测到 ts02 训练进程已退出, 等待最终 checkpoint 写入 (宽限 ${GRACE_SEC}s)..."
        sleep "$GRACE_SEC"
        if ! pgrep -f "$TS02_PATTERN" > /dev/null 2>&1 && [ ! -d "$FINAL_CKPT" ]; then
            log "错误: ts02 训练进程已退出但最终 checkpoint 未生成, 判定为异常结束。"
            log "不启动 wr06, 请人工排查 $TS02_OUTPUT。"
            exit 1
        fi
    fi
    sleep "$POLL_SEC"
done

log "ts02 训练完成 (进程已退出 + 最终 checkpoint 已生成), 启动 wr06 训练..."

# 二次兜底: 确认 8 张 GPU 显存均已释放 (仅系统占用), 防止其他残留进程占卡
log "确认 GPU 显存已释放..."
GPU_BUSY=1
for _ in 1 2 3 4 5; do
    BUSY_COUNT=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1 > 500 {c++} END {print c+0}')
    if [ "$BUSY_COUNT" -eq 0 ]; then
        GPU_BUSY=0
        break
    fi
    log "仍有 $BUSY_COUNT 张卡显存占用 (used>500MiB), ${POLL_SEC}s 后重试..."
    sleep "$POLL_SEC"
done
if [ "$GPU_BUSY" -eq 1 ]; then
    log "错误: 等待后 GPU 仍被占用, 不启动 wr06, 请人工排查显存占用。"
    nvidia-smi --query-gpu=index,memory.used --format=csv
    exit 1
fi
log "GPU 显存已全部释放。"

# 后台启动 wr06 (nohup 脱离看门狗进程组, 训练可持续 ~38 小时)
nohup bash "$WR06_SCRIPT" > "$LAUNCH_LOG" 2>&1 &

WR06_PID=$!
log "wr06 训练已启动 (nohup PID $WR06_PID), 日志: $LAUNCH_LOG"
log "看门狗结束。"
