#!/usr/bin/env bash
# Wait for WR03 to finish successfully, then start WR07 on the released GPUs.
#
# Usage:
#   nohup bash scripts/train/run_wr07_v061_scheduled.sh \
#       > outputs/run_wr07_v061_scheduled.log 2>&1 &

set -Eeuo pipefail

REPO_ROOT="/home/chensihang/membench"
cd "$REPO_ROOT"

export PATH="/home/chensihang/miniconda3/bin:$PATH"
source /home/chensihang/miniconda3/etc/profile.d/conda.sh
conda activate membench

WR03_OUTPUT="$REPO_ROOT/outputs/gr00t_n1_7_wr03_v061_lora"
WR03_FINAL_CKPT="$WR03_OUTPUT/checkpoints/100000"
WR03_PATTERN='lerobot_train_entry.py.*wr03_200seeds_v061'
WR07_OUTPUT="$REPO_ROOT/outputs/gr00t_n1_7_wr07_v061_lora"
WR07_PATTERN='lerobot_train_entry.py.*wr07_200seeds_v061'
WR07_SCRIPT="$REPO_ROOT/scripts/train/train_gr00t_n1_7_wr07_v061.sh"

PID_FILE="$REPO_ROOT/outputs/run_wr07_v061_scheduled.pid"
LOCK_FILE="$REPO_ROOT/outputs/run_wr07_v061_scheduled.lock"
STATUS_FILE="$REPO_ROOT/outputs/run_wr07_v061_scheduled.status"

POLL_SEC="${POLL_SEC:-120}"
CHECKPOINT_GRACE_SEC="${CHECKPOINT_GRACE_SEC:-300}"
GPU_SETTLE_SEC="${GPU_SETTLE_SEC:-30}"
GPU_COUNT_REQUIRED="${GPU_COUNT_REQUIRED:-8}"
GPU_USED_LIMIT_MIB="${GPU_USED_LIMIT_MIB:-500}"

log() {
    echo "[$(date '+%F %T')] $*"
}

write_status() {
    printf '%s\n' "$1" > "$STATUS_FILE"
}

mkdir -p "$REPO_ROOT/outputs"
exec 9>"$LOCK_FILE"
if ! flock -n 9; then
    log "错误: 已有 WR07 接续任务持有锁 $LOCK_FILE。"
    exit 1
fi

printf '%s\n' "$$" > "$PID_FILE"
trap 'rm -f "$PID_FILE"' EXIT

[[ -x "$WR07_SCRIPT" ]] || {
    log "错误: WR07 训练脚本不存在或不可执行: $WR07_SCRIPT"
    write_status "failed: WR07 training script is unavailable"
    exit 1
}
[[ ! -e "$WR07_OUTPUT" ]] || {
    log "错误: WR07 输出目录已存在，为避免覆盖而停止: $WR07_OUTPUT"
    write_status "failed: WR07 output directory already exists"
    exit 1
}
if pgrep -f "$WR07_PATTERN" >/dev/null 2>&1; then
    log "错误: 已检测到 WR07 训练进程，为避免重复训练而停止。"
    write_status "failed: WR07 is already running"
    exit 1
fi

write_status "waiting for WR03"
log "接续任务已启动 (PID $$)。"
log "等待 WR03 正常完成: $WR03_FINAL_CKPT"

while pgrep -f "$WR03_PATTERN" >/dev/null 2>&1; do
    if [[ -d "$WR03_FINAL_CKPT" ]]; then
        log "WR03 最终 checkpoint 已出现，等待训练进程完全退出。"
    else
        log "WR03 仍在训练，${POLL_SEC}s 后再次检查。"
    fi
    sleep "$POLL_SEC"
done

if [[ ! -d "$WR03_FINAL_CKPT" ]]; then
    log "WR03 进程已退出，等待最终 checkpoint 写入（最多 ${CHECKPOINT_GRACE_SEC}s）。"
    waited=0
    while [[ ! -d "$WR03_FINAL_CKPT" && "$waited" -lt "$CHECKPOINT_GRACE_SEC" ]]; do
        sleep 10
        waited=$((waited + 10))
    done
fi

if [[ ! -d "$WR03_FINAL_CKPT" ]]; then
    log "错误: WR03 已退出但未生成 100000 checkpoint，判定为异常结束。"
    log "不会启动 WR07，请先检查 WR03 日志和 checkpoint。"
    write_status "failed: WR03 exited without checkpoint 100000"
    exit 1
fi

log "WR03 已正常完成，等待 GPU 全部释放。"
write_status "waiting for GPUs"

while true; do
    if ! gpu_usage="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null)"; then
        log "暂时无法读取 GPU 状态，${POLL_SEC}s 后重试。"
        sleep "$POLL_SEC"
        continue
    fi

    gpu_count="$(awk 'NF {count++} END {print count+0}' <<< "$gpu_usage")"
    busy_count="$(awk -v limit="$GPU_USED_LIMIT_MIB" '$1 > limit {count++} END {print count+0}' <<< "$gpu_usage")"
    if [[ "$gpu_count" -eq "$GPU_COUNT_REQUIRED" && "$busy_count" -eq 0 ]]; then
        log "8 张 GPU 均已释放，等待 ${GPU_SETTLE_SEC}s 后进行二次确认。"
        sleep "$GPU_SETTLE_SEC"
        gpu_usage="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null || true)"
        gpu_count="$(awk 'NF {count++} END {print count+0}' <<< "$gpu_usage")"
        busy_count="$(awk -v limit="$GPU_USED_LIMIT_MIB" '$1 > limit {count++} END {print count+0}' <<< "$gpu_usage")"
        if [[ "$gpu_count" -eq "$GPU_COUNT_REQUIRED" && "$busy_count" -eq 0 ]]; then
            break
        fi
    fi

    log "GPU 尚未全部空闲（检测到 ${gpu_count}/${GPU_COUNT_REQUIRED} 张卡，${busy_count} 张占用超过 ${GPU_USED_LIMIT_MIB} MiB），${POLL_SEC}s 后重试。"
    sleep "$POLL_SEC"
done

if [[ -e "$WR07_OUTPUT" ]] || pgrep -f "$WR07_PATTERN" >/dev/null 2>&1; then
    log "错误: 等待期间 WR07 已由其他方式启动或创建了输出目录，本任务停止以避免重复运行。"
    write_status "failed: WR07 was started elsewhere"
    exit 1
fi

log "GPU 已稳定空闲，开始 WR07 预检和训练。"
write_status "starting WR07"

set +e
bash "$WR07_SCRIPT"
train_status=$?
set -e

if [[ "$train_status" -eq 0 ]]; then
    log "WR07 训练已正常结束。"
    write_status "completed"
else
    log "错误: WR07 训练退出，状态码: $train_status"
    write_status "failed: WR07 exited with status $train_status"
fi
exit "$train_status"
