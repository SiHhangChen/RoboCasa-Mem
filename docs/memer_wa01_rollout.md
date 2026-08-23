# WA-01 MemER 完整 Rollout

WA-01 的 MemER checkpoint 是高层记忆策略，只输出当前 subtask；动作由
OpenPI pi05 checkpoint 输出。因此完整闭环使用 `serve_memer_wa01.py` 组合服务：

```text
RoboCasa observation
        |
        v
MemER Qwen3-VL: recent frames + episodic memory -> current_subtask
        |
        v
OpenPI pi05: current_subtask + state + RGB -> 50-step action chunk
        |
        v
RoboCasa action execution
```

## 环境要求

- Python 3.11；使用 `policy/openpi/.venv/bin/python`，不要使用系统 Python 3.9。
- 可用 CUDA GPU 和驱动。MemER Qwen3-VL checkpoint 约 8.8GB，OpenPI checkpoint
  约 9GB，建议至少 24GB 显存或使用模型并行方案。
- OpenPI venv 中固定安装：

```bash
uv pip install --python policy/openpi/.venv/bin/python \
  'transformers>=4.57.0,<5' accelerate
```

当前 checkpoint 的 Qwen3-VL processor 可由本地目录直接读取，不需要联网下载
processor。若使用 LoRA adapter，再额外安装 `peft`。

## 先做协议 preflight

终端 1 启动组合策略服务：

```bash
cd /home/chensihang/membench
CUDA_VISIBLE_DEVICES=6,7 ./scripts/eval/eval_memer_wa01.sh serve
```

服务默认使用：

- MemER：`ckpts/memer-qwenvlm/wa01-50-stride10-v2-qwen3vl-4b`
- OpenPI：`ckpts/memer-pi05/wa01/20000`
- OpenPI config：`memer_wa01`（使用 per-frame subtask prompt 训练合同）
- norm stats：checkpoint 内的 `assets/wa01_200seeds_v061_subtasks_v2`
- MemER query stride：每 10 个物理环境步一次

终端 2 只验证协议、相机、state/action schema 和 action horizon：

```bash
./scripts/eval/eval_memer_wa01.sh preflight
```

## 单 seed 闭环

```bash
SUITE=configs/eval/suites/wa01_memer_full_v1.yaml \
RUN_ID=wa01-memer-full-20000-seed1050 \
./scripts/eval/eval_memer_wa01.sh eval

./scripts/eval/eval_memer_wa01.sh summary
```

单 seed 通过后再启动 50 seeds：

```bash
SUITE=configs/eval/suites/wa01_memer_full_50seeds_v1.yaml \
RUN_ID=wa01-memer-full-20000-50seeds \
./scripts/eval/eval_memer_wa01.sh eval
```

每个 episode 的 request metrics 会记录 MemER latency、subtask index、JSON
解析状态、memory 数量和 fallback 状态。分析时先区分：

1. MemER 输出不在 `configs/memer/wa01_subtasks.json` 的 label 集合中；
2. MemER 解析失败并发生 fallback；
3. subtask 正确但 OpenPI/环境没有完成对应阶段。

`replan_steps: 10` 是刻意设置的：runner 每 10 个环境步重新请求一次组合策略，
而 OpenPI 仍返回 horizon=50 的 action chunk，队列只消费前 10 步后重新规划。
