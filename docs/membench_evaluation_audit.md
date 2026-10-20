# membench 实际评测配置审查

数据来源：`/home/chensihang/membench/outputs/eval/`。核对 81 份选定任务的 run manifest：act 10 份、diffusion 17 份、pi05 52 份、replay 2 份。

本报告从运行时记录读取实际 `max_steps`，不按任务上限推算。核对保存的 summary，并检查最近 Pi0.5 完整运行的 450 份 episode result。没有重新运行仿真或修改历史结果。

## 最近完成 50 seeds 的 Pi0.5 配置

“最近”按 manifest 的 `created_at`；每任务 planned/recorded 均为 50、seed 为 1050–1099。此处列出实测配置，尚不替代最终发布基线选择。

| 任务 | 实际 max_steps | replan_steps | 当前对应 suite | publishable |
| --- | --- | --- | --- | --- |
| WX-01 | 1200 | 40 | `configs/eval/suites/wx01_pi05_r40_t1200_v1.yaml` | true |
| WA-01 | 2000 | 40 | `configs/eval/legacy_suites/wa01_gr00t_r40_v1.yaml` | true |
| WA-05 | 2000 | 40 | `configs/eval/legacy_suites/wa05_pi05_r40_t2000_v1.yaml` | true |
| WR-01 | 2600 | 40 | `configs/eval/legacy_suites/wr01_pi05_r40_t2600_v1.yaml` | true |
| WR-08 | 2000 | 40 | `configs/eval/legacy_suites/wr08_gr00t_r40_v1.yaml` | true |
| TS02 | 2800 | 40 | `configs/eval/suites/ts02_pi05_r40_v1.yaml` | true |
| TS03 | 1800 | 40 | `configs/eval/suites/ts03_pi05_r40_v1.yaml` | false |
| TM01 | 3200 | 40 | `configs/eval/suites/tm01_pi05_r40_v1.yaml` | true |
| TM02 | 3600 | 40 | `configs/eval/suites/tm02_pi05_r40_v1.yaml` | true |

表中 TS03 的 `false` 是原始历史 summary 的值。其 seed 1056 的报告字段名已修复，
[修复后汇总](historical_result_repairs/ts03_pi05_r40/summary.json)为 `publishable=true`、
阶段覆盖 50/50，总成功率仍为 14/50（28%）。原始历史文件及本审查 JSON 的原始记录保留，
修复来源另记在 JSON 的 `historical_result_repairs` 中。

上述 9 个当前 suite 的规范化 hash 均与选定 manifest 中记录的 hash 一致。TS02 的 suite 未显式设置 max_steps，但 manifest 已记录当次生效值为 2800。

共同设置：`env_split=pretrain`、`action_repeat=1`、`stop_on_success=true`、`ignore_done=true`；20 Hz；三路 RGB 224×224、state37 / action13、`membench-policy/v2`。模型声明的模态为 RGB、state、prompt，不包含深度。三路视频开启，20 FPS。

对应当前 YAML 的传输设置：WX-01 timeout 300 秒，其余为 30 秒；connect attempts 10、retry delay 2 秒、inference retries 0。早期 manifest 未完整保存 transport，不能仅据此声称已独立核验每次运行的所有传输参数。

## 原始运行依据

- WX-01：[wx01-pi05-s19999-r40-t1200-seedrng-v1](/home/chensihang/membench/outputs/eval/wx01-pi05-s19999-r40-t1200-seedrng-v1/run_manifest.json)，创建于 2026-09-24T05:49:16.426581+00:00。
- WA-01：[wa01-pi05-25000-r40-stability-a](/home/chensihang/membench/outputs/eval/wa01-pi05-25000-r40-stability-a/run_manifest.json)，创建于 2026-09-18T09:56:15.860605+00:00。
- WA-05：[wa05-pi05-3view-full-stride10-s20000-r40-t2000-seedrng-v1](/home/chensihang/membench/outputs/eval/wa05-pi05-3view-full-stride10-s20000-r40-t2000-seedrng-v1/run_manifest.json)，创建于 2026-09-21T07:39:00.432082+00:00。
- WR-01：[wr01-pi05-3view-full-stride10-s20000-r40-t2600-seedrng-v1](/home/chensihang/membench/outputs/eval/wr01-pi05-3view-full-stride10-s20000-r40-t2600-seedrng-v1/run_manifest.json)，创建于 2026-09-19T06:37:15.213233+00:00。
- WR-08：[wr08-pi05-3view-full-stride10-s30000-r40-seedrng-v1](/home/chensihang/membench/outputs/eval/wr08-pi05-3view-full-stride10-s30000-r40-seedrng-v1/run_manifest.json)，创建于 2026-09-21T03:51:08.489047+00:00。
- TS02：[ts02-pi05-3view-full-stride10-s20000-r40-seedrng-v1](/home/chensihang/membench/outputs/eval/ts02-pi05-3view-full-stride10-s20000-r40-seedrng-v1/run_manifest.json)，创建于 2026-09-21T11:06:41.259749+00:00。
- TS03：[ts03-pi05-20000-r40-50seeds-v1](/home/chensihang/membench/outputs/eval/ts03-pi05-20000-r40-50seeds-v1/run_manifest.json)，创建于 2026-09-21T08:33:34.832491+00:00。
- TM01：[tm01-pi05-3view-full-stride10-s30000-r40-seedrng-v1](/home/chensihang/membench/outputs/eval/tm01-pi05-3view-full-stride10-s30000-r40-seedrng-v1/run_manifest.json)，创建于 2026-09-22T09:57:23.728161+00:00。
- TM02：[tm02-pi05-3view-full-stride10-s30000-r40-seedrng-v2](/home/chensihang/membench/outputs/eval/tm02-pi05-3view-full-stride10-s30000-r40-seedrng-v2/run_manifest.json)，创建于 2026-09-24T05:11:31.612936+00:00。

## ACT、Diffusion 与 Pi0.5 的差异

表中为该模型最近完成 50 seeds 的运行，格式为 `max_steps / 客户端请求间隔`。
这里的间隔来自 suite 的 `replan_steps`；对带服务端动作缓存的 DP，它不等于模型
重新生成动作块的间隔。

| 任务 | Pi0.5 | ACT | Diffusion |
| --- | --- | --- | --- |
| WX-01 | 1200 / 40 | 3200 / 50 | 1200 / 1 |
| WA-01 | 2000 / 40 | 5000 / 50 | 5000 / 1 |
| WA-05 | 2000 / 40 | 1600 / 20 | 1600 / 1 |
| WR-01 | 2600 / 40 | 2600 / 40 | 2600 / 1 |
| WR-08 | 2000 / 40 | 2000 / 50 | 2000 / 1 |
| TS02 | 2800 / 40 | 2800 / 50 | 2800 / 1 |
| TS03 | 1800 / 40 | 1800 / 50 | 1800 / 1 |
| TM01 | 3200 / 40 | 3200 / 50 | 3200 / 1 |
| TM02 | 3600 / 40 | 3600 / 50 | 3600 / 1 |

历史 Pi0.5 完整运行还包括 WX-01 的 1200/r50、WA-05 的 1600/r20、TS03 的 1800/r20 和 1800/r50、TM02 的 3600/r30。TM02 的 r50 运行还存在未完成或取消的记录。完整清单及状态见 [JSON 清单](membench_evaluation_audit.json)。

## ACT 的 WA-01 5000 步核验

已进一步核验 `act-wa01-40000` 的 manifest、评测日志、全部 50 个 episode result、
动作 NPZ 和请求 JSONL，而非仅依据静态 suite：

- 每条 `env_steps=5000`、`policy_actions_consumed=5000`；
- 每个 `actions.npz` 的动作数组均为 `(5000, 13)`；
- 每条 `policy_calls=100`，对应 100 行请求记录；
- 每条失败原因为 `max_steps`，日志也明确记录 `steps=5000 calls=100`。

因此这轮 ACT WA-01 实际执行了 5000 个环境控制步。ACT checkpoint 的
`chunk_size=100`、`n_action_steps=100`，服务端调用 `predict_action_chunk()` 返回
100 个动作，客户端按 suite 每次使用前 50 个动作后重新请求。模型配置中的 100
不能替代本轮客户端实际执行的 50。

原始日志：[wa01_eval.log](/home/chensihang/membench/outputs/eval/act-40k-batch-logs/wa01_eval.log)。
实际 handler：[LeRobotACTBackend.infer](/home/chensihang/membench/bench/policy/backends/lerobot.py:146)。

## DP 的 r1 与原生 47 动作缓存

已核验 9 个 DP checkpoint：均为 `n_obs_steps=2`、`horizon=48`、
`n_action_steps=47`。服务端使用 `DiffusionPolicy.select_action()`，协议每次返回
一个动作，因此 suite 的 `replan_steps=1` 用于每步上传连续观测、取回并执行一个动作。

`select_action()` 的实际顺序是：更新观测队列；动作队列为空时才调用
`predict_action_chunk()`；缓存生成的 47 个可执行动作；每次弹出一个动作。
队列非空时继续使用缓存，不重新运行扩散模型。模型新预测发生在 step 0、47、94
等位置，episode reset 会清空两种队列。

对 WA-01 seed 1050，原始请求记录共 5000 行。按上述队列逻辑，5000 步对应 107 次
动作块生成、4893 次缓存动作读取。请求耗时也支持这一解释：队列补充位置的
`model_ms` 中位数约 101.8 ms，其余位置约 0.7 ms；例如 step 47 为 97.7 ms，step
48 为 0.7 ms。`model_ms` 是整个 backend 处理耗时，不是独立的模型 forward 计数。

发布应保留 r1 与服务端原生缓存的组合。直接把 suite 改成 r47 会改变连续观测更新
方式，也不符合协议返回 horizon=1 的合同。不能把每条结果的 `policy_calls=5000`
解释为 5000 次扩散动作块生成。

代码依据：[DP backend](/home/chensihang/membench/bench/policy/backends/lerobot.py:312)、
[DiffusionPolicy.select_action](/home/chensihang/membench/policy/lerobot/src/lerobot/policies/diffusion/modeling_diffusion.py:125)。

## 发布交接进展

- 最近 Pi0.5 使用的 9 个 suite 文件在初次审查时均未被 membench 的 Git 跟踪，当前 checkout 也缺失。现已原样复制这 9 个 suite，以及 9 个 DP suite；ACT 使用的 9 个 suite 原已存在，逐字节核对一致。共 27 个 suite 的规范化 hash 均匹配对应原始运行记录。
- 来源、文件 SHA-256、suite hash、实际运行预算和模型执行方式已记录在 [配置来源清单](../configs/benchmark/evaluation_suite_sources.json)。这份清单用于复现交接，不替代后续正式 release lock。
- TS03 的 r40 运行原始 summary 为 reported_episodes=49、publishable=false。进一步检查发现 seed 1056 的完整阶段报告存于空字段名，已验证其与原始诊断一致并恢复字段名；全部 50 份报告核验后重新汇总为 reported_episodes=50、publishable=true。原始记录保留，发布使用[修复汇总](historical_result_repairs/ts03_pi05_r40/summary.json)，无需重跑模型或仿真。
- 本目录 docs 中还记录 GR00T release15 的其他预算，例如 WR-01 的 t5000 试运行，但对应原始运行目录在此次 outputs 扫描中未找到；文档说明不能与本次已找到的 50-seed manifest 混为同一证据。
- 后续发布规则以明确选定的实际运行及对应 suite 为基线，保留各模型差异的来源与身份；不再通过任务上限选择预算。
