# Bench 架构重构状态

本文记录 `docs/bench_architecture_consolidation_plan.md` 的实际迁移落点。

## 已完成

- 冻结 `TaskDefinition`、`ResolvedEpisode`、`EpisodeRuntime`、`ExpertPlan` 和 policy 合同。
- 建立 `bench/tasks/catalog.py`，以 14 个稳定任务 ID 作为唯一发现入口。
- 14 个 canonical leaf package (`tm01` … `wx01`) 具备统一目录和模块出口；每个包都有
  `resources/` 的 sampling/navigation/interaction/scene/assets 占位 schema。
- 建立 `bench/runtime/` 的 RoboCasa 生命周期；录制与评测保存并复用同一
  `ResolvedEpisode`。
- 建立 `bench/execution/` 的专家阶段、导航目标和机械动作合同；评测不会进入该路径。
- 建立纯领域 family：`sugar_count`、`drawer_inspection`、`cabinet_memory`、
  `shelf_return`、`interleaved_task`。
- 建立统一 policy 导入边界 `bench/policy/`，其协议与 backend 不包含任务或 RoboCasa 依赖。
- 建立 `DatasetPlan`、原子 JSON artifact、capture/validation 和 LeRobot exporter 边界。
- 增加架构合同测试：任务布局、episode 稳定性、family 所有权、import 边界和专家短路。

## 已完成的旧实现收口

- 14 个任务的 scene、initialization、observer、语义、expert stages 和任务资源均由 canonical
  leaf package 直接拥有。
- 旧 `new_*`、`physical_*`、`_vendor_*`、`TaskContext`、`RecordingBundle` 和 `TaskSpec` 已完成迁移并删除。
  路径不再作为运行后端。
- 唯一专家录制入口为 `scripts/record_entrypoint.py`，评测使用 canonical evaluation adapter。
- policy 协议与 backend 的唯一实现位于 `bench/policy/`。
- RoboCasa simulator smoke、canonical contract 和 release lock 构成最终 gate。

## 验证

验证统一使用 Conda 环境 `robocasa`。网络 socket 测试仍可能受到运行环境权限限制。
