# WR05 搬运与 WR06 简化任务实施计划

状态：已实施（静态、单元与模拟器物理验证完成；WR-06 四种 (层, plate 侧) 组合物理 rollout 全部通过）  
实施日期：2026-08-11  
源仓库：`/home/user/chensihang/MobileMemBench`  
目标仓库：`/home/user/chensihang/membench`

## 1. 目标

本次工作包含两个交付项：

1. 将源仓库的 `bench/tasks/new_WR05` 搬运为目标仓库的 canonical `bench/tasks/wr05`。
2. 将 WR05 简化为单物体任务 WR06，其与 WR05 的关系应对应当前仓库中 WR02 与 WR04 的关系。

这里的“搬运”只搬运任务行为、场景定义、物理参数、标定资源和已经验证的操作策略。目标代码必须遵循当前仓库的 task package v1、canonical runtime、ExpertPlan 和插件发现架构，不照搬源仓库的 legacy `new_*` 目录结构。

## 2. 已确认的现状与约束

### 2.1 源 WR05 的状态

- 源 WR05 是 `new_WR05`，当前在源仓库中尚未被 Git 跟踪。
- 源 WR05 的 smoke suite 和架构测试同样尚未被 Git 跟踪。
- 源 WR05 依赖源仓库若干未提交的通用导航和机械臂改动，包括支撑面下降提前停止、释放前检查和更严格的停靠容差等能力。
- 因此迁移范围必须根据实际运行依赖闭包确定，不能只复制 `new_WR05` 目录或只读取源仓库已提交内容。

### 2.2 目标仓库的架构约束

- 新任务以 `bench/tasks/_template` 为必需骨架，并提供 task package v1 的全部稳定出口和占位资源。
- `task.py` 是任务组合入口；episode 解析、runtime descriptor、专家计划、环境和语义各自位于 canonical 模块中。
- 一个叶子任务不得导入另一个叶子任务。`wr05` 不得导入 `wr02`，`wr06` 不得导入 `wr04` 或 `wr05`。
- 任务资源必须本地持有。禁止通过 YAML `inherits` 或相对路径引用其他任务的配置、station、calibration、sampling、scene 或 interaction 资源。
- 共享能力只允许进入现有的 `bench/core`、`bench/runtime`、`bench/execution` 或纯领域 family，并且不得包含任务 ID、瓶子、餐盘等具体任务语义。
- catalog 和 plugin 的发现过程必须保持 lazy，不得在注册任务描述符时加载 RoboCasa 或专家执行模块。

### 2.3 工作树保护

目标仓库当前已有多处未提交修改，其中包括 WR04、通用 execution、评测配置和 release lock。实施前必须：

1. 保存源、目标仓库的 `git status` 和相关文件 diff 清单。
2. 对需要修改的现有文件逐个检查重叠区域。
3. 只做局部补丁，不用源文件整体覆盖目标文件。
4. release lock 在任务和资源稳定后统一处理，不在开发中反复重写。

## 3. 源架构到目标架构的映射

| 源 WR05 组成 | 目标仓库落点 | 迁移规则 |
|---|---|---|
| `bundle.py` | `task.py`、catalog、plugin | 不搬 legacy bundle 和 recording 注册逻辑 |
| `workflow.py` | `expert/spec.py` | 将旧 `TaskSpec` stages 改为 canonical `ExpertPlan` |
| 旧 `runtime.py` / `TaskContext` | `episode.py`、新 `runtime.py` | 使用 `ResolvedEpisode` 和 `RuntimeDescriptor` |
| 旧 `bench.model` / `bench.stages` | 当前 execution contracts | 禁止引入 legacy compatibility 层 |
| 旧 navigation/manipulation/planning | `bench/execution/*` 和 task-local execution | 优先复用目标已有合同，只补真正缺失的通用能力 |
| `new_WR05/environment.py` | `wr05/environment.py` | 使用当前环境描述符和 canonical runtime 生命周期 |
| `new_WR05/domain/*` | `wr05/domain/*` 或现有纯领域 family | 不得从 `wr02` 叶子包继承或导入 |
| `inherits: ../../new_WR02/...` | `wr05/resources/task.yaml` | 完全物化为独立配置 |
| WR02 calibration/station 引用 | WR05 本地资源 | 复制后更新 task/env 身份并校验内容 |
| 源 WR05 测试 | 当前 contract/unit/simulator 测试 | 按目标 API 重写，不保留 legacy API 断言 |

## 4. WR05 的目标规格

WR05 保留源任务的完整双物体关系记忆行为：

- 物体：真实 ketchup 瓶和 mustard 瓶；模拟器对象名保持 `ketchup`、`mustard`。
- 初始状态：两个瓶子分别位于开放货架的上下层，episode 覆盖平衡的层级和 plate slot 配对。
- 中间阶段：两个瓶子分别暂存于固定 serving plate 的 `plate_left` 和 `plate_right` 区域。
- 返回阶段：从餐盘旁重新抓取两瓶并返回其原始货架层级。
- 公开 prompt 不泄漏 condiment 到 shelf level 的私有绑定。
- 成功必须由物理状态和最终关系语义共同决定，不以专家 stage 完成代替任务成功。

需要保留并验证的源 WR05 特性包括：

- 真实瓶模型和固定餐盘场景。
- plate-left/right 独立放置站位和独立拾取站位。
- WA02 风格的低位侧抓候选及 yaw 选择。
- 根据 live EEF-to-object attachment 计算货架返回位姿。
- 放置下降的支撑面提前停止。
- 释放前的高度、XY、直立和支撑检查。
- 释放后的垂直 clearance、稳定时间和防分离干扰策略。
- 货架返回时的直立瓶体四元数和自然释放。

## 5. WR06 的目标规格

WR06 是 WR05 的单物体简化版：去掉 mustard，ketchup 独立覆盖层级与 plate 侧两个维度。

| 维度 | WR05 | WR06 |
|---|---|---|
| 物体 | ketchup + mustard | 固定真实 ketchup 瓶 |
| 初始层 | 双物体四种平衡配对 | 偶数 seed 在下层，奇数 seed 在上层 |
| 暂存点 | plate-left + plate-right | 随 seed 在 plate-left / plate-right 间交替（偶 placement-pair 在左，奇在右），四种 (层, 侧) 组合平衡覆盖 WR05 全部配对 |
| 记忆目标 | 恢复双瓶层级关系 | 将同一瓶返回原始层 |
| 专家流程 | 双物体完整往返 | 单物体观察、暂存、再拾取和返回 |
| 语义协议 | 双物体关系成功条件 | WR04 风格的严格单物体记忆状态机 |

WR06 的关键协议为：

1. reset 后从模拟器物理状态确认初始 shelf level。
2. 机器人观察货架，抓取固定 ketchup 瓶并将其放到 episode 决定的 plate 侧。
3. 只有瓶子稳定到达餐盘暂存区、离开货架并获得支撑后，才能锁存 memory barrier。
4. 机器人重新抓取同一瓶并返回原始 shelf level。
5. memory barrier 前提前返回、首次尝试错误层、返回错误物体、暂存回退或终态回退均为协议失败。
6. 返回后必须经过连续物理 dwell/terminal verification 才能成功。
7. public prompt 不公开 seed、原始层或 plate 侧，oracle metadata 仅用于录制审计和评测诊断。

ketchup 在右槽位的停靠站使用 WR05 已验证的 `ketchup_plate_right_station`
（镜像左槽位 base 几何以保持当前 IK branch），不用 mustard 的
`plate_right_station`；左右拾取分别使用 `plate_left_pick_station` /
`plate_right_pick_station`。

WR06 复用的是 WR04 的设计方法和语义协议，不复用 WR04 的叶子包代码；其真实瓶、餐盘、抓放和站位资源由 `wr06` 独立持有。

## 6. 分阶段实施计划

### Phase 0：冻结行为和冲突基线

- 枚举源 `new_WR05`、WR05 suite、测试和所有未提交依赖。
- 为源文件建立内容 hash 或快照清单，保证迁移期间来源可追溯。
- 记录目标工作树中与本任务重叠的修改。
- 建立“源行为 -> 目标模块 -> 验证用例”的迁移矩阵。

完成条件：源 WR05 的任务内文件和任务外依赖均有明确处理结论。

### Phase 1：建立 canonical 任务骨架

- 从 `bench/tasks/_template` 完整创建 `wr05` 和 `wr06` 必需文件与资源骨架。
- 对照当前 `wr02` 和 `wr04` 补齐当前同族任务使用的 `domain/`、task-local `execution/`、`runtime.py` 等 canonical 实现目录。
- 保持所有稳定出口：`TASK`、`load_task_config()`、`resolve_episode()`、`build_environment_descriptor()`、`initialize_episode()`、`observe_task_state()`、`evaluate_task_state()`、`resolve_prompt()` 和四个 expert 出口。

完成条件：两个空任务包先通过 package layout 检查，且没有 legacy 或跨叶子 import。

### Phase 2：迁移缺失的通用能力

- 对照源仓库未提交的 navigation/manipulation 改动与目标 `bench/execution` 当前实现。
- 已有能力直接复用；只有目标确实缺失时才移植。
- 预计需要重点核对：
  - arm trajectory 的 task-provided stop callback；
  - place early-stop 的结果表示和 near-miss 接受条件；
  - release 前的通用检查 hook；
  - 每个导航 stage 的 XY/yaw tolerance override；
  - dry-run 和 execute 模式下相同的诊断结构。
- 为通用层新增不含任务语义的单元测试和失败路径测试。

完成条件：WR05 所需底层能力可由 task-local handler 通过稳定合同调用，且不在通用层出现 WR05 专用分支。

### Phase 3：实现 canonical WR05

- 以当前 `wr02` 的双物体流程为结构参照，但将所需代码本地实现到 `wr05`。
- 将源 task YAML 与基础 WR02 配置完全合并为 WR05 独立配置。
- 搬运并校验真实瓶模型、餐盘摆放、四类 episode、prompt、语义和成功条件。
- 搬运 task-local station sampler、目标计算、plate place、plate pick、shelf return 和操作 handler。
- 将旧 26-stage workflow 改写为 canonical `ExpertPlan`。
- 将 calibration 和 station 文件复制到 WR05 本地，更新其中的 task ID、env name、资源路径和 schema。
- 构造 canonical `ResolvedEpisode`、`RuntimeDescriptor` 和 runtime API。

完成条件：WR05 能通过 catalog dry-run 构造相同 episode、descriptor 和专家计划，且不依赖任何 `new_*` 或其他叶子任务。

### Phase 4：验证 WR05

静态和无模拟器验证：

- task ID、env name、alias、horizon 和 scene 身份正确。
- 所有资源路径均位于 `bench/tasks/wr05`。
- 四类 episode 平衡且 deterministic。
- prompt 不泄漏 lower/upper 私有绑定。
- ExpertPlan 阶段顺序、holding 状态、station purpose 和操作 metadata 正确。
- calibration schema、对象名、station 名称和 plate-left/right 几何关系正确。
- leaf import boundary、legacy import boundary 和 lazy plugin 检查通过。

模拟器验证：

1. 对 seed 0--3 执行 headless scene/reset probe。
2. 对 seed 0--3 执行专家 dry-run。
3. 先对 seed 0 执行完整物理 rollout 并检查 stage diagnostics。
4. 稳定后执行 seed 0--3 完整 rollout，覆盖全部配对。
5. 检查无瞬移、无悬空释放、无明显穿透、瓶体直立、plate 支撑成立、两瓶返回正确层且终态稳定。

完成条件：四种 WR05 episode 均满足任务成功条件，而非仅专家计划完成。

### Phase 5：实现 canonical WR06

- 以当前 `wr04` 的单物体任务结构和严格物理事件状态机为设计参照。
- 将物体替换为真实 ketchup 瓶，暂存位置随 seed 在 plate-left / plate-right 间交替。
- episode 使用 seed parity 决定原始 shelf level，并保留相同层级的独立 placement randomness。
- expert plan 精简为单物体 observe -> pick -> plate stage -> pick -> shelf return 流程。
- 将 WR04 的 sink 状态语义改写为 plate staging 语义，并保留首次错误返回、阶段回退和终态验证的吸收失败规则。
- 从 WR05 的资源中选择 ketchup 所需子集，复制到 WR06 本地并更新身份；不保留 mustard 专属资源（如 `plate_mustard_return_grasp`、`shelf_mustard_grasp`），但保留 ketchup 右槽停靠站 `ketchup_plate_right_station` 和 `plate_right_pick_station`。

完成条件：WR06 是独立、可发现、可运行的 canonical 任务，并且在功能上确实是 WR05 的单物体简化版。

### Phase 6：全局接入

- 在 `bench/tasks/catalog.py` 增加 WR05 和 WR06 catalog entries。
- 在 `bench/tasks/__init__.py` 增加 lazy builtin plugin descriptors 和 aliases。
- 更新所有显式任务集合、数量断言、任务独立性检查、simulator smoke 参数和 import token 门禁。
- 增加 `wr05_smoke_v1.yaml` 和 `wr06_smoke_v1.yaml`，使用当前 observation profile 和 canonical task config 路径。
- 如果 task probe 仍采用显式任务映射，则增加 WR05/WR06。
- 在任务和资源稳定后生成或合并 release entries，并重新计算 release lock hash。
- 更新必要的任务文档和迁移状态，但不把调试输出或临时视频纳入任务包。

完成条件：recording、evaluation 和 simulator 均通过同一 catalog/runtime 加载 WR05 和 WR06。

### Phase 7：最终验收

聚焦测试至少覆盖：

- WR05/WR06 config、episode、prompt、expert plan 和 resource tests。
- WR05 plate place/pick、live attachment 和 station sampler tests。
- WR06 seed parity、memory barrier、wrong-first-return、stage regression 和 terminal verification tests。
- 通用 trajectory stop、pre-release hook 和 navigation tolerance tests。
- task package layout、task plugin、catalog、task independence、import boundary 和 builtin suite tests。
- canonical runtime smoke 和 release gate。

模拟器最终矩阵：

- WR05：seed 0--3，覆盖四种双物体配对。
- WR06：seed 0--3 覆盖四种 (层, plate 侧) 组合；最终使用 seed 0--3 做重复性回归。
- eval smoke：seed 1000、1001、1002，与录制 seed 集合隔离。

## 7. 明确不做的事项

- 不直接复制源 `new_WR05` 后只做包名替换。
- 不搬运 `bundle.py`、`workflow.py`、`RecordingBundle`、旧 `TaskContext` 或旧 `TaskSpec`。
- 不允许 WR05 继承 WR02 的 YAML 或 Python 叶子包。
- 不允许 WR06 直接继承 WR04/WR05 的 YAML、资源或 Python 叶子包。
- 不为了 WR05 在通用 execution 中加入 task ID 分支。
- 不用专家 stage 的成功代替环境最终物理成功。
- 不覆盖或回退目标工作树中与本任务无关的现有修改。

## 8. Definition of Done

只有同时满足以下条件，任务才算完成：

1. `wr05`、`wr06` 均符合 task package v1，并能通过 canonical catalog 和 lazy plugin 发现。
2. 两个任务均无 legacy import、跨叶子 import 和跨任务资源引用。
3. recording 与 evaluation 使用同一个 `ResolvedEpisode` 和 `RuntimeDescriptor`。
4. WR05 四种 episode 和 WR06 两种原始层均经过真实物理执行验证。
5. 暂存、支撑、释放、重新抓取、返回层和终态稳定均由物理观测确认。
6. 聚焦测试、架构合同、suite、simulator smoke 和 release gate 全部通过。
7. 最终 diff 不包含调试产物、临时脚本、输出视频或对现有用户修改的意外覆盖。
