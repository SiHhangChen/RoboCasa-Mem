# MobileMemBench `bench/` 架构精简与双主线重构计划

> 状态：架构决策草案
> 日期：2026-08-03
> 基线提交：`138b483165a42aa83f3a672d36ba7eb6ce47032d`
> 适用范围：`bench/`、14 个内置任务、数据录制、策略评测及其入口脚本
> 本文是当前重构的主设计文档；早期的 `architecture_restructuring_plan.md` 和
> `architecture_review_and_optimization_plan.md` 仅作为历史背景参考。

## 1. 结论先行

MobileMemBench 当前真正需要长期维护的产品能力只有两条主线：

1. 为 14 个任务录制专家数据。
2. 在完全相同的任务定义、初始化和成功标准下评测策略。

因此，重构不应围绕现有文件夹逐个搬家，而应先建立一个共享的“任务—回合运行时”，再让录制和评测分别消费它：

```text
TaskDefinition
└── resolve(seed, variant, request)
    └── ResolvedEpisode（本回合不可变事实）
        └── create/reset/initialize
            └── EpisodeRuntime
                ├── 录制：ExpertPlan → StageRunner
                │                    ├── Navigation
                │                    └── Manipulation
                └── 评测：Policy → RolloutRunner → env.step

两条路径共同使用：
  - 相同的随机决策和初始状态
  - 相同的 observation/action 约定
  - 相同的任务成功与失败语义
  - 相同的任务、资源和版本身份
```

关键决策：

- 录制和评测不得各自维护一套任务定义。
- 当前 `TaskSpec` 实际描述的是专家演示步骤，应重命名为 `ExpertPlan`。
- `StageRunner`、导航和机械臂操作只服务于专家录制，不参与策略评测。
- 评测策略直接产生动作，由 `RolloutRunner` 驱动环境。
- server/client 只是策略进程边界，不等于完整的评测系统。
- 通用层不得出现任务 ID、任务物体名称或任务专用分支。

## 2. 范围与非目标

### 2.1 本轮重构目标

- 14 个任务各有唯一、可发现、可版本化的 `TaskDefinition`。
- 录制和评测通过同一个 `ResolvedEpisode` 创建相同的环境初态。
- 任务的站位、随机区域、导航关键点、操作标定均归任务资源所有。
- 导航和机械臂执行能力在 14 个任务之间统一复用。
- 有一个统一录制入口和一个统一评测入口。
- 有一个策略无关的通信协议、client 和 server 核心。
- 每种 policy 只保留薄的独立启动脚本和 backend 集成。
- 删除已弃用、重复、兼容性过渡和无法到达的代码路径。
- 用明确的目录所有权和单向依赖替代全局注册与延迟导入。

### 2.2 本轮非目标

- 不在目录重构时改变任务难度、成功标准或随机分布。
- 不强行让所有任务拥有完全相同的内部文件结构。
- 不让评测复用专家导航或专家机械臂规划来“帮助”策略。
- 不同时维护多套原生数据格式；外部训练格式由 exporter 转换。
- 不仅为了控制行数而机械拆分文件。

## 3. 完整能力清单

用户已经提出的录制入口、独立 spec、站位、随机区域、导航关键点、统一 manipulation/navigation、阶段 spec、统一 server/client 都是必要能力。还需要补齐以下容易遗漏的部分。

### 3.1 录制与评测共享能力

1. **任务目录与发现**
   - 14 个稳定任务 ID。
   - 每个 ID 对应一个 `TaskDefinition` 和显式版本。
   - 一个只管理任务的 catalog，避免同时注册环境、Station、bundle builder。

2. **确定性回合解析**
   - seed、variant、布局、物体关系和随机区域选择必须可复现。
   - 所有随机决策在创建环境前解析为 `ResolvedEpisode`。
   - 录制和评测对同一任务与 seed 得到相同的解析结果。

3. **环境生命周期**
   - 统一 create、reset、任务初始化、观测、step、close。
   - 统一控制频率、相机、机器人控制器和 reset 后初始化顺序。
   - 环境创建不能隐式反向导入整个 `bench.tasks`。

4. **观测与动作合同**
   - 相机名称、图像尺寸、深度、机器人状态和坐标约定需要版本化。
   - 动作维度、关节/末端表示、夹爪语义、action chunk 和 control Hz 需要固定。
   - 录制数据与评测请求使用相同的标准化定义。

5. **公开信息与 oracle 信息隔离**
   - policy 可见：prompt、允许的传感器、公开 episode 信息。
   - policy 不可见：真实目标关系、隐藏物体选择、专家位姿、成功检查内部状态。
   - recorder 可以保存 oracle metadata，但不能通过 policy observation 泄漏。

6. **统一任务语义**
   - 同一套 success、failure 和 diagnostics 同时服务录制与评测。
   - 专家阶段成功不等于任务成功。
   - 需要统一最大 horizon、终止条件和失败原因分类。

7. **身份与可追溯性**
   - 记录任务版本、配置哈希、资源哈希、代码提交、observation profile 和 action schema。
   - benchmark release 应生成不可变 release lock。
   - 修改成功语义、随机分布或 artifact schema 时必须显式升版本。

### 3.2 专家数据录制专属能力

1. **DatasetPlan**
   - 明确 task × seed × variant × attempt 的录制矩阵。
   - 支持断点恢复、重试、并发占用锁和跳过已完成回合。

2. **ExpertPlan**
   - 阶段式专家步骤，包含导航与操作阶段。
   - 可引用任务本地 Station、route、region 和 manipulation calibration。
   - 只描述专家执行，不承担任务真值和评测语义。

3. **专家执行引擎**
   - `StageRunner` 负责编排和失败短路。
   - 任务本地 `expert/operations.py` 决定目标位姿、位姿顺序、夹爪时序和物理结果验证。
   - 通用 navigation 只接收任务提供的 Station、route 或 waypoint，然后规划并执行底盘路径。
   - 通用 manipulation 只接收任务提供的 `PoseTarget` / `MotionSegment`，调用 MPLib 并执行关节轨迹。
   - 框架不提供 `press_button`、`operate_drawer`、`operate_slide_door`、`rotate_knob` 等带任务语义的共享执行算法。

4. **数据捕获与一致性校验**
   - action、state、RGB、depth、时间戳和阶段事件严格对齐。
   - 保存前检查长度、dtype、shape、FPS 和必需字段。
   - 区分 `accepted`、`rejected` 和 `error`，不能把失败演示混入正式数据。

5. **可靠写盘**
   - 临时目录写入、校验后原子 commit。
   - 保存 checksum、manifest 和 provenance。
   - 中断后不会留下看似完整的 episode。

6. **格式边界**
   - 只定义一个 MobileMemBench 原生数据格式。
   - LeRobot 等训练格式作为独立 exporter，不嵌入主录制循环。

### 3.3 策略评测专属能力

1. **EvaluationSuite 与 Campaign**
   - Suite 描述单组任务、seeds、horizon、action repeat/replan 和 artifact 配置。
   - Campaign 汇总 14 个任务，生成逐任务和整体指标。

2. **策略无关 rollout**
   - 标准循环：observe → infer → validate action → step → success/failure。
   - 确定性消费 action chunk，处理 chunk 用尽、超时和非法动作。
   - rollout 不知道 OpenPI、LeRobot 或其他模型的实现细节。

3. **通用通信边界**
   - client/server 共同实现 handshake、reset、infer、request ID、超时和错误协议。
   - server 不导入 RoboCasa、任务或 evaluation suite。
   - client 不知道实际 policy 类型。

4. **Policy backend**
   - backend 负责 checkpoint 加载、模型预处理、推理、后处理和合同声明。
   - 每种 policy 使用独立薄脚本构造 backend，再启动同一个 server。

5. **评测产物与统计**
   - episode result、轨迹、日志和汇总均需原子、可恢复写入。
   - 统计成功率、Wilson 置信区间、缺失/重复 seed 和失败原因。
   - 结果必须绑定 checkpoint fingerprint 和 benchmark release identity。

### 3.4 开发与质量保障能力

- Station、route、region、grasp/place pose 校准工具。
- 单阶段和单动作 probe。
- 任务资源 schema 校验。
- 固定 seed 的 golden `ResolvedEpisode`。
- 录制/评测初态与成功语义一致性测试。
- 不启动模拟器的纯合同单元测试。
- 少量带模拟器的 smoke/release gate 测试。

这些内容属于 `scripts/tools/` 和 `tests/`，不应成为生产运行时的顶层框架板块。

## 4. 四个必须先冻结的核心接口

文件移动前先冻结以下接口。它们分别归自己的领域所有，不建立一个新的“大而全 contracts.py”。

### 4.1 `TaskDefinition`

归属：`bench/tasks/contracts.py`

职责：

- 声明任务 ID、版本、资源和默认配置。
- 根据 request、seed 和 variant 解析 `ResolvedEpisode`。
- 提供环境描述、初始化描述、prompt 和任务语义。
- 必须提供统一签名的 `ExpertPlan` factory，供专家录制使用；评测路径不调用它。

不负责：

- 创建 recorder。
- 启动评测 client。
- 直接运行 StageRunner。
- 保存任何 artifact。

### 4.2 `EpisodeRuntime`

归属：`bench/runtime/contracts.py`

职责：

- 持有一个已解析回合的活动环境。
- 统一 reset/initialize、observe、step、task state 和 close。
- 提供 policy 使用的标准 I/O，以及专家执行需要的受控 robot/environment port。
- 对外返回统一的 success/failure diagnostics。

不负责：

- 决定 seed 随机结果。
- 选择评测 policy。
- 组织数据集或评测 campaign。

### 4.3 `ExpertPlan`

归属：`bench/execution/contracts.py`

职责：

- 表示专家演示的有序阶段。
- 阶段至少分为 `NavigateStage` 和 `OperateStage`。
- 引用任务本地资源键，不直接塞入 recorder 或 policy 配置。

`TaskSpec` 在完成迁移后应重命名为 `ExpertPlan`，以免误以为评测也必须经过 stage。

### 4.4 `PolicyContract`

归属：`bench/policy/protocol/contracts.py`

职责：

- 声明协议版本、observation schema、action schema、chunk 行为和控制频率。
- 支持严格验证和兼容性检查。
- 绑定模型/checkpoint 身份，但不绑定具体任务。

## 5. 目标目录结构

```text
bench/
├── core/                              # 跨领域且纯粹的基础能力
│   ├── identity.py                    # 哈希、版本、release identity
│   ├── seeds.py                       # stable_seed、SeedStream
│   └── errors.py                      # 跨主线的基础错误与失败分类
│
├── runtime/                           # 活动 episode 与 RoboCasa 生命周期
│   ├── contracts.py                   # EpisodeRuntime 等端口
│   ├── resolver.py                    # TaskDefinition → ResolvedEpisode
│   ├── robocasa.py                    # 创建/reset/initialize/close
│   ├── sensors.py                     # observation profile 与传感器校验
│   └── robot_io.py                    # 标准 observe/step 与机器人 I/O 适配
│
├── tasks/                             # 14 个任务及任务族共享实现
│   ├── contracts.py                   # TaskDefinition、ResolvedEpisode
│   ├── catalog.py                     # 唯一任务注册/发现入口
│   ├── _template/                     # 叶子任务强制目录模板及占位内容
│   ├── families/                      # 同一任务族的领域模板，不注册为任务
│   ├── tm01/
│   ├── tm02/
│   ├── tm03/
│   ├── ts01/
│   ├── ts02/
│   ├── ts03/
│   ├── ts04/
│   ├── wa01/
│   ├── wa02/
│   ├── wr01/
│   ├── wr02/
│   ├── wr03/
│   ├── wr04/
│   └── wx01/
│
├── execution/                         # 仅服务专家演示的脚本化执行
│   ├── contracts.py                   # ExpertPlan、Stage、StageResult
│   ├── runner.py                      # StageRunner
│   ├── navigation/
│   │   ├── contracts.py               # StationTarget、Waypoint、PathResult
│   │   ├── planner.py                 # 给定目标后的 2D grid/A* 路径规划
│   │   ├── executor.py                # 给定路径后的 waypoint 执行
│   │   ├── base_driver.py             # 底盘动作、速度约束和碰撞监控
│   │   └── transforms.py              # 底盘坐标与世界坐标转换
│   └── manipulation/
│       ├── contracts.py               # PoseTarget、MotionSegment、MotionResult
│       ├── mplib.py                   # 给定目标位姿后的 MPLib 规划
│       ├── executor.py                # 关节轨迹执行与环境 step
│       ├── gripper.py                 # 通用夹爪命令与保持
│       ├── collision.py               # 通用碰撞读取和轨迹预检
│       └── transforms.py              # 世界/基座/TCP/关节顺序转换
│
├── recording/                         # 数据集编排和落盘，不含任务逻辑
│   ├── config.py                      # RecordingConfig
│   ├── plan.py                        # DatasetPlan 与 episode work item
│   ├── runner.py                      # 统一录制入口的核心编排
│   ├── capture.py                     # step/observation/action 捕获
│   ├── validation.py                  # 对齐、schema、accepted/rejected
│   ├── artifacts.py                   # 原子写盘、resume、manifest
│   └── exporters/
│       └── lerobot.py                 # 原生格式 → LeRobot
│
├── evaluation/                        # policy-neutral benchmark 执行
│   ├── suite.py                       # suite 与 seed 展开
│   ├── campaign.py                    # 14 任务汇总编排
│   ├── rollout.py                     # 闭环 episode 主循环
│   ├── observation.py                 # runtime obs → canonical obs
│   ├── action.py                      # action/chunk 校验和消费
│   ├── artifacts.py                   # episode/result 原子写入与 resume
│   ├── schemas.py                     # 版本化结果 schema
│   └── summary.py                     # 指标与失败统计
│
└── policy/                            # 唯一策略通信与模型接入边界
    ├── protocol/
    │   ├── contracts.py               # PolicyContract 与消息合同
    │   ├── codec.py                   # msgpack/NumPy 安全编码
    │   └── errors.py                  # 协议错误报文
    ├── client.py                      # 通用同步/异步 client
    ├── server.py                      # 通用服务核心
    ├── checkpoint.py                  # checkpoint 指纹与模型身份
    └── backends/
        ├── base.py                    # PolicyBackend Protocol
        ├── fake.py                    # CPU/E2E 测试后端
        ├── lerobot.py                 # LeRobot 集成
        └── openpi.py                  # OpenPI 集成
```

仓库入口与工具建议：

```text
scripts/
├── record.py                           # 统一专家录制 CLI
├── evaluate.py                         # 统一评测 CLI
├── serve/
│   ├── fake.py                         # 只解析参数并构建对应 backend
│   ├── lerobot.py
│   └── openpi.py
└── tools/
    ├── calibration/                    # station/route/pose 标定
    ├── probes/                         # stage、EEF、scene probe
    └── validation/                     # 资源与 release 校验

configs/
├── benchmark/                          # observation profiles、release locks
├── recording/                          # DatasetPlan
└── evaluation/                         # suites/campaigns

tests/
├── unit/                               # 纯函数、合同、codec、算法
├── contract/                           # 14 任务共同合同
├── integration/                        # recording/evaluation/policy 边界
├── simulator/                          # 少量 RoboCasa smoke
└── release_gate/                       # 固定 seed 与 release lock
```

## 6. 单个任务的建议结构

每个任务只有一个组合入口和一份任务真值；录制与评测不建立平行 task 包。14 个叶子任务必须遵守完全相同的目录与文件骨架，不允许因任务简单而合并、删除或跳过文件。

```text
tasks/<task_id>/
├── __init__.py                          # 只导出 TASK 和稳定身份
├── README.md                            # 任务说明、资源和占位模块说明
├── task.py                              # 唯一组合入口，导出 TASK
├── config.py                            # 任务强类型配置与资源路径
├── episode.py                           # seed 决策、隐藏关系、variant、horizon
├── environment.py                       # 任务环境声明和 fixture/object 映射
├── initialization.py                    # reset 后任务初始化；无额外步骤时保留占位
├── observation.py                       # 模拟器状态 → 本任务物理状态/事件
├── semantics.py                         # 任务成功、失败和 diagnostics
├── prompts.py                           # 公开 prompt 与 oracle 信息边界
├── expert/                              # 仅专家录制使用
│   ├── __init__.py                      # expert 子包稳定出口
│   ├── spec.py                          # ExpertPlan 阶段
│   ├── stations.py                      # 本任务 Station/route 解析
│   ├── targets.py                       # 本任务标定/实时状态 → MotionSegment
│   └── operations.py                    # 本任务操作 → 位姿/夹爪/保持序列
└── resources/
    ├── task.yaml                        # 任务静态配置
    ├── sampling/
    │   └── regions.json                 # 初始/目标随机区域；无区域时合法空对象
    ├── navigation/
    │   ├── stations.json                # 可停靠站位；无站位时合法空对象
    │   └── routes.json                  # 人工关键点；无 route 时合法空对象
    ├── interactions/
    │   └── calibrations.json            # 本任务全部交互标定；无标定时合法空对象
    ├── scene/
    │   └── placements.json              # 本任务场景摆放；无额外摆放时合法空对象
    └── assets/
        └── manifest.yaml                # 任务资产清单；无专用资产时合法空清单
```

结构规则：

- 上述路径是强制结构，不是示例；TM01 到 WX01 必须逐项存在。
- 简单任务不得把 `episode.py`、`environment.py`、`initialization.py`、`observation.py`、`semantics.py` 或 `prompts.py` 合并进 `task.py`，也不得省略 `expert/targets.py`。
- 当前结构之外不得为某一个任务单独增加顶层 Python 文件；需要增加职责时，先升级任务结构版本，再为 14 个任务同时增加该文件。
- Python 占位模块必须可以正常 import，并用模块 docstring 说明本任务为何不需要额外实现；不得只留下空文件或无意义的 `pass`。
- JSON/YAML 占位资源必须符合对应 schema，例如包含 `schema_version` 和空的 `entries`，不能使用无法校验的空文件或 `.gitkeep` 代替。
- `bench/tasks/_template/` 保存这套规范模板；新增任务必须由模板生成，不能手工自由搭建目录。
- 任务结构本身具有 `TASK_PACKAGE_LAYOUT_VERSION`；结构调整是一次覆盖模板和全部叶子任务的显式迁移。
- 所有 interaction calibration、Station、route、scene placement、随机区域和物理阈值均由单个任务独立持有。
- 即使两个任务当前使用完全相同的标定数值，也分别保存；一个任务的重新标定不得静默改变另一个任务。
- 任务之间禁止通过相对路径引用对方的 `resources/`。
- 两个明确同族的任务可以共享纯领域代码；三个及以上独立任务共同需要且不含任务语义的机制，才考虑进入框架通用层。

一致性同时覆盖文件 API：

| 固定文件 | 必须提供的稳定出口 |
|---|---|
| `task.py` | `TASK` |
| `config.py` | `load_task_config()` |
| `episode.py` | `resolve_episode()` |
| `environment.py` | `build_environment_descriptor()` |
| `initialization.py` | `initialize_episode()` |
| `observation.py` | `observe_task_state()` |
| `semantics.py` | `evaluate_task_state()` |
| `prompts.py` | `resolve_prompt()` |
| `expert/spec.py` | `build_expert_plan()` |
| `expert/stations.py` | `resolve_station_targets()` |
| `expert/targets.py` | `resolve_motion_targets()` |
| `expert/operations.py` | `build_operation_handlers()` |

占位模块仍实现同一个接口，并显式返回 no-op 结果。例如没有额外初始化逻辑的任务仍保留：

```python
"""This task requires no initialization beyond the shared runtime reset."""

from bench.runtime.contracts import InitializationReport


def initialize_episode(runtime, episode) -> InitializationReport:
    return InitializationReport.noop(reason="shared_runtime_reset_is_sufficient")
```

这样调用方不需要 `hasattr()`、可选 import 或任务 ID 分支；差异只存在于函数返回的数据和任务资源内容中。

### 6.1 共享运动机制与任务动作的边界

框架不抽象“按按钮”“开抽屉”“拉滑门”“旋转旋钮”等语义动作。机械臂共享层只有一个统一模型：

```text
任务本地 expert operation
└── 解析本任务资源和实时状态
    └── 生成 PoseTarget / MotionSegment 序列
        └── MPLib.plan(target_pose)
            └── execute(joint_trajectory)
                ├── 可选 gripper command
                ├── 可选 dwell/hold
                └── 返回 MotionResult
                    └── 任务本地 semantics/handler 验证物理结果
```

例如，按钮、抽屉和旋钮在框架看来没有区别：

```text
按钮：任务生成 prepress → contact → retreat
抽屉：任务生成 prehook → hook → pull waypoints → retreat
旋钮：任务生成 pregrasp → grasp → rotation poses → retreat

共享层：只逐段规划和执行，不知道这些序列分别代表什么。
```

任务本地负责：

- 决定目标位姿及其坐标系。
- 决定位姿顺序和是否使用实时状态重新计算。
- 决定夹爪开合、保持时间、接触策略和碰撞策略参数。
- 保存所有交互标定。
- 判断按钮是否真实位移、抽屉是否打开、滑门是否关闭、旋钮是否转到目标区间。

通用 execution 只负责：

- 世界、机器人基座、TCP 和关节坐标转换。
- 给定目标位姿后调用 MPLib。
- 对齐关节顺序并执行规划轨迹。
- 发出通用夹爪命令和环境 step。
- 执行通用碰撞检查并返回结构化结果。
- 给定 Station/route/waypoint 后规划并执行底盘移动。

因此，不应在通用目录中出现以下文件：

```text
press_button.py
operate_drawer.py
operate_slide_door.py
rotate_knob.py
```

对应逻辑应位于具体任务的 `expert/operations.py`；只有位姿规划和轨迹执行机制位于 `execution/`。

### 6.2 `families/` 的准确职责

`tasks/families/` 共享的是“同族任务结构”，不是物理执行算法或标定资源。

允许放入 family：

- 参数化的 episode 决策算法。
- 参数化的任务状态机或状态转换骨架。
- 不包含任务 ID 的 ExpertPlan 阶段生成骨架。
- 同族任务共同的数据合同和纯几何计算。
- 将任务本地参数转换为通用 `PoseTarget` / `MotionSegment` 的纯生成逻辑，前提是行为确实完全一致。

禁止放入 family：

- 具体任务 ID、prompt、horizon 和 release version。
- 具体物体名、fixture 名或任务专用 success condition。
- interaction pose、Station、route、scene placement 和随机区域。
- 任何任务 YAML 或其他叶子任务资源的相对路径。
- 创建 recorder、启动 evaluation 或连接 policy 的逻辑。
- 名为 `press_button()`、`open_drawer()` 等但内部只是固化某任务位姿与阈值的高层执行器。

依赖方向必须是：

```text
tasks/<task_id>/
  └── tasks/families/<family>/
      └── tasks/contracts + execution/contracts + core

family 不得反向 import 任何 tasks/<task_id>/
一个叶子任务不得 import 另一个叶子任务
```

### 6.3 从当前 14 个任务出发的 family 划分

| Family | 当前成员 | 可以共享的候选 | 必须留在任务本地 |
|---|---|---|---|
| `sugar_count` | TS01、TS03 | K/N 校验、糖块名称生成、目标集合采样、exact-count 状态机骨架、阶段序列骨架 | TS01/TS03 prompt、profile 规则、场景、糖块区域、抓取/释放位姿、Station、route、阈值 |
| `drawer_inspection` | TS02、TS04 | fixture 顺序采样、open→inspect→close 状态转换、exactly-once 检查、阶段序列骨架 | 抽屉数量/名称、场景摆放、hook/pull 位姿、Station、route、检查阈值、完成按钮标定 |
| `cabinet_memory` | WR01、WR03 | level 校验、存储关系表示、memory barrier 状态骨架、专家阶段骨架 | 单罐/双罐规则、目标选择、柜体摆放、抓取/存储/取回位姿、Station、route、成功细节 |
| `shelf_return` | WR02、WR04 | shelf level 表示、初始层采样、stage→remember→return 状态骨架、专家阶段骨架 | 单物体/双物体规则、sink staging 区域、shelf/sink 位姿、torso 高度、Station、route、阈值 |
| `interleaved_task` | TM01、TM03；TM02 后续评估 | precedence/event-route 的纯表示、合法事件推进、确定性路线采样框架 | 每个任务的事件集合和依赖图、炉灶/抽屉/烤箱/水/柜体标定、全部 route、物理 observer 阈值、成功条件 |

逐任务落位：

| 任务 | Family 使用计划 | 明确保留的任务本地内容 |
|---|---|---|
| TM01 | 使用 `interleaved_task` 的事件图合同；不再作为其他任务的基类 | 炉灶与两个抽屉的事件图、全部位姿、物理 observer、route、ExpertPlan |
| TM02 | 第一阶段保持独立；只有证明三类 TM 状态机可由同一纯模型表达后才接入 family | 水、锅、罐头、柜门和完成按钮的全部状态、位姿、route 与 ExpertPlan |
| TM03 | 与 TM01 共享纯事件图机制，不继承 TM01 的具体环境或操作实现 | toaster 线程、扩展事件图、全部炉灶/抽屉/烤箱标定、route 与 ExpertPlan |
| TS01 | `sugar_count` 基础使用者 | 自己的 episode policy、资源、位姿、route、prompt 和 success report |
| TS02 | `drawer_inspection` 三 fixture 配置 | 三个 fixture 的具体身份、场景、标定、route 和完成按钮资源 |
| TS03 | `sugar_count` 的四 profile 使用者，不再 import TS01 | profile 层级、prompt、自己的全部资源和 release identity |
| TS04 | `drawer_inspection` 两 fixture 配置，不再 import TS02 | 两个 fixture 的具体身份、场景、标定、route 和完成按钮资源 |
| WA01 | 暂不建立单成员 family | 咖啡/糖关系、场景、阶段、全部资源和语义 |
| WA02 | 暂不建立单成员 family | 冷热水关系、水按钮/杯子标定、Station、route、阶段和语义 |
| WR01 | `cabinet_memory` 双罐配置 | 红蓝罐映射、目标选择、全部柜体/物体标定和 route |
| WR02 | `shelf_return` 双物体配置 | 两个物体和相对位置规则、全部 shelf/sink 标定和 route |
| WR03 | `cabinet_memory` 单罐配置，不再 import WR01 | 单罐隐藏层规则、自己的全部柜体/物体标定和 route |
| WR04 | `shelf_return` 单物体配置，不再 import WR02 | 单物体原始层规则、自己的全部 shelf/sink 标定和 route |
| WX01 | 暂不建立单成员 family | wipe/return 状态机、场景、位姿、route、阶段和语义 |

### 6.4 每个 family 的建议文件形态

固定骨架要求适用于 14 个可运行叶子任务。family 是不可注册、不可独立运行的内部库，不冒充第 15 个任务，因此按实际共享职责保留最小文件集合。

```text
tasks/families/<family_name>/
├── README.md                 # 使用任务、共享不变量、非目标和版本说明
├── contracts.py              # family 输入/输出，不含任何任务常量
├── episode.py                # 可选：纯 seed/variant 决策
├── semantics.py              # 可选：参数化状态转换骨架
└── expert.py                 # 可选：生成 ExpertPlan/MotionSegment 骨架
```

不是每个 family 都必须具有全部文件。family 不应包含 `resources/`；具体资源始终位于 14 个叶子任务内。

叶子任务通过显式参数使用 family：

```python
family_spec = DrawerInspectionFamilySpec(
    fixture_ids=task_config.fixture_ids,
    required_inspections=len(task_config.fixture_ids),
)

episode = resolve_drawer_episode(
    family_spec,
    seed=request.seed,
)
```

family API 不读取任务 YAML、不根据 task ID 分支，也不从调用栈寻找资源。

### 6.5 新增 shared/family 代码的流程

以后出现通用候选时，按以下顺序处理：

1. **先确定共享的是什么**
   - 相同任务族的领域结构：进入 `tasks/families/<name>/`。
   - 仅仅都是“给位姿再规划”：继续使用 `execution/manipulation`，不新增高层 shared。
   - 相同数值标定或相同 route：仍各任务独立保存，不提取。

2. **提交复用证据**
   - 列出至少两个同族调用者。
   - 写清共同不变量和每个任务的差异。
   - 证明共享 API 不需要 task ID 分支或读取其他任务资源。

3. **先冻结行为**
   - 为每个调用任务保留固定 seed 的 `ResolvedEpisode` golden。
   - 固定 ExpertPlan 阶段序列、MotionSegment 序列和 success diagnostics。
   - 提取前后逐任务对比，禁止顺便修改任务行为。

4. **设计最小参数化合同**
   - 输入只包含调用任务显式提供的数据。
   - 输出为纯数据、状态转换结果、ExpertPlan 或 MotionSegment。
   - 不接收 recorder、evaluation client 或全局 registry。

5. **逐任务迁移**
   - 先迁移一个任务并保持测试通过。
   - 再迁移同族任务。
   - 删除叶子任务之间的 import 和跨任务资源引用。
   - 临时兼容 import 必须注明删除阶段，不能成为永久 API。

6. **建立所有权和测试**
   - family `README.md` 列出正式 consumers。
   - 每个 consumer 至少有一个合同测试和一个固定 seed 测试。
   - 架构测试禁止 family import 叶子任务、叶子任务相互 import、任务资源跨目录引用。

新增第 15 个任务时：

- 从 `bench/tasks/_template/` 生成完整目录，声明相同的 `TASK_PACKAGE_LAYOUT_VERSION`，不得删减占位模块或资源文件。
- 再实现独立的 TaskDefinition、语义和资源内容。
- 如果它与现有 family 的不变量完全匹配，则通过 typed spec 接入。
- 如果只是存在相似代码，先保留任务本地实现；完成第二个稳定消费者和行为对比后再提取。
- 新任务永远不能直接 import TS01、WR01 等现有叶子任务来“复用”。

## 7. 两条主线的最终数据流

### 7.1 专家录制

```text
scripts/record.py
└── DatasetPlan（tasks × seeds × variants × attempts）
    └── TaskCatalog.get(task_id)
        └── TaskDefinition.resolve(...)
            └── ResolvedEpisode
                └── RuntimeFactory.create(...)
                    └── EpisodeRuntime
                        └── ExpertPlanFactory.build(...)
                            └── StageRunner.run(...)
                                ├── task expert resolves Station/route
                                │   └── NavigationExecutor
                                ├── task expert resolves MotionSegment
                                │   └── MPLib + TrajectoryExecutor
                                └── Capture
                                    └── Validation
                                        ├── accepted
                                        ├── rejected
                                        └── error
                                            └── atomic artifact commit
                                                └── optional exporter
```

### 7.2 策略评测

```text
scripts/evaluate.py
└── EvaluationCampaign
    └── EvaluationSuite（tasks × seeds × horizons）
        └── TaskCatalog.get(task_id)
            └── TaskDefinition.resolve(...)
                └── ResolvedEpisode
                    └── RuntimeFactory.create(...)
                        └── EpisodeRuntime
                            └── RolloutRunner
                                ├── canonical observation
                                ├── PolicyClient.infer
                                ├── action/chunk validation
                                ├── runtime.step
                                └── shared task semantics
                                    └── episode artifacts
                                        └── campaign summary
```

### 7.3 Policy server

```text
scripts/serve/openpi.py ─┐
scripts/serve/lerobot.py ├── build PolicyBackend
scripts/serve/fake.py ───┘
                              │
                              ▼
                    shared PolicyServer
                    ├── hello/contract
                    ├── reset
                    ├── infer
                    ├── request id/timeout
                    ├── auth/error/health
                    └── backend.predict(...)
```

Policy 脚本只处理 policy 特有参数、构造 backend 和启动公共 server，不复制 WebSocket 协议或 rollout 代码。

## 8. 当前目录到目标目录的合并方案

| 当前板块 | 处理方式 | 目标所有者 |
|---|---|---|
| `registry.py` | 缩减为只注册 `TaskDefinition` 的 catalog；删除 Station、环境和 bundle 多重注册 | `tasks/catalog.py` |
| `env/` + `task_context/` | 合并生命周期、context resolver、sensor profile | `runtime/` |
| `model/` | 不再作为通用杂物层；按所有权分发类型 | `tasks/`、`runtime/`、`execution/` |
| `stages/` | 明确为专家执行编排 | `execution/runner.py` |
| `sampling/rng.py` | 作为跨领域确定性基础能力 | `core/seeds.py` |
| station/route 选择 | 具体候选和资源归任务；通用层只执行已解析目标 | `tasks/<id>/expert/` + `execution/navigation/` |
| target/region sampling | episode 随机区域和交互目标全部归具体任务 | `tasks/<id>/episode.py` + `tasks/<id>/expert/` |
| `navigation/` | 保留给定 Station/route 后的规划、底盘路径执行和坐标转换 | `execution/navigation/` |
| `planning/nav_algo.py` | 归统一导航 planner | `execution/navigation/planner.py` |
| MPLib、模型对齐、轨迹执行、碰撞 | 作为给定目标位姿后的统一运动机制 | `execution/manipulation/` |
| `pick_place_targets.py` 等目标生成 | 通用坐标数学留下；带任务语义的目标序列迁回任务 | `execution/manipulation/transforms.py` + `tasks/<id>/expert/operations.py` |
| `manipulation/controller.py` | 删除高层操作调度；保留 plan-to-pose、轨迹、夹爪和结果合同 | `execution/manipulation/` |
| `recording/` | 保留顶层职责，增加 DatasetPlan、validation、resume、exporter | `recording/` |
| `evaluation/` | 保留 rollout/suite/artifact；删除 RecordingBundle 兼容路径 | `evaluation/` |
| `policy_protocol/` + `serving/` + `evaluation/client.py` | 合并成唯一策略 I/O 边界 | `policy/` |
| `experiments/` | 迁移稳定 probe | `scripts/tools/probes/` |
| `examples/` | 有文档价值则移入 docs；否则删除 | `docs/examples/` |
| `controls/` | 只有历史 pyc，无维护价值 | 删除 |

### 8.1 高置信度清理候选

以下内容应先通过 `rg`、import graph 和测试复核；确认无调用后优先删除：

- `manipulation/adapter.py` 旧 `MobileManipulationContext` 适配层。
- `navigation/simple_navigator.py`。
- `navigation/rotation_primitive.py`。
- `sampling/target_sampler.py`。
- `sampling/pose_targets.py`。
- `tasks/_options.py`。
- `evaluation/bundle_adapter.py`，前提是所有任务已切到共享 runtime。
- 空的任务 evaluation adapter 子类。
- 未使用的 Task/Station/bundle registry 分支。
- 未被使用的大型配置兼容解析函数。
- 仅占位且近期没有落地计划的 backend。

删除原则：先证明不可到达，再删除；禁止一边大规模移动、一边改变运行行为。

## 9. 强制依赖规则

这些规则应通过 import-linter 或架构测试固定下来：

1. `tasks/*/semantics.py` 不得导入 recording、evaluation、policy 或运动执行器。
2. evaluation 不得依赖 `RecordingBundle`、`ExpertPlan` 或 `StageRunner`。
3. recording 不得包含任何任务 ID 或任务物体 token。
4. policy server 不得导入 tasks、RoboCasa 或 evaluation suite。
5. policy client 不得导入具体 backend。
6. navigation/manipulation 不得识别任务 ID，也不得定义按钮、抽屉、滑门、旋钮等任务语义动作。
7. runtime 不得反向导入全部内置任务；任务由 catalog 显式装载。
8. task-local expert code 可以依赖通用 execution，反向依赖禁止。
9. expert stage 成功不得替代最终任务成功检查。
10. exporter 不得反向影响原生录制格式和录制主循环。
11. 一个叶子任务不得 import 另一个叶子任务；共享逻辑只能经过显式 family 合同。
12. 任务 interaction、Station、route、scene placement 和 sampling 资源不得跨任务引用。
13. family 不得 import 叶子任务、持有具体任务标定或根据 task ID 分支。

允许的主要依赖方向：

```text
scripts/record
  └── recording
      ├── tasks + runtime
      └── execution ── runtime + core

scripts/evaluate
  └── evaluation
      ├── tasks + runtime + core
      └── policy client ── protocol

scripts/serve/*
  └── policy server
      ├── protocol
      └── backend

runtime ── task contracts + core
policy server 不接触 tasks/runtime/evaluation
```

## 10. 文件与文件夹的判断标准

只有满足以下至少一项时，能力才值得成为顶层文件夹：

- 有独立生命周期或编排入口。
- 有明确公共合同和多个实现。
- 被两个产品主线独立消费。
- 需要独立测试、版本或 artifact schema。

以下情况优先放在所属领域内部，而不是新建顶层目录：

- 只被 navigation 使用的 A*、速度曲线、路径执行和碰撞监控。
- 只被 manipulation 使用的 MPLib、轨迹执行、夹爪控制、坐标转换和碰撞监控。
- 任务所需的 PoseTarget、MotionSegment、Station、route 和交互标定。
- 按按钮、开抽屉、拉滑门、转旋钮等任务语义操作。
- 只被单个任务使用的随机区域或物体语义。
- 只用于开发的 calibration/probe。

文件拆分标准：

- 一个文件应有一个可用一句话说明的职责。
- 只有调用方向清晰、可单独测试的边界才值得拆分。
- 超过约 800 行时必须审查职责，但行数不是唯一标准。
- 两个高度耦合、总是一起修改且没有独立复用价值的文件应合并。
- 大量 `__init__.py` 转发、字符串延迟导入和兼容 shim 是迁移债务，不是稳定架构。
- 上述一般标准不能破坏叶子任务的固定骨架：若某个任务确实需要新的职责文件，必须提升 `TASK_PACKAGE_LAYOUT_VERSION`，将该文件加入 `_template/` 和全部 14 个任务，其余任务提供合法占位实现。
- 禁止只为单个任务创建额外的 `helpers.py`、`utils.py`、`geometry.py` 或临时顶层模块；新增职责必须获得全体任务一致的文件名和稳定出口。

## 11. 分阶段迁移计划

### Phase 0：冻结基线

当前状态已在提交 `138b483` 建立快照。

完成标准：

- 记录当前可运行任务和已知失败。
- 固定代表性 seeds、初态、stage 序列、success report 和 artifact schema。
- 在结构重构期间不混入任务行为调整。

### Phase 1：冻结四个核心接口

实现或收敛 `TaskDefinition`、`EpisodeRuntime`、`ExpertPlan`、`PolicyContract`，先建立合同测试，不立即移动所有代码。

完成标准：

- 14 个任务可被统一 catalog 发现。
- 同一 task/seed 可生成可序列化、可哈希的 `ResolvedEpisode`。
- recording/evaluation 对接口的依赖不再通过 bundle 互相绕行。

### Phase 2：删除高置信度死代码和兼容层

先减少迁移面，再做目录合并。

完成标准：

- 每个删除项有搜索证据和对应测试。
- 删除未使用 registry、adapter、placeholder 和旧导航路径。
- 不新增兼容 shim。

### Phase 3：建立共享 TaskDefinition 与 Runtime

把 `env/` 与 `task_context/` 收敛到 runtime；让 14 个任务通过相同 resolver 创建回合。

完成标准：

- recording/evaluation 对相同 task/seed 得到相同解析哈希和初态。
- 环境生命周期只有一个实现入口。
- 消除 `env.robocasa_runtime ↔ tasks` 循环依赖。

### Phase 4：评测完全切换到共享 Runtime

评测直接消费 `ResolvedEpisode` 和 `EpisodeRuntime`。

完成标准：

- 删除 `bundle_adapter.py` 和从 RecordingBundle 反装配环境的路径。
- 所有 14 个 suite 使用同一个 rollout 核心。
- server/client 与 policy backend、任务 runtime 完全解耦。

### Phase 5：收敛统一录制 API

引入 `DatasetPlan` 和通用 bundle/runtime 装配，逐步消除 14 份重复 bundle builder。

完成标准：

- 任务包只提供 TaskDefinition 和统一签名的 ExpertPlan factory。
- 录制入口不知道任务特有构造函数。
- resume、retry、accepted/rejected/error 和原子写入有测试覆盖。

### Phase 6：合并专家执行目录

将 navigation/manipulation 的底层运动机制迁入 `execution/`；把高层操作分解为任务本地生成的 Station/route/PoseTarget/MotionSegment。

完成标准：

- 不再存在泛化且所有权不清晰的顶层 `planning/` 和 `sampling/`。
- 通用层只负责“给定目标后的规划与执行”，不含任务 token 或按钮/抽屉/滑门/旋钮等动作名称。
- 每个任务独立拥有 interaction calibration、Station、route、scene placement、阈值和物理验证。
- 消除任务之间的资源相对路径引用。
- StageRunner 只被 recording/expert 工具调用。

### Phase 7：整理 14 个任务包

去掉 `new_` 前缀和 import shim，任务资源归位；按第 6 节的 family 边界提取已验证的纯领域逻辑。

完成标准：

- 14 个任务 ID、目录名和 suite 名一致。
- 14 个任务与 `_template/` 具有相同的强制相对路径集合和 `TASK_PACKAGE_LAYOUT_VERSION`。
- 所有占位 Python 模块可导入且返回显式 no-op；所有占位资源通过 schema 校验。
- 每个任务有唯一入口 `TASK`。
- 不存在录制版 task 与评测版 task 两套定义。
- 不存在叶子任务之间的 Python import 或资源引用。

### Phase 8：发布与文档收尾

完成 release lock、原生 artifact schema、数据 exporter 和运维说明。

完成标准：

- 能从空输出目录执行一轮录制和一轮远程评测。
- 结果可关联 task/config/resource/code/checkpoint 身份。
- 旧文档、示例和脚本完成归档或删除。

## 12. 每阶段必须保持不变的兼容性

除非显式提升对应版本，否则迁移必须保持：

- 相同 seed 的随机决策。
- 相同环境初始状态。
- 相同 ExpertPlan 阶段序列。
- 相同任务成功和失败语义。
- 相同 observation/action 数值约定。
- 相同已发布 artifact schema。

目录路径和 Python import 可以分阶段改变，但需要短期迁移表；不应永久保留多层 shim。

## 13. 测试与验收矩阵

| 层级 | 核心测试 |
|---|---|
| core | seed 稳定性、规范化哈希、release identity |
| task layout | 14 个任务与 `_template/` 强制路径一致、结构版本一致、占位模块可导入、空资源 schema 合法 |
| task contract | 14 个任务 ID 唯一、稳定模块出口一致、ResolvedEpisode 可序列化、无叶子任务互相 import/资源引用 |
| family contract | family 无任务常量/资源，输入显式，所有 consumer 的固定 seed 行为保持一致 |
| task golden | 固定 seed 的隐藏关系、prompt、初始化描述和哈希 |
| runtime | create/reset/initialize/observe/step/close 生命周期 |
| consistency | 同 task/seed 的 recording 与 evaluation 初态及 semantics 一致 |
| expert execution | stage 分派、失败短路、任务生成的 MotionSegment、MPLib 规划与轨迹结果汇总 |
| recording | 数据长度对齐、原子 commit、resume、accepted/rejected/error |
| policy protocol | codec、合同协商、非法输入、超时、错误码 |
| evaluation | action chunk、horizon、成功终止、artifact resume、summary |
| remote E2E | fake backend + server + client + rollout |
| release gate | 14 个任务的 release lock 和少量模拟器 smoke |

## 14. 尚需明确的产品决策

这些不是目录问题，应在实现对应阶段前明确：

1. MobileMemBench 原生 dataset schema 的唯一正式版本是什么。
2. policy 可见的 observation profile 有哪些正式组合，RGB-D 是否进入首版。
3. 动作合同最终维度、坐标系、control Hz、action repeat 和 chunk 规则。
4. 哪些 episode 失败可重试，哪些必须记为 rejected。
5. 公开 prompt 与 oracle metadata 的精确字段边界。
6. benchmark 总分如何跨 14 个任务聚合，是否按任务等权。
7. WebSocket 是否是唯一公开 transport；本地 in-process 是否只用于测试。
8. 旧数据 artifact 是否需要读取兼容，兼容期持续多久。
9. AC-DiT/RGB-D backend 是否进入近期发布范围；若否，应删除占位实现。

## 15. 最终验收指标

重构完成不是“文件变少”，而是以下条件同时成立：

- 14 个任务各有且只有一个 TaskDefinition。
- 14 个任务使用完全一致、可由模板验证的文件结构和模块出口。
- 同一 task/seed 的录制和评测共享解析结果、初态和成功语义。
- 录制只有一个主入口，评测只有一个主入口。
- 不再有 14 份重复的 RecordingBundle 装配逻辑。
- 策略协议、client、server 只有一套；policy 差异只存在于 backend 和薄启动脚本。
- evaluation 不依赖 StageRunner 或 RecordingBundle。
- 通用 navigation/manipulation 只负责给定目标后的规划与执行，不含任务 token 或高层语义动作。
- interaction calibration、Station、route、scene placement、随机区域和物理阈值全部由各任务独立持有。
- 不存在叶子任务之间的 Python import 或资源引用。
- 核心包之间无循环 import。
- 失败回合、超时、中断和部分 artifact 都可被准确识别和恢复。
- 任一正式结果都能追溯到任务、资源、配置、代码、协议和 checkpoint 版本。

这套标准比“每个文件少于多少行”更重要。精简的目标是减少概念数量、重复装配和双重事实来源，同时保留任务本身确实需要的复杂度。
