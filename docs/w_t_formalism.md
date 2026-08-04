# W/T 记忆形式化体系（提炼自 membench）

> 来源：`/home/user/chensihang/membench/docs/design.md` + `membench/specs/schema.py` + 9 个 task 实现。
> 目的：把 membench 已经成型的 **W/T 形式化** 和 **reveal-memory 双轨设计** 抽成 housemem 的理论基础，明确什么继承、什么修改、什么扩展。
> 适用范围：housemem 的 task 库设计、wrapper YAML schema 升级、paper outline §3 改写。

---

## 一、核心 POMDP 形式化

POMDP 中，agent 每步只看到局部观测 $o_t$，但最优动作并不是 $o_t$ 单独决定的。决定最优动作的是 "**做出正确动作所必须的决策变量**"。

> 当前观测不足以决定最优动作，agent 必须从历史信息中恢复某些缺失的决策变量，才能做出正确决策。

形式：

$$
a_t^* = \pi^*(z_t),\qquad z_t = (w_t,\ \tau_t)
$$

| 符号 | 含义 | 性质 |
|---|---|---|
| $w_t$ | 任务相关的**世界状态**变量 | 即使没有任务，仍然是世界事实 |
| $\tau_t$ | 任务相关的**目标/规则/上下文**变量 | 只在特定任务语境下有意义 |

**关键洞察**（这是 membench 的 design.md 最有价值的一句）：

> Memory demand 不是由任务长度决定的，也不是由是否存在视频历史决定的。memory demand 由**当前最优动作所依赖的变量，是否已经从当前观测中消失**决定。

→ 任务设计的本质是 **engineering "信息消失"**：让 agent 必须在某时刻把信息存入记忆，并在合适时机取出来用。

---

## 二、W 维度的 3 类细分（object-centric 分解）

```
w_t = { X_t,  A_t,  R_t }
       └─单体几何 └─单体功能 └─多体关系
```

### W-X：单实体空间/几何状态记忆

$X_t$ = 单个实体的位置、姿态、原始位置、被遮挡后的当前位置。

| 经典场景 | 缺失变量 | 测试什么 |
|---|---|---|
| 物体被遮挡后移动到不可见区域 | `current_position(object)` | 隐藏定位 |
| 物体被操作后必须放回原位 | `original_position(object)` | 历史空间记忆 |

**RoboCasa 对应**：
- 物体在 cabinet 第几层（`inside(obj, shelf_i)` 既算 W-R 也算 W-X）
- 多次开关 fridge 后物体在哪一层
- 离开 stove 区域后回来，记得 kettle 放在哪个 burner

### W-A：单实体物理/功能状态记忆

$A_t$ = 实体内部状态：开/关、锁/解锁、激活/未激活、满/空、加盐/未加盐、冷热状态、charged。

| 经典场景 | 缺失变量 | 测试什么 |
|---|---|---|
| 两个外观一样的杯子，一个被加热过 | `heated(cup_i)` | 隐藏属性 |
| 三块电池外观一样，一块充过电 | `charged(battery_i)` | 隐藏功能状态 |

**RoboCasa 对应（关键！）**：
- `burner.is_on(loc)` — 哪个炉头开过
- `kettle.turned_on` — 水壶通电没
- `cabinet.door_state` — 哪个柜门已经打开过
- 这些都是 W-A 类型，**RoboCasa 的 88 fixture × 33 OU 函数大部分对应 W-A 变量**

### W-R：多实体关系记忆

$R_t$ = 实体之间的关系：inside / on / under / occlude / support / contain / contact。

| 经典场景 | 缺失变量 | 测试什么 |
|---|---|---|
| 三个杯子盖住一个物体后被洗牌 | `covered_by(ball, cup_i)` | 隐藏包含关系 |
| 把物体放入某层柜格后关门 | `inside(obj, shelf_i)` | 跨实体容器关系 |
| 食材分别放进相同盒子，盖盖后交换 | `inside(item, box_i)` | 隐藏归属 |

**RoboCasa 对应**：
- 跨 zone 任务里"物体在哪个 zone"是天然的 W-R
- 把物体放入 fridge 抽屉后关门 → `inside(obj, fridge.drawer)`
- 多 burner 共存时 `on(saucepan, burner_i)` 的 i 是 W-R

---

## 三、T 维度的 2 类细分

```
τ_t = { G,  Q_t }
       └─静态成功约束 └─动态任务进度
```

### T-G：成功约束记忆

$G$ = 任务成功时世界应满足的约束条件，整局静态，但**可能从当前观测中消失**。

| 经典场景 | 缺失变量 | 测试什么 |
|---|---|---|
| 早期 reveal 目标构型，随后乱推桌面 | `desired_layout` | 成功构型记忆 |
| 早期 reveal 复现路线，随后清理障碍 | `desired_trajectory` | 路径约束记忆 |

**RoboCasa 对应**：
- 多任务 episode 开始时给定的"准备一顿晚餐"的具体子项清单
- 早期 reveal 一个"摆桌图样"，agent 中间忙别的事，最后必须按图样摆
- `SetupBowls` 之类任务的"碗对应哪个 stool"映射

> **重要的反例**（design.md §"T-G 限制"）：
> 把"分拣规则 / 配方规则"做成"早期给出，随后规则卡消失"**不算干净的 T-G**——它更接近 instruction retention，应单独讨论。

### T-Q：任务进度记忆

$Q_t$ = 当前任务执行到了什么阶段、已完成子目标集合、计数进度、条件分支路径。

| 经典场景 | 缺失变量 | 测试什么 |
|---|---|---|
| 红蓝按钮交替按下 4 次 | `current_step ∈ {1,2,3,4}` | 步骤进度 |
| 泡茶 → 收纳 → 中途回去取茶包 → 恢复收纳 | `pending_followup`, `resume_subgoal`, `interrupted_task` | **prospective memory** |

> **`T-Q-2 / DeferredTeaCleanup` 就是 prospective memory 的现成模板**。它已经实现了：
> - A1: 启动泡茶（产生 `pending_tea_followup=true`）
> - B1/B2/B3: 收纳餐具子目标
> - A2: 在 B1/B2/B3 后随机中断，回去取茶包
> - Resume B: 必须从正确中断点继续
> - B4: 移动收纳盒到服务区
>
> housemem 的 K=2/3/4 多任务交错就是 T-Q-2 模板的**结构化扩展**。

---

## 四、W/T 边界判定准则

> **该信息在任务不存在时是否仍然成立？**
> 是 → W。否 → T。

| 例 | 判定 |
|---|---|
| "按钮当前处于激活状态" | W-A（即使没有任务，按钮也确实激活了）|
| "球在杯子里" | W-R（即使没有任务，球确实在杯子里）|
| "成功条件是球在杯子里" | T-G（只在特定任务下才有意义）|
| "已经完成两个子目标" | T-Q（只在特定任务下才能定义子目标）|
| "电水壶已经通电过" | W-A |
| "agent 当前要执行第几个 sub-task" | T-Q |

**为什么这个区分重要**：
- W 缺失 → 解决方案是"环境感知 + 状态记忆"模块（如 vision encoder + RNN）
- T 缺失 → 解决方案是"任务追踪 + plan 记忆"模块（如 hierarchical planner + sub-goal stack）
- 实验 finding "baseline 在 W 上 fail vs 在 T 上 fail" 会直接指向不同的架构改进方向

---

## 五、reveal-memory 双轨设计

membench 的关键工程产物：**同一批 episode spec，录两种 variant**。

| Variant | observation 中是否包含缺失变量 | 用途 |
|---|---|---|
| `memory` | **否**（必须靠历史恢复） | 主评估 |
| `reveal_all` | **是**（当前帧或当前 prompt 可读） | 诊断瓶颈 |

**诊断逻辑**：

```
if reveal_all 成功 AND memory 失败:
    → 瓶颈是 memory（这正是 paper 要证明的）
if reveal_all 失败:
    → 瓶颈在 low-level skill 而非 memory，要改 spec 或换 baseline
if reveal_all 和 memory 都成功:
    → 任务太容易，没有 memory demand
if reveal_all 和 memory 都失败:
    → 任务太难，baseline 能力天花板低于 task
```

→ **这是 paper §4 falsifiable prediction 的实验基础**。

### Reveal method 的几种实现

membench 的 `reveal_method` 字段（来自 schema.py）：

| method | 内容 | 适用 |
|---|---|---|
| `language_prompt` | 直接在 prompt 里说 "cup_1 是 heated cup" | W-A、T-G（最容易实现）|
| `object_target_markers` | 在桌面贴构型轮廓标记 | T-G 成功构型 |
| `progress_markers` | 显示可翻动的进度标记 | T-Q |
| `return_route_markers` | 把轨迹画在桌面 | T-G 路径约束 |
| `transparent_cabinet_door` | 透明柜门让 inside 关系可见 | W-R |
| `task_progress_board` | 任务进度板 | T-Q-2 |

---

## 六、Hidden Variable 工程编码

membench 的 episode spec 把每个隐藏变量显式标注（见 schema.py / hot_cup spec）：

```json
{
  "hidden_variables": [
    {
      "id": "heated(cup_0)",
      "axis": "W-A",
      "value": true,
      "visible_until_phase": "heating",
      "required_from_phase": "serving"
    }
  ]
}
```

| 字段 | 含义 | 对 wrapper 的作用 |
|---|---|---|
| `id` | 变量名 | metric collector 用它 attribute failure |
| `axis` | W-X / W-A / W-R / T-G / T-Q | failure mode 雷达图维度 |
| `value` | 真值 / 描述 | env oracle 接口 + reveal_all 显示什么 |
| `visible_until_phase` | 信息在哪一阶段从 obs 中消失 | trigger memory store |
| `required_from_phase` | 信息在哪一阶段重新必需 | trigger memory recall |

**`visible_until_phase` + `required_from_phase` 这两个字段是关键**——它们形式化定义了"信息消失"的时机，是 W-anchor 概念的精确版本。

---

## 七、Membench 9 个任务的 W/T 标注矩阵

| 设计编号 | 任务名 | 主诊断 | 隐藏变量 | 信息消失模式 |
|---|---|---|---|---|
| W-X-1 | ReturnOrigin | W-X | `original_position(obj)` | 物体被搬走后原位置无标记 |
| W-A-1 | HotCupCoolingCleanup | W-A | `heated(cup_i)` | 两杯最终外观一致 |
| W-A-2 | ChargedBattery | W-A | `charged(battery_i)` | 三电池外观一致 |
| W-R-1 | ThreeCupsHiddenObject | W-R + W-X | `covered_by(obj,cup_i)` + `current_slot(cup_i)` | shell game |
| W-R-1' | RetrieveFromHiddenShelf | W-R | `inside(obj, shelf_i)` | 不透明柜门 |
| W-X-3 | PutObjectInDrawer | W-X | `storage_drawer(cube)` | 抽屉关闭后内容不可见 |
| T-G-1 | ReproduceConfiguration | T-G | `desired_layout` | 早期构型图消失 |
| T-G-2 | ReturnSameWayAroundPosts | T-G | `chosen_route` | 路线 demo 消失 |
| T-Q-1 | AlternateButtonSequence | T-Q | `current_step ∈ {1..4}` | 按钮复位 + 无进度卡 |
| T-Q-2 | **DeferredTeaCleanup** | T-Q | `pending_followup` + `resume_subgoal` + `interrupted_task` | **多任务交错 + 中断恢复** |

→ housemem 的"K 个多任务交错"**本质上是 T-Q-2 的结构化扩展**：把 K=2（A+B）扩到 K=3、K=4，且每个子任务自身可以承载 W-X/W-A/W-R 维度。

---

## 八、对 HouseMem-Bench 的迁移路径

### 8.1 保留（100% 继承）

1. **W/T 形式化**：$z_t = (w_t, \tau_t)$，$w_t = \{X_t, A_t, R_t\}$，$\tau_t = \{G, Q_t\}$
2. **信息消失原则**：memory demand 来自"当前观测不足以决定最优动作"
3. **W/T 边界判定**：任务不在时是否仍然成立
4. **reveal-memory 双轨**：每个 episode 录两版，诊断瓶颈
5. **hidden variable 显式标注**：`id / axis / value / visible_until_phase / required_from_phase`

### 8.2 修改（适配 RoboCasa 厨房 + 移动机械臂）

| 维度 | membench (ManiSkill3 tabletop) | housemem (RoboCasa kitchen) |
|---|---|---|
| Scene | 单 table + 9 个独立场景 | 25 layout × 12 style + 88 fixture |
| Robot | Franka Panda (固定) | PandaOmron / B1 (移动) |
| W-X 例子 | 桌面三个位置之一 | 厨房 zone × fixture × slot |
| W-A 例子 | heated/charged 杯子 | burner_on / kettle_on / cabinet_door_open |
| W-R 例子 | shell game | cross-zone container relations |
| T-G 例子 | 桌面构型 | 多 sub-task 的最终摆放（"晚餐准备"）|
| T-Q 例子 | 按钮序列 / 茶包-收纳 | **K=2/3/4 多任务交错** |

### 8.3 扩展（housemem 新增）

1. **Spatial-memory 子轴**（W-X 子类）：因为有 mobile manipulator，agent 跨 zone 后必须记得"上一个 zone 的状态"
2. **Trigger-driven T-Q**：T-Q-2 用 episode 流程切换，housemem 引入显式 trigger（timer / event / condition / completion）
3. **K-axis 难度梯度**：T-Q 中的 sub-task 数从 2 提到 4
4. **跨 task 的 W 共享**：episode 里 task A 的 W 变量可能影响 task B 的执行（如 burner 被 A 占用，B 必须选另一个）

### 8.4 不沿用 / 重新设计

| 旧 | 不沿用的原因 | 新 |
|---|---|---|
| 9 个独立 family 各做各的 | RoboCasa 已有 65 atomic + 88 fixture，没必要重做 9 个独立 env | **task 库**：~24 个 memory-augmented task，由 atomic 组合而成 |
| `family` 字段强制 1-to-1 映射 | 现实任务往往跨多个 W/T 维度 | hidden_variables 列表可包含多个 axis 标注 |
| Reveal 用静态 marker（标签/透明门） | RoboCasa 没有透明柜门 asset，且 marker 在 1500 scene 里要重复实现 | Reveal 优先用 `language_prompt`（baseline VLM-friendly）|

---

## 九、housemem task 库设计的形式化骨架

每个 housemem task 由以下字段定义（YAML schema 升级）：

```yaml
task_id: BoilEggsWithMemory     # 名字
atomic_sequence:                # 子序列对齐 atomic（demo 复用）
  - { atomic: OpenFridge, stage: 1 }
  - { atomic: PickPlaceFridgeShelfToCounter, stage: 2, args: {target_shelf: shelf_2} }
  - { atomic: CloseFridge, stage: 3 }
  - { atomic: PickPlaceCounterToStove, stage: 4 }
  - { atomic: TurnOnStove, stage: 5 }
hidden_variables:               # 仿 membench 的 W/T 形式化
  - id: original_shelf(eggs)
    axis: W-X
    value: { shelf: shelf_2 }
    visible_until_phase: after_close_fridge
    required_from_phase: never        # 单任务自身不重用
  - id: eggs_on_counter_pos
    axis: W-X
    value: runtime_tracked
    visible_until_phase: after_pick   # 离开后视野外
    required_from_phase: stage_4      # 拿起放进锅时要回忆
wait_points:                   # 多任务交错锚点
  - stage: 5
    trigger: { type: time_based, condition: "burner_on >= 30" }
    resume_to: stage_6
reveal_methods:                # 双轨设计支持
  memory:
    - { variable: original_shelf(eggs), method: none }
  reveal_all:
    - { variable: original_shelf(eggs), method: language_prompt, text: "eggs are on shelf_2" }
success_condition:
  - on(eggs, saucepan_on_burner_X)
  - burner_X.is_on == true at stage_6
```

**这个 schema 把 5 件事统一进一个 YAML**：
1. atomic sequence（demo 复用）
2. W-X/W-A/W-R/T-G/T-Q 隐藏变量声明
3. wait points（K-组合的交错锚点）
4. reveal vs memory 双轨
5. success condition

→ wrapper_design.md §6.6 binding section 之后应该增加 §6.8 hidden_variables section（仿 membench schema），以及 §6.9 reveal_methods section。

---

## 十、下一步设计动作

| 优先级 | 动作 | 依赖 | 时间 |
|---:|---|---|---:|
| 1 | 更新 `docs/wrapper_design.md`：把 hidden_variables / reveal_methods / wait_points 三个 section 加进 YAML schema | 本文档 | 0.5d |
| 2 | 更新 `docs/paper_outline.md` §3：把 W-X/W-A/W-R 和 T-G/T-Q 写进 POMDP 形式化 | 本文档 | 0.5d |
| 3 | 升级 `housemem/spec_loader.py`：把 5 类 hidden variable axis 加入 Pydantic 校验 | (1) | 0.5d |
| 4 | 起草 `docs/task_library.md`：用 65 atomic 组合 ~24 个 memory-augmented task，每个标注 W/T axis 和 reveal method | (1)(2)(3) | 2d |
| 5 | 把 `HS-TQ-S-01` 按新 schema 写成参考 spec | (3)(4) | 0.5d |
| 6 | 起草 `docs/scenarios.md`：4 个 scenario × K=2/3/4 的组合矩阵 | (4) | 1d |

总计：约 5 天，对应 paper deadline 还很充裕。

---

## 附录 A：membench 直接可借鉴的代码资产

| 文件 | 内容 | 借鉴方式 |
|---|---|---|
| `membench/specs/schema.py` | 9 个 task 的 episode spec 验证函数 | 仿照写 housemem Pydantic models |
| `membench/envs/task_progress.py` | TaskStep + TaskProgressChecker | 仿照写 housemem MultiStageDetector |
| `membench/tasks/deferred_tea_cleanup/env.py` | 多任务交错的 env 实现 | 直接参考 K=2 → K=3 的扩展模式 |
| `membench/tasks/hot_cup_cooling_cleanup/env.py` | W-A hidden state 的 oracle/reveal 分支 | 学习 oracle 字段如何避免泄漏到 obs |

## 附录 B：membench 设计教训（不犯重）

1. **避免把 instruction retention 伪装成 T-G**：早期给规则卡后让它消失，看似 T-G 但其实更像 LLM eval。housemem 的 T-G 必须是真正的"成功约束消失"，比如"按记忆中给定的桌面构型摆放"。
2. **避免 oracle 字段泄漏**：`heated_cup_idx` 这类字段绝对不能进入 policy observation，只能保留在 episode metadata 和 result 里。housemem 实现 wrapper 时要严格区分 oracle channel 和 policy channel。
3. **observation aliasing 是 memory demand 的核心**：membench 反复强调"最终视觉一致，但正确动作不同"。housemem 要在 RoboCasa 的视觉多样性中**主动设计 aliasing**（如让两个 stove burner 视觉无法区分）。
