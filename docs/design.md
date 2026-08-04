# 基于 POMDP 的 memory demand 定义

具身智能任务通常可以被建模为 POMDP。智能体在每一步并不能直接观察完整真实状态 $s_t$，只能获得局部观测 $o_t$。但是，在一般情况下，最优动作并不是由当前观测 $o_t$ 单独决定的，而是由真实的、与当前决策相关的隐藏状态决定的。

这里的关键不是“模型有没有看到图像”，而是：

**当前观测中是否包含了做出正确动作所需的全部决策变量。**

如果当前观测 $o_t$ 已经足以恢复这些决策变量，那么任务不产生记忆需求；如果当前观测无法恢复这些变量，而这些变量曾经在历史观测或历史交互中出现过，那么任务就产生了 memory demand。

因此，具身任务中的记忆问题可以定义为：

> 当前观测不足以决定最优动作，智能体必须从历史信息中恢复某些缺失的决策变量，才能做出正确决策。

也就是说，memory demand 不是由任务长度决定的，也不是由是否存在视频历史决定的，而是由一个更本质的问题决定：

> 当前最优动作所依赖的变量，是否已经从当前观测中消失。

---

# 决策变量的两大来源：W 和 T

对于一个具身智能体而言，理论上只要知道两类信息，就可以做出最优决策：

**第一，当前世界到底是什么状态。**
 例如物体在哪里、容器里有什么、哪个按钮被按过、哪个物体已经处理过、某个不可见属性现在是什么状态。这类信息对应 **W-memory**。

**第二，当前任务到底要求我做什么。**
 例如成功条件是什么、应该遵循什么规则、当前任务执行到哪一步。这类信息对应 **T-memory**。

因此，可以把真实的决策状态抽象写成：

$$
z_t = (w_t, \tau_t)
$$

其中，$w_t$ 表示任务相关的世界状态变量，$\tau_t$ 表示任务相关的目标、规则和上下文变量。注意，$\tau_t$ 不一定是物理世界的一部分，但它同样会决定最优动作。

由此得到 W/T 的核心定义：

**W-memory**：当前观测无法恢复某个任务相关的世界状态变量 $w_t$，而该变量会影响当前最优动作，因此智能体必须从历史中记住它。

**T-memory**：当前观测无法恢复某个任务相关的目标、规则或任务上下文变量 $\tau_t$，而该变量会影响当前最优动作，因此智能体必须从历史中记住它。

因此，W/T 的分类标准不是“任务看起来像什么”，而是：

> 缺失的决策变量到底是什么。

---

# W 的细分：世界状态由什么组成

W-memory 是对当前隐藏世界状态的恢复。世界状态有一个关键约束：

> 它描述的是世界现在是什么，而不是任务希望世界变成什么。

在机器人操作场景中，可以采用 object-centric 的方式对世界状态进行分解。

世界首先由“有哪些实体、各实体身份是什么”构成。实体集合是世界状态的底层承载。没有实体，就无法继续描述实体的属性、位置或关系。

先描述单实体再描述多实体

---

## W-X：单实体空间/几何状态记忆

$$
X_t
$$

$X_t$ 描述单个实体的空间状态，例如位置、姿态、朝向、原始位置、被遮挡后的当前位置等。

W-X 对应的问题是：

> 当前观测无法恢复某个实体的空间状态，而该空间状态会决定当前动作。

例如，物体被遮挡后移动到不可见区域，智能体必须记住它的位置才能抓取；或者物体被操作后，任务要求将其放回原始位置。

---

## W-A：单实体物理/功能状态记忆

$$
A_t
$$

$A_t$ 描述单个实体的非几何状态，例如开/关、锁定/解锁、激活/未激活、满/空、加盐/未加盐、冷热状态等。

W-A 对应的问题是：

> 当前观测无法恢复某个实体的内部物理状态或功能状态，而该状态会影响操作结果。

例如，两个按钮当前外观看起来一样，但其中一个按钮处于激活状态；两个抽屉外观看起来一样，但只有一个已经被解锁。

---

## W-R：多实体关系记忆

$$
R_t
$$

$R_t$ 描述实体之间的关系，例如 inside、on、under、occlude、support、contain、contact 等。

W-R 对应的问题是：

> 当前观测无法恢复实体之间的关系，而该关系决定当前应该操作哪个实体或如何操作。

例如，小球被放入某个杯子后，杯子被重新排列，最终三个杯子外观完全相同。此时缺失的不是杯子的单体状态，而是：

$$
inside(ball, cup_i)
$$

---

## W 的整体形式

因此，世界状态可以抽象为：

$$
w_t = \{X_t,\; A_t,\; R_t\}
$$

其完备性来自 object-centric 分解：任何与世界状态有关的决策变量，要么描述“有什么实体”，要么描述“实体自身处于什么状态”，要么描述“实体之间处于什么关系”。

需要注意的是，某些变量看起来像额外类别，但通常可以归入这四类。例如，“按钮已被按下”属于 $A_t$，“球在杯子里”属于 $R_t$，“物体原来在哪里”属于历史中的 $X_t$。

---

# T 的细分：任务上下文由什么组成

T-memory 是对任务定义 $\tau_t$ 的恢复。同一世界状态 $w_t$ 在不同任务上下文下，最优动作可能完全不同。因此，除了知道世界是什么状态，智能体还必须知道当前任务要求什么。

判断一个变量是否构成 T-memory 需要同时满足两个条件：

1. **任务依赖性**：它在任务不存在时不成立，必须依赖任务的存在才有意义；
2. **历史依赖性**：它无法从当前 prompt 或当前观测中读出，必须从历史信息中恢复。

第一条把世界事实排除在 T 之外，例如 `ordered(seat_2, coffee)` 这类订单关系即使在任务中被指称，本身在任务不存在时也仍然成立，属于 W-R 而不是 T。第二条把持续可见的任务目标排除在 T 之外，例如"做一杯咖啡并放到餐桌上"如果一直挂在当前 prompt 中，就不产生 T-memory。

在 T 中真正稳定产生 memory demand 的来源，是那些**随任务执行显式变化、并且天然无法被写回持续 prompt** 的内部状态。本版本把任务上下文分解为三类变量：任务进度 $Q_t$、任务调度 $S_t$，以及成功约束 $G$（仅在特定 instruction retention 设计下产生 memory demand，本基准不作为主测维度）。

---

## T-G：成功约束记忆（边界条目）

$$
G
$$

$G$ 描述任务成功时世界应满足的约束条件，例如球在杯中、抽屉打开、物体排列满足某个目标构型。T-G 的成立条件非常苛刻：它要求 $G$ **在 episode 早期被告知，之后从当前 prompt 和当前观测中消失**。在这一严格定义下，两类常见情况都不产生 T-G：

- **$G$ 持续在 prompt 中可见**，例如"做一杯咖啡并放到餐桌上"、"加热当前 cookware"。此时 $G$ 没有从输入中消失，不存在 memory demand。
- **$G$ 指向的实体是世界事实**，例如"加过糖的杯子"、"被点单的座位"。具体是哪个实体应由 W-memory 或历史交互状态来解析，缺失的是 W 而不是 $G$。

剩下严格符合 T-G 的设计——早期给出构型卡或规则卡、随后该卡片消失——本质上属于 **instruction retention**：它是输入流的人为设计，而不是自然任务中产生的记忆需求。因此 T-G 在本基准中保留为完整性条目，**不作为主测维度**。任务表中标注为 T-G 的任务在新定义下应被视为 instruction retention 的边界 case，本轮不进入正式录制。

---

## T-Q：任务进度记忆

$$
Q_t
$$

$Q_t$ 描述单任务内部的执行阶段，包括当前步骤、已完成子目标集合、已处理物体集合、计数进度、条件分支路径等。它和下面的 $S_t$ 一样**显式随任务执行变化**，并且**天然无法被写回持续 prompt**——这正是它稳定产生 memory demand 的根因。

T-Q 对应的问题是：

> 当前世界状态看起来一样，但任务内部进度不同，因此下一步最优动作不同。

例如，任务要求按照“红、蓝、红、蓝”的顺序循环按按钮。每次按完后，按钮外观和物理状态都恢复一致。此时当前观测无法判断现在该按红还是按蓝，缺失的是任务进度：

$$
Q_t \in \{1,2,3,4\}
$$

T-Q 只覆盖**单任务内部**的步骤进度，不涉及多任务之间的挂起与恢复——后者由 $S_t$ 承载。

---

## T-S：任务调度记忆

$$
S_t
$$

$S_t$ 描述多任务交错执行时的调度状态，包括挂起的 follow-up、被中断的任务、需要恢复的子目标位置。它和 $Q_t$ 一样显式随时间变化、天然无法写回持续 prompt，但表达的不是某个单任务的线性进度，而是**任务之间的关系**：哪个任务尚未完成、之前在做哪个任务、被中断后应该从哪一阶段继续。

T-S 对应的问题是：

> 当前世界状态可见，单任务进度也可以从环境中读出，但智能体不知道还有哪个任务挂起、被打断的任务应该回到哪一阶段，因此下一步动作错误。

例如，机器人先启动泡茶任务（茶包浸泡），随后开始执行收纳任务并被打断。当前观测中茶包仍在杯中、收纳盒仍可见，但智能体必须从历史中恢复：

$$
S_t = \{\text{pending\_followup},\; \text{interrupted\_task},\; \text{resume\_subgoal}\}
$$

T-S 之所以独立于 T-Q，是因为它描述的是 **prospective memory**（未来某时刻要回去完成已挂起的任务）和 **cross-task resumption**（被中断的任务应从哪一阶段继续），而不是单任务内部的步骤计数。

---

## T 的整体形式

因此，任务上下文可以抽象为：

$$
\tau_t = \{ Q_t,\; S_t,\; [G] \}
$$

其中，$Q_t$ 是单任务内部的步骤进度，$S_t$ 是多任务调度状态（挂起、中断、恢复），$G$ 是成功约束。$Q_t$ 和 $S_t$ 都显式随时间变化、且天然无法被写回持续 prompt，是 T-memory 的稳定核心；$G$ 加方括号表示它只在特定 instruction retention 设计下（早期 reveal 后从输入中消失）产生 memory demand，本基准不作为主测维度。

任务上下文对最优动作的影响主要通过三条路径实现：指定当前执行到任务的哪一步（$Q_t$）、指定多任务之间的挂起与恢复状态（$S_t$）、或指定最终需要满足什么成功约束（$G$，仅在特殊输入设计下才缺失）。具体操作哪个实体，通常由这些约束结合当前或历史世界状态来解析。

---

# W/T 的边界

W 与 T 的边界由两个判据**共同**决定。一个变量归 T 需要同时满足：

1. **任务依赖性**：它在任务不存在时不成立；
2. **历史依赖性**：它无法从当前 prompt 或当前观测中读出，必须从历史信息中恢复。

只满足第一条但不满足第二条的信息（持续可见的任务目标），不构成 memory demand，应作为 prompt-resident goal 处理。只满足第二条但不满足第一条的信息（任务不存在时仍然成立的事实），属于 W——即使任务规范中显式指称了它。

例如：

- “按钮当前处于激活状态”即使没有任务也成立，属于 W-A；
- “球在杯子里”是世界事实，属于 W-R；
- “seat_2 点了 coffee”是订单关系，即使任务规范在指称它，它本身在任务不存在时仍然成立，属于 W-R（关系类 `ordered(seat_i, drink_j)`）；
- “做一杯咖啡并放到餐桌上”如果一直挂在 prompt 中，不构成 memory demand，不属于 T；
- “成功条件是球在杯子里”是任务成功约束，且该约束已从当前输入中消失时，属于 T-G（边界条目，接近 instruction retention）；
- “已经完成两个子目标”只在特定任务中有意义且无法写回 prompt，属于 T-Q；
- “泡茶 follow-up 尚未完成、收纳任务应从 B3 继续”只在多任务调度中有意义且无法写回 prompt，属于 T-S。

---

# 完整结构

因此，当前最优动作依赖的隐藏决策变量可以写成：

$$
a_t^* = \pi^*(z_t)
$$

其中：

$$
z_t = (w_t, \tau_t)
$$

并且：

$$
w_t = \{X_t,\; A_t,\; R_t\}
$$

也就是说：

$$
a_t^* \text{ 依赖的隐藏变量 } z_t \subseteq W \cup T
$$

这里最重要的一点是：

> W/T 不是互斥的任务类别，而是缺失决策变量的诊断维度。

一个任务可以主要测试某一类变量，但真实的具身任务往往会同时缺失多个变量。因此，任务设计不应该强行追求“一个任务只属于一个类别”，而应该明确标注每个任务的 hidden decision variables。

也就是说，我们真正需要问的是：

> 当前任务中，智能体到底缺少哪些决策变量？
> 只 reveal 其中一个变量，任务是否仍然不可解？
> reveal 所有必要变量之后，任务是否变成可解？

---

# 任务设计

尽管 W 和 T 分别可以被拆成多个变量，但在真实任务中，这些变量并不是完全独立的。大多数时候，一个具身模型需要的 memory 同时包含 W 和 T 两大类，并且可能同时涉及 W 或 T 内部的多个子变量。

因此，下面的任务不应被理解为互斥类别，而应被理解为不同 hidden-variable set 的实例。这里重新约定：T-G 只表示成功约束本身是否从当前输入中消失；“目标物体到底是哪一个”不作为 T-G 子类，而是由 W-memory 或历史交互变量来解析。

reveal 设计：直接录制两种版本的数据。一种是 Memory 版本，决策时需要从历史中恢复 memory 信息；另一种是 Reveal 版本，决策时不需要 memory，只需要观察当前世界即可获得足够的信息进行决策。

当前核心任务保留 T-G 中的成功构型和成功轨迹任务；但暂不采用我之前补充的“早期给出分拣规则/配方规则，随后规则卡消失”的 W+T-G 组合任务。这类组合虽然形式上满足定义，但更像人为移除 instruction，而不是自然实机任务中的记忆需求。

在设计任务的时候期望能够设计实机的任务。如果可以尽量不要太toy。

## 当前实施顺序

当前数据录制先按 memory-only 版本推进，Reveal 版本暂时作为诊断设计保留，不进入第一轮正式录制。每个任务的 expert demonstration 可以访问 spec / env internal state 来生成成功轨迹，但 model-facing observation 不能直接暴露 hidden variable。

第一轮任务顺序如下：

| 顺序 | 设计编号 | 当前任务 | task id | env id | hidden variable | 当前状态 |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | W-X-1 | ReturnOrigin | `return_the_cup_to_origin_position` | `MemBenchReturnOrigin-v1` | `original_position(object)` | 先录 `memory`；已明确数据字段、spec 逻辑和 oracle 泄漏边界。 |
| 2 | W-A-1 | Hot Cup Cooling and Cleanup | `hot_cup_cooling_cleanup` | `MemBenchHotCupCoolingCleanup-v1` | `heated(cup_i)` | 下一步规划；先补齐 memory 录制口径，避免 `heated_cup_idx` 等 oracle 字段进入 policy observation。 |
| 3 | W-R-1 | Three Cups Hidden Object | `three_cups_hidden_object` | `MemBenchThreeCupsHiddenObject-v1` | `covered_by(hidden_object, cup_i)` + `current_slot(cup_i)` | 工程 MVP 采用 shell-game 版本。 |
| 4 | T-G-1 | Reproduce Configuration | `reproduce_configuration` | `MemBenchReproduceConfiguration-v1` | `desired_layout` | 后续录制。 |
| 5 | T-Q-1 | Alternate Button Sequence | `alternate_button_sequence` | `MemBenchAlternateButtonSequence-v1` | `current_step` | 后续录制。 |
| 6 | T-Q-2 | Deferred Tea and Table Cleanup | `deferred_tea_cleanup` | `MemBenchDeferredTeaCleanup-v1` | `pending_tea_followup`, `cleanup_resume_subgoal`, `interrupted_task` | 后续录制。 |

`charged_battery_dispatch` 当前实现属于 W-A family，但不作为本轮“第二个任务”。本轮第二任务以 design 中的 `W-A-1 / Hot Cup Cooling and Cleanup` 为准。`W-A-2` 的锅/电池类状态任务暂时作为扩展任务保留。

| 编号 | 主测试维度 | 成功约束 G | 缺失的 W 变量 | 缺失的 T 变量 | Memory 版本 | Reveal 版本 |
| --- | --- | --- | --- | --- | --- | --- |
| W-X-1 | W-X：历史空间位置 | 把当前物体放回它最开始所在的位置。 | `original_position(object)` | 无。G 可以在当前 prompt 中保持可见，例如“放回原位”。 | 桌上有三个固定位置。开始时一个杯子、遥控器或钥匙放在其中一个位置。机器人把物体拿到临时托盘后，继续擦桌面、整理其他物体。最终物体和三个位置都可见，但当前观测无法知道物体最初来自哪里。 | 在原始位置保留小标签、轮廓标记或“原位置卡片”。 |
| W-A-1 | W-A：内部/功能状态 | 把之前被加热过的杯子放到隔热服务区。 | `heated(cup_i)` | 无。G 可以在当前 prompt 中保持可见，例如“把加热过的杯子放到隔热服务区”。 | 桌上有两个外观相同的杯子，一个需要加热，一个不需要加热。机器人把目标杯子放到炉上并打开旋钮；加热等待期间，将另一个杯子整理到冷却等待区。加热完成后关闭旋钮，把热杯也移动到等待区，再擦拭炉面。最终两个杯子都在等待区且外观一致，当前观测无法判断哪个杯子曾被加热。 | 在最终查询阶段通过语言显式 reveal `heated(cup_i)`，例如在 prompt 中说明“cup_1 是加热过的杯子”。 |
| W-A-2 | W-A：功能状态 | 把烫的锅放到隔热垫，或指出哪口锅是烫的。 | `hot(pot_i)` | 无。 | 三口锅外观相同。早期只加热其中一口，关火后移动锅具和炉架，最终当前画面中三口锅外观一致。 | 烫的锅显示热敏颜色，或贴“hot”状态标签。 |
| W-R-1 | W-R：隐藏包含关系 | 从储物柜中取回之前放入的物体。 | `inside(object, shelf_i)` | 无。 | 把一个物体放入三层储物柜中的某一层并关门，随后执行整理和移动干扰。最终柜门关闭，当前观测无法知道物体在哪一层。 | 使用透明柜门，或移除 Memory 版本中的遮挡纸。 |
| W-R-2 | W-R：隐藏容器关系 | 取出当前 prompt 指定的物品，例如“取出茶包”。 | `inside(item_j, box_i)` | 无。目标物品名称不作为 memory，当前 prompt 可以一直说“取出茶包”。 | 茶包、勺子、调料包、小饼干被分别放进三个或四个相同盒子。盒盖盖上后盒子可以交换位置。最终只能看到封闭盒子。 | 透明盒盖 reveal `inside` 关系。 |
| W-R-3 | W-R：功能连接关系 | 加热当前放在目标炉口上的锅、水壶或平底锅。 | `controls(knob_i, burner_j)` | 无。目标物体和目标炉口在最终阶段当前可见，G 可以保持为“加热这个物体”。 | 多口炉灶开局随机采样旋钮到炉口的一一映射。机器人依次开关每个旋钮，观察哪个炉口亮起；全部复位后插入清理或取锅等干扰动作。最终环境把 cookware 随机放到一个可见炉口上，机器人必须打开控制该炉口的旋钮。 | 在炉灶面板上显示旋钮到炉口的连线/编号，或在最终阶段高亮正确旋钮。 |
| T-G-1 | T-G：成功构型 | 最终桌面满足早期给定的一组空间约束，例如 `leftOf(eraser, tape)`、`behind(clip, eraser)`。 | 物体当前位置可见，不作为主要缺失变量。 | `desired_layout` | 桌上有五个小物品，例如橡皮、胶带、夹子等。任务给出一个最终构型，随后物品被打乱，机器人还需要完成清空桌面、移动临时托盘、处理无关物体等中间步骤。最终进入摆放阶段时，当前观测中所有物品可见，但目标构型不在当前观测中。 | 桌面上放构型参考照片，或在桌面贴轮廓标记。 |
| T-G-2 | T-G：成功轨迹/路径约束 | 工具移动轨迹必须满足早期给定的路线顺序，例如绕过 2 号柱、4 号柱、1 号柱后到达托盘。 | 柱子和工具当前位置可见。 | `desired_trajectory` | 设置几个柱子和一个工具。任务给出需要复现的轨迹或路线，随后机器人先取工具、移动障碍、打开通道，最后才进入路线执行阶段。此时柱子都可见，但当前观测本身不包含目标轨迹。 | 把轨迹画在桌面上，或保留路线参考卡。 |
| W-X/T-G-1 | W-X + T-G：隐藏位置 + 成功构型 | 最终构型要求某些物体放到特定相对关系中，例如 `on(tool, pad_2)`、`leftOf(cup, tray)`。 | `hidden_position(object_i)` | `desired_layout` | 桌面上有几个工具和垫片。任务先给出最终构型，随后机器人清理桌面并把部分物体临时放到遮挡板后方；中间执行擦桌面、移动托盘等操作后，最终需要按目标构型摆放。此时部分物体当前位置不可见，且当前观测不包含目标构型。 | 透明遮挡板 reveal 物体位置，同时保留构型参考图或桌面轮廓标记。也可以只 reveal 其中一个变量做 ablation。 |
| W-A/T-G-1 | W-A + T-G：内部状态 + 成功构型 | 最终构型要求不同状态的物体处在特定结构中，例如“加过糖的杯子在托盘左侧，纯水杯在托盘右侧”。 | `sweetened(cup_i)` | `desired_state_layout` | 三个外观相同的杯子中，一个早期被加糖，一个保持纯水。任务给出最终状态构型后，机器人继续搅拌、移动杯子、清理桌面，最终进入摆放阶段。此时杯子外观看起来一致，当前观测也不包含目标构型；机器人既要记住哪个杯子加过糖，也要记住成功构型。 | 用颜色或标签 reveal 杯子状态，同时保留构型参考图。也可以只 reveal 状态或只 reveal 构型。 |
| W-R/T-G-1 | W-R + T-G：隐藏关系 + 成功构型 | 最终构型要求被包含关系满足某个结构，例如“装有茶包的盒子放在杯子左侧，装有勺子的盒子放在杯子右侧”。 | `inside(item_j, box_i)` | `desired_relation_layout` | 茶包、勺子、糖包被分别放入外观相同的盒子，盒子盖上后位置可以交换。任务给出最终构型后，机器人继续摆杯子、清理桌面、移动盒子。最终阶段只能看到封闭盒子和杯子，当前观测不包含盒内关系，也不包含目标构型。 | 透明盒盖 reveal `inside` 关系，同时保留构型参考图或桌面轮廓标记。也可以分别 reveal W-R 或 T-G。 |
| T-Q-1 | T-Q：步骤进度 | 按固定循环程序执行，例如红、蓝、红、蓝。 | 按钮当前状态被复位，不提供进度。 | `current_step` | 两个按钮每次按完都自动复位，外观一致。当前观测无法判断现在是第几步，因此不知道下一步该按哪个。 | 放置可翻动的进度标记，显示当前步骤。 |
| T-Q-2 | T-Q：延迟任务与中断恢复 | 先启动泡茶任务，再收纳桌面物品；中途返回处理泡茶 follow-up，随后恢复收纳任务。 | 当前世界尽量控制为不能直接显示泡茶 follow-up 是否待完成，或收纳任务已完成到哪一阶段。 | `pending_tea_followup`，`cleanup_resume_subgoal`，`interrupted_task` | A 任务是泡茶：先把茶包放入杯中开始浸泡，稍后回来取出茶包。B 任务是收纳餐具：按顺序把勺子、糖包、纸巾放入不透明收纳盒，最后把盒子移动到服务区。episode 随机选择在 B1/B2/B3 后中断去完成 A2，回来后必须恢复 B 的正确下一阶段。 | 放置可见任务板，显示泡茶 follow-up 是否 pending/done，以及收纳任务下一步应执行的子目标。 |
| W-A/T-Q-1 | W-A + T-Q：状态解析 + 步骤进度 | 按固定序列服务，例如甜水、纯水、甜水、纯水。 | `sweetened(cup_i)` | `current_step` | 三杯水外观相同，其中一杯早期被加糖。服务序列固定，但没有进度标记；每次服务后杯子外观仍然一致。机器人既要知道哪杯甜，也要知道现在服务到第几步。 | 用标签 reveal 杯子状态，并用进度卡 reveal 当前步骤。 |
| W-R/T-Q-1 | W-R + T-Q：隐藏容器关系 + 步骤进度 | 按固定顺序取物，例如茶包、勺子、糖包。 | `inside(item_j, box_i)` | `current_step` | 茶包、勺子、糖包分别放入相同盒子，盒子盖上并交换位置。机器人每次取出序列中的下一个物品，取完后盒子重新盖上。当前观测既看不到盒内物品，也没有进度标记。 | 透明盒盖 reveal `inside`，并放置进度卡。 |
| W-X/T-Q-1 | W-X + T-Q：隐藏位置 + 当前阶段 | 交替处理 A 和 B，例如先处理 A，再处理 B，重复一轮。 | `hidden_position(object)` | `current_step` | 两个物体被遮挡并在遮挡后移动。任务要求交替取出 A/B 执行操作，但最终每次决策只看到遮挡板和操作区域，既不知道物体隐藏在哪里，也不知道当前该处理 A 还是 B。 | 使用透明遮挡板 reveal 位置，并用进度卡 reveal 当前阶段。 |

## T-Q-2：Deferred Tea and Table Cleanup

### 任务定位

`T-Q-2 / Deferred Tea and Table Cleanup` 用于测试真实生活中常见的延迟任务和中断恢复记忆。任务中的核心隐藏变量是：

```text
pending_tea_followup
cleanup_resume_subgoal
interrupted_task
```

它们表示机器人已经启动了泡茶任务，但取出茶包这一步仍然挂起；同时机器人在收纳桌面物品时被中断，完成泡茶 follow-up 后还必须记得收纳任务应该从哪一阶段继续。该任务主要测试 T-Q 记忆，而不是物体身份、隐藏位置或容器内容关系。

### 场景动机

场景模拟一个自然的桌面服务流程：机器人先开始泡茶，茶包浸泡期间转去收纳桌面上的服务物品。过一段时间后，机器人需要回来取出茶包，避免继续浸泡；取完茶包后，还要回到之前被打断的收纳任务，并从正确阶段继续。

这种流程对应现实生活中的 prospective memory：人类在做一件事时会挂起另一件稍后要完成的事。中断时机不是单纯的按钮序列，而是由“泡茶 follow-up”和“桌面收纳”两个真实任务交叠产生的。

### 任务流程

1. **A1 / Start tea**

   机器人将茶包放入杯中，开始浸泡。完成后 `pending_tea_followup=true`，但当前 prompt 不持续提醒茶包稍后需要取出。

2. **B1 / Store spoon**

   机器人将勺子放入不透明收纳盒。

3. **B2 / Store sugar packet**

   机器人将糖包放入不透明收纳盒。

4. **B3 / Store napkin**

   机器人将纸巾放入不透明收纳盒。

5. **A2 / Remove tea bag**

   episode 随机选择在 `B1`、`B2` 或 `B3` 之后中断收纳任务。机器人返回杯子，将茶包取出并放到废弃区或茶包回收区。完成后 `pending_tea_followup=false`。

6. **Resume B**

   机器人回到收纳任务，并从被中断后的正确下一阶段继续。例如中断发生在 `B2` 之后，则 `A2` 之后必须继续执行 `B3`，而不是重复 `B1/B2` 或直接执行 `B4`。

7. **B4 / Move storage box**

   当勺子、糖包和纸巾都已经按顺序放入收纳盒后，机器人将收纳盒移动到服务区。

允许的示例轨迹包括：

```text
A1 -> B1 -> A2 -> B2 -> B3 -> B4
A1 -> B1 -> B2 -> A2 -> B3 -> B4
A1 -> B1 -> B2 -> B3 -> A2 -> B4
```

### Memory 版本

在 `memory` 版本中，当前观测不直接提供任务进度板。泡茶任务完成 `A1` 后，机器人必须在后续收纳过程中记得茶包还需要取出。收纳盒采用不透明或深盒设计，使勺子、糖包和纸巾放入后不容易从当前视角判断盒内已经有哪些物品。这样在完成 `A2` 后，机器人不能只靠当前观测恢复 `cleanup_resume_subgoal`。

该版本不要求所有物理进度完全不可见；重点是不能提供显式 checklist 或 next-step marker。B 任务的阶段信息应该主要由历史动作序列决定，而不是由当前画面直接读出。

### Reveal 版本

在 `reveal_all` 版本中，环境提供一个可见任务板或进度 marker，显式 reveal 缺失的 T-Q 变量：

```text
tea_followup: pending / done
cleanup_next_step: spoon / sugar_packet / napkin / move_box / done
```

Reveal 版本用于验证：当 `pending_tea_followup` 和 `cleanup_resume_subgoal` 被当前观测直接给出后，任务是否可以在不依赖历史记忆的情况下完成。如果 `memory` 版本失败而 `reveal_all` 成功，说明瓶颈主要来自任务进度记忆，而不是抓取、放置或低层控制。

### 成功与失败条件

成功条件：

```text
A1_completed = true
A2_completed = true
B1_completed = true
B2_completed = true
B3_completed = true
B4_completed = true
B_steps_ordered = true
A2_after_interrupt_point = true
resume_cleanup_subgoal_correct = true
robot_static = true
```

失败条件包括：

- 在 `A1` 之前执行 `A2`，或从未取出茶包。
- 未按 `B1 -> B2 -> B3 -> B4` 顺序完成收纳任务。
- 在勺子、糖包、纸巾都放入收纳盒前移动收纳盒到服务区。
- 完成 `A2` 后没有恢复到正确的 B 子阶段。
- 超时或低层执行失败。

### Episode Spec 草案

```json
{
  "episode_id": 0,
  "seed": 1700000,
  "split": "train",
  "family": "T-Q-2",
  "env_id": "MemBenchDeferredTeaCleanup-v1",
  "task_goal": "Start steeping tea, tidy the table items into the storage box, return to remove the tea bag at the remembered interruption point, then resume the table cleanup.",
  "task_goal_zh": "先开始泡茶，再把桌面物品收进收纳盒；在记住的中断点返回取出茶包，然后继续完成桌面收纳。",
  "variant_support": ["memory", "reveal_all"],
  "interrupt_after_b_step": 2,
  "tea": {
    "cup_pose": {
      "position": [0.12, -0.16, 0.0],
      "quat": [1.0, 0.0, 0.0, 0.0]
    },
    "tea_bag_initial_pose": {
      "position": [-0.08, -0.18, 0.0],
      "quat": [1.0, 0.0, 0.0, 0.0]
    },
    "tea_bag_disposal_pose": {
      "position": [0.22, -0.18, 0.0],
      "quat": [1.0, 0.0, 0.0, 0.0]
    }
  },
  "cleanup_items": [
    {"name": "spoon", "stage": "B1"},
    {"name": "sugar_packet", "stage": "B2"},
    {"name": "napkin", "stage": "B3"}
  ],
  "storage_box": {
    "pose": {
      "position": [0.02, 0.14, 0.0],
      "quat": [1.0, 0.0, 0.0, 0.0]
    },
    "service_pose": {
      "position": [0.22, 0.16, 0.0],
      "quat": [1.0, 0.0, 0.0, 0.0]
    }
  },
  "hidden_variables": [
    {
      "id": "pending_tea_followup",
      "axis": "T-Q",
      "value": "A2_required_after_A1",
      "visible_until_phase": "none",
      "required_from_phase": "after_A1"
    },
    {
      "id": "cleanup_resume_subgoal",
      "axis": "T-Q",
      "value": "runtime_tracked",
      "visible_until_phase": "none",
      "required_from_phase": "after_A2"
    },
    {
      "id": "interrupted_task",
      "axis": "T-Q",
      "value": "cleanup",
      "visible_until_phase": "none",
      "required_from_phase": "after_A2"
    }
  ],
  "success": {
    "type": "deferred_tea_cleanup_completed",
    "place_xy_threshold_m": 0.05,
    "z_threshold_m": 0.04
  },
  "reveal": {
    "reveal_all": [
      {
        "variable": "pending_tea_followup",
        "method": "task_progress_board"
      },
      {
        "variable": "cleanup_resume_subgoal",
        "method": "task_progress_board"
      }
    ]
  }
}
```

### 实现建议

首版可以将收纳盒做成不透明深盒，成功判定不依赖视觉识别盒内物品，而由环境内部记录 `cleanup_completed_steps`。茶包取出可以通过茶包 actor 是否移动到 disposal 区域判定。`interrupt_after_b_step` 在 episode spec 中随机取 `1`、`2` 或 `3`，用于决定 rule policy 的示教轨迹和顺序检查。

## W-A-1：Hot Cup Cooling and Cleanup

### 任务定位

`W-A-1 / Hot Cup Cooling and Cleanup` 用于测试单个物体的隐藏属性记忆。任务中的核心隐藏变量是：

```text
heated(cup_i)
```

它表示某个杯子曾经被放到炉上加热过。该变量是世界状态中的物体属性，而不是任务目标或语言上下文。最终目标始终是把加热过的杯子放到隔热服务区；真正缺失的是“哪个杯子具有 heated 属性”。

### 场景动机

场景模拟一个自然的厨房流程：机器人准备两只杯子，其中一只需要加热，另一只不需要。杯子加热后不能立刻完成服务流程，因为机器人还需要先关闭旋钮、把杯子放到冷却等待区、整理另一只杯子，并擦拭炉面。等清理完成后，机器人再从等待区中选出之前加热过的杯子，将它放到隔热服务区。

这样的设计避免了“为了制造记忆而移动物体”的不自然流程。每一步都有现实动机：

- 打开旋钮和关闭旋钮对应加热设备的安全使用。
- 加热等待期间整理未加热杯子，是厨房流程中的并行整理操作。
- 加热后把热杯放到等待区，表示短暂冷却并释放炉面。
- 擦拭炉面，是加热后的清理操作。
- 最终把热杯放到隔热服务区，是为了安全服务或保护桌面。

### 任务流程

1. **Heat target cup**

   机器人将指定杯子放到炉子上的固定加热位置，打开固定旋钮，使该杯子获得隐藏属性 `heated(cup_i)=true`。加热位置和旋钮位置不随机化，避免把 W-A 状态记忆混入位置/按钮搜索问题。

2. **Organize other cup**

   在加热等待期间，机器人将另一个未加热杯子整理到冷却等待区。未加热杯子的 cooling slot 在两个候选 slot 中按 episode 随机采样；加热杯后续放到剩下的 slot。这样最终两个杯子都在 cooling area，但“哪个 slot 是热杯”不会固定。

3. **Turn off**

   加热完成后，机器人关闭旋钮。旋钮状态本身不是最终记忆变量，只是让加热流程更符合真实厨房操作。

4. **Cooling**

   机器人将加热过的杯子从炉上移动到冷却等待区的剩余 slot。此时两个杯子都在等待区，位置按 episode 随机交换，避免模型只学习固定位置偏置。

5. **Clean**

   机器人拿取清洁工具，擦拭炉面上刚才放杯子的固定加热位置。擦拭区域与 `heating.cup_position` 对齐，不额外随机化。该步骤提供自然的中间操作，使最终查询与加热事件之间存在时间间隔。

6. **Query / serve**

   机器人收到固定目标：“Place the heated cup on the heat-resistant serving area.” 机器人必须从两个外观相同的杯子中选出之前被加热过的杯子，并放到隔热服务区。

### Memory 版本

在 `memory` 版本中，最终查询阶段不提供哪个杯子被加热的信息。两个杯子在视觉上保持相同的形状、材质、颜色和可见状态；当前观测无法恢复 `heated(cup_i)`。正确动作必须依赖历史中的加热事件。

最终指令可以保持为：

```text
Place the heated cup on the heat-resistant serving area.
```

### Reveal 版本

在 `reveal_all` 版本中，环境直接提供缺失的 W-A 变量。当前实现采用语言 reveal：最终 prompt 直接说明哪个杯子是加热过的杯子，例如：

```text
cup_1 is the heated cup. Place cup_1 on the heat-resistant serving area.
```

Reveal 版本的作用是验证：当 `heated(cup_i)` 被显式给出后，任务是否可以在不依赖历史记忆的情况下完成。如果 `memory` 版本失败而 `reveal_all` 版本成功，说明瓶颈主要来自 W-A hidden attribute memory，而不是抓取、放置、擦拭或低层运动控制。

### 成功与失败条件

成功条件：

```text
heated(cup_i) = true
on(cup_i, heat_resistant_serving_area)
stovetop_cleaned = true
robot_static = true
```

失败条件包括：

- 将未加热杯子放到隔热服务区。
- 没有完成炉面清洁。
- 没有把任何杯子放到隔热服务区。
- 超时或低层执行失败。

### 实现对应关系

当前实现使用以下标识：

- task id：`hot_cup_cooling_cleanup`
- env id：`MemBenchHotCupCoolingCleanup-v1`
- memory variant：`memory`
- reveal variant：`reveal_all`
- hidden variable：`heated(cup_i)`
- reveal method：`language_prompt`

Episode spec 中应保证 `heated_cup` 与最终等待区 slot 独立随机，从而形成 observation aliasing：最终视觉观测高度相似，但正确目标杯随历史加热事件变化。具体约定是：

- `heated_cup` 在 `cup_0 / cup_1` 中随机采样。
- 未加热杯子的 cooling slot 在两个候选 slot 中随机采样。
- 加热杯放到剩余 cooling slot。
- 初始杯位仍随机采样，但带首个搬运动作距离约束：加热杯初始位置到固定加热点不超过 `0.50 m`，未加热杯初始位置到其 cooling slot 不超过 `0.40 m`。这样保留空间多样性，同时避免 expert demonstration 中出现过大的机械臂绕行。
- 两个 cooling slot 的固定布局也带末段约束：任一 cooling slot 到 serving area 的平面距离不超过 `0.27 m`。这样加热杯最后从 cooling slot 放到 serving area 时不会出现长距离、大幅度搬运动作。
- 炉上放杯子的加热位置固定为画面左下的 burner。
- 需要拨动的旋钮固定为 `front_left`。
- 清洁工具擦拭位置固定为炉上放杯子的加热位置，即 `cleaning.wipe_position == heating.cup_position`。

### 当前录制规划

本任务是当前第二个正式规划任务。第一轮只录 `memory` 版本：

```text
task_id = hot_cup_cooling_cleanup
branch = memory
variant = memory
```

专家演示必须能够读取 spec / env internal state 中的 `heated_cup`，否则无法稳定生成成功轨迹；但保存给模型的 observation 不能包含直接答案字段。特别需要避免以下字段进入 model-facing trajectory observation：

```text
heated_cup_idx
heated_cup_pose
heated_cup_name
```

用于复现和验收的 metadata 仍然应该保留：

```text
episode_spec.heated_cup
episode_spec.hidden_variables
result.heated_cup_name
result.nonheated_cup_name
result.heating_pos
result.heated_cooling_pos
result.nonheated_cooling_pos
```

因此录制前需要确认：

1. scripted expert 可以从 env internal state 或 `episode_spec` 读取目标状态。
2. `trajectory.h5/obs` 中不泄漏 `heated_cup_idx`、`heated_cup_pose` 这类 oracle 字段。
3. memory 视频中两个杯子最终外观一致，不能从当前帧直接判断哪个杯子曾被加热。
4. `episode.json` / `result` 中保留完整验收字段，用于判断 expert demonstration 是否真的服务了加热过的杯子。

## W-R-1 MVP：Three Cups Hidden Object

当前第三个任务的工程 MVP 采用“三个杯子盖住物体并交换位置”的 shell-game 场景，而不是先做三层柜子。这样低层几何更轻，能够更快验证 `W-R` 关系记忆是否成立。

核心关系变量：

```text
covered_by(hidden_object, cup_i)
```

由于盖住物体后杯子会交换位置，任务还包含一个辅助空间/身份追踪变量：

```text
current_slot(cup_i)
```

因此该任务主诊断维度是 `W-R`，同时标注 secondary `W-X`。这不是缺陷；它正是“调换杯子后找出藏物体的杯子”这个任务自然需要的 hidden-variable set。

### MVP 任务流程

1. **Cover**

   桌面上有三个外观相同的倒扣杯子 `cup_0 / cup_1 / cup_2` 和一个小物体 `hidden_object`。episode 随机采样一个杯子盖住物体，例如：

   ```text
   covered_by(hidden_object, cup_1)
   ```

2. **Shuffle**

   三个杯子保持倒扣并交换位置。spec 中记录每个杯子的 final slot，并强制盖住物体的杯子移动到不同 slot，避免 identity shuffle。

3. **Query**

   最终 prompt 固定为：

   ```text
   Find the hidden object and place it on the retrieval mat.
   ```

   Memory 版本中，当前画面只有三个外观一致的杯子，物体不可见；正确动作必须从历史恢复哪个杯子盖住物体，以及该杯子最终移动到了哪里。

4. **Retrieve**

   机器人移开正确杯子，物体变为可取，然后把物体放到取回垫上。

### Memory / Reveal

Memory 版本：

```text
three opaque cups visible
hidden_object not visible
no marker on the correct cup
```

Reveal 版本：

```text
the correct cup has a visible reveal marker
```

第一版 reveal 用标记而不是透明杯子，目的是先跑通任务逻辑。后续如果透明材质稳定，可以替换为透明杯/可见物体 reveal。

### 当前实现标识

- task id：`three_cups_hidden_object`
- env id：`MemBenchThreeCupsHiddenObject-v1`
- family：`W-R-1`
- primary hidden variable：`covered_by(hidden_object, cup_i)`
- secondary hidden variable：`current_slot(cup_i)`
- spec path：`data/episode_specs/three_cups_hidden_object/reveal/{train,val,test}.json`

## W-R-3：Stove Knob-Burner Mapping

### 任务定位

`W-R-3 / Stove Knob-Burner Mapping` 用于测试厨房设备中的功能连接关系记忆。任务中的核心隐藏变量是：

```text
controls(knob_i, burner_j)
```

该变量表示某个炉灶旋钮控制哪个炉口。它不是任务目标，也不是某个旋钮自身的状态，而是两个实体之间的功能关系。即使没有当前任务，`controls(knob_i, burner_j)` 作为厨房设备连接关系仍然成立，因此属于 `W-R`。

这个任务比单纯的 `inside / on / under` 更能覆盖 W-R 的功能关系侧：智能体必须通过早期试验观察建立关系，之后在当前观测无法直接读出映射时恢复该关系。

### 与 T-G / W-A 的边界

本任务不是 `T-G`。最终目标可以一直保持可见和固定，例如：

```text
Heat the cookware on the occupied burner.
```

智能体缺失的不是“任务要求加热什么”，而是“哪个旋钮会点亮当前 cookware 所在炉口”。

本任务也不是 `W-A`。缺失的不是 `knob_i` 自身是否可用、是否开启、是否坏掉，而是：

```text
knob_i -> burner_j
```

这种跨实体连接关系。为了避免混入 `W-X`，最终阶段 cookware 放在哪个炉口上必须当前可见；为了避免混入 `W-A`，所有旋钮和炉口在 probe 结束后都恢复关闭状态。

### RoboCasa 可行性

RoboCasa 原生 `Stove` fixture 已经支持多炉口和多旋钮：

```text
front_left
front_right
rear_left
rear_right
front_center / rear_center  # 某些 stove asset 支持
```

并且已有接口：

```text
stove.set_knob_state(env, rng, knob, mode)
stove.get_knobs_state(env)
stove.is_burner_on(env, burner_loc)
stove.check_obj_location_on_stove(env, obj_name)
```

已有 atomic task：

```text
TurnOnStove
TurnOffStove
PickPlaceCounterToStove
PickPlaceStoveToCounter
```

需要注意：RoboCasa 原生实现中通常是同名固定映射，例如 `knob_front_left` 控制 `burner_front_left`。本任务需要由 MemBench / HouseMem wrapper 在 episode 内额外维护随机逻辑映射：

```text
knob_to_burner_map: Dict[knob_id, burner_id]
```

在 model-facing observation 中不暴露该映射。环境内部可以用该映射决定 probe 阶段哪个炉口亮起，以及最终 success 判断哪个旋钮才是正确旋钮。

### 任务流程

1. **Sample mapping**

   每个 episode 随机采样一个旋钮到炉口的一一映射。例如四口炉：

   ```text
   controls(knob_front_left,  burner_rear_right)
   controls(knob_front_right, burner_front_left)
   controls(knob_rear_left,   burner_front_right)
   controls(knob_rear_right,  burner_rear_left)
   ```

   映射可以是任意 permutation，但应避免长期固定为 RoboCasa 默认的同名映射。

2. **Probe**

   机器人依次测试每个旋钮：

   ```text
   TurnOnStove(knob_i)
   observe mapped burner_j lights up
   TurnOffStove(knob_i)
   ```

   该阶段建立 `controls(knob_i, burner_j)` 关系。probe 结束后，所有旋钮与炉口都处于关闭状态。

3. **Distractor**

   插入一个低风险厨房动作，例如：

   ```text
   PickPlaceCabinetToCounter(cookware)
   wipe / clear counter
   move a bowl or utensil to waiting area
   ```

   该步骤只制造自然时间间隔，不应引入新的隐藏关系。

4. **Place cookware**

   环境或机器人把一个 cookware 随机放到某个炉口上：

   ```text
   on(cookware, burner_k)
   ```

   `burner_k` 必须在最终阶段当前可见。这里不是 memory 变量；它只是查询时的当前世界状态。

5. **Query**

   最终指令固定为：

   ```text
   Heat the cookware on the occupied burner.
   ```

   机器人必须打开满足以下关系的旋钮：

   ```text
   controls(knob_i, burner_k)
   ```

### Memory 版本

Memory 版本中，最终查询阶段不能从当前观测恢复 `controls(knob_i, burner_j)`：

```text
all knobs are off
all burners are off before final action
no visible line / label from knob to burner
cookware location on burner_k is visible
```

正确动作必须依赖 probe 阶段观察到的旋钮-炉口映射。

### Reveal 版本

`reveal_all` 版本需要让 `controls(knob_i, burner_j)` 在当前观测中可恢复。可选方式：

- 在炉灶面板上显示旋钮到炉口的连线或编号。
- 在最终阶段高亮正确旋钮。
- 在 prompt 中显式说明映射，例如 `knob_front_right controls burner_front_left`。该方式最容易实现，但视觉 reveal 更接近物理场景。

Reveal 版本的诊断意义是：当旋钮-炉口映射当前可见时，任务不再依赖历史记忆；若 Reveal 成功而 Memory 失败，瓶颈主要来自 W-R functional relation memory。

### 成功与失败条件

成功条件：

```text
knob_to_burner_map was probed
on(cookware, burner_k) at query time
opened_knob = inverse_map[burner_k]
burner_k becomes on / cookware is heated
wrong burners remain off
robot_static = true
```

失败条件包括：

- 打开错误旋钮，导致非目标炉口亮起。
- 目标 cookware 所在炉口没有被点亮。
- 在 probe 阶段没有完整建立映射就直接进入最终加热。
- 最终 cookware 位置不可见，导致任务混入 `W-X`。
- 超时或低层执行失败。

### Episode Spec 草案

建议新增任务 id：

```text
stove_knob_burner_mapping
```

建议新增环境 id：

```text
MemBenchStoveKnobBurnerMapping-v1
```

核心 spec 字段：

```text
knobs: [front_left, front_right, rear_left, rear_right]
burners: [front_left, front_right, rear_left, rear_right]
knob_to_burner_map: Dict[str, str]
query_burner: str
query_cookware: str
correct_knob: str
variant: memory | reveal_all
```

第一版可以只使用四口炉。若后续选择支持六口炉资产，再扩展到 `front_center / rear_center`，但不应在 MVP 中增加不必要的低层难度。

## W-R-1 备选：Retrieve Object from Hidden Shelf

### 任务定位

该方案是 `W-R-1` 的备选实机场景规划。当前工程第三个任务优先采用上面的 `Three Cups Hidden Object` shell-game MVP；本节保留为后续更接近储物柜实机流程的扩展方案。`W-R-1 / Retrieve Object from Hidden Shelf` 用于测试多实体关系记忆，核心隐藏变量是：

```text
inside(object, shelf_i)
```

这里缺失的不是目标物体身份，也不是成功约束。目标始终是“取回之前放入柜子的物体”；真正需要记住的是物体和某一层柜格之间的包含关系。该关系即使没有当前任务也仍然是世界事实，因此属于 W-R。

### 场景动机

场景模拟一个自然的收纳流程：机器人把一个小物体暂存进三层储物柜中的某一层，关门后继续整理桌面或处理无关物体。最终用户要求机器人取回这个物体。由于柜门关闭、三层柜格外观一致，当前观测无法判断物体在第几层；正确动作必须依赖之前放入时形成的 `inside(object, shelf_i)` 关系。

相比 `W-A-1`，这个任务不再追踪单物体属性，而是追踪跨实体关系：

- `object`：被暂存的小物体，例如茶包、海绵、小方块或钥匙状物。
- `shelf_i`：三层柜格之一，例如 `shelf_0 / shelf_1 / shelf_2`。
- `inside(object, shelf_i)`：最终开柜取物时必须恢复的隐藏关系。

### MVP 任务流程

1. **Store**

   机器人把目标物体放入某一层开放柜格。`shelf_i` 按 episode 随机采样。该阶段建立隐藏关系 `inside(object, shelf_i)`。

2. **Close / occlude**

   机器人或环境脚本关闭柜门，或将不透明遮挡板移动到柜子前方。Memory 版本中，三层柜格从当前视觉上不可区分；Reveal 版本中，柜门透明或保留可见窗口。

3. **Distractor cleanup**

   机器人执行一个低风险中间动作，例如移动托盘、擦拭桌面、把无关物体放到等待区。该步骤的目的不是制造新的 memory 变量，而是让最终查询与存放事件之间有自然间隔。

4. **Query / retrieve**

   当前 prompt 保持固定：“Retrieve the stored object from the cabinet.” 机器人需要打开正确柜格并取回目标物体，放到指定取回区域。

### Memory 版本

在 `memory` 版本中，最终查询阶段不能从当前观测恢复 `inside(object, shelf_i)`：

- 柜门或遮挡板不透明。
- 三个柜格外观和外部标记保持一致。
- 当前 prompt 不说明物体在哪一层，只说明要取回之前存放的物体。

最终指令可以保持为：

```text
Retrieve the stored object and place it on the retrieval mat.
```

### Reveal 版本

`reveal_all` 版本需要让 `inside(object, shelf_i)` 在当前观测中可恢复。优先采用视觉 reveal，而不是语言直接报答案：

- 使用透明柜门，能看见物体所在层。
- 或移除 Memory 版本中的遮挡纸/挡板。
- 如果透明材质在渲染或碰撞上不稳定，再退回到层级标签 reveal，例如正确层门上显示小白点。

Reveal 版本的诊断意义是：当包含关系在当前帧可见时，任务应不再依赖历史记忆；若 Reveal 成功而 Memory 失败，瓶颈主要来自 W-R hidden relation memory。

### 成功与失败条件

成功条件：

```text
inside(object, sampled_shelf_i) was established
object_on_retrieval_mat = true
opened_shelf = sampled_shelf_i
robot_static = true
```

失败条件包括：

- 打开错误柜格并取错物体，或没有取到目标物体。
- 将目标物体放到取回区域以外。
- 在未建立 `inside(object, shelf_i)` 关系前直接跳到取回。
- 超时或低层执行失败。

### Episode Spec 草案

建议新增任务 id：

```text
retrieve_object_from_hidden_shelf
```

建议新增环境 id：

```text
MemBenchHiddenShelfRetrieval-v1
```

spec 放置路径：

```text
data/episode_specs/retrieve_object_from_hidden_shelf/reveal/{train,val,test}.json
```

单条 spec 的核心字段：

```json
{
  "episode_id": 0,
  "seed": 300000,
  "split": "val",
  "family": "W-R-1",
  "env_id": "MemBenchHiddenShelfRetrieval-v1",
  "task_goal": "Retrieve the stored object and place it on the retrieval mat.",
  "task_goal_zh": "取回之前放入柜子的物体，并把它放到取回垫上。",
  "variant_support": ["memory", "reveal_all"],
  "object": {
    "name": "stored_object",
    "asset": "ycb:036_wood_block"
  },
  "cabinet": {
    "shelves": [
      {"name": "shelf_0", "position": [0.04, -0.12, 0.06]},
      {"name": "shelf_1", "position": [0.04, 0.00, 0.12]},
      {"name": "shelf_2", "position": [0.04, 0.12, 0.18]}
    ],
    "target_shelf": "shelf_1"
  },
  "retrieval_area": {
    "position": [-0.18, 0.16, 0.0]
  },
  "hidden_variables": [
    {
      "id": "inside(stored_object,shelf_1)",
      "axis": "W-R",
      "value": {
        "object": "stored_object",
        "relation": "inside",
        "container": "shelf_1"
      },
      "visible_until_phase": "store",
      "required_from_phase": "retrieve"
    }
  ],
  "success": {
    "type": "stored_object_retrieved",
    "target_position": [-0.18, 0.16, 0.0],
    "xy_threshold_m": 0.055,
    "z_threshold_m": 0.04,
    "requires_correct_shelf": true
  },
  "reveal": {
    "reveal_all": [
      {
        "variable": "inside(stored_object,shelf_1)",
        "method": "transparent_cabinet_door",
        "target_shelf": "shelf_1"
      }
    ]
  }
}
```

### 实现对应关系

建议新增模块：

- `membench/envs/hidden_shelf.py`
- `membench/rules/hidden_shelf.py`
- `scripts/generate_hidden_shelf_specs.py`
- `scripts/preview/preview_hidden_shelf_camera.py`

并在以下位置接入：

- `membench/envs/__init__.py` import 新环境，确保 ManiSkill 注册。
- `membench/tasks/registry.py` 新增 `retrieve_object_from_hidden_shelf`。
- `membench/specs/schema.py` 新增 `W-R-1` spec 校验分支。
- `membench/recording/runner.py` 如需任务专属 demo style，再加最小分支；否则沿用通用 rule policy 初始化。

### 首轮工程策略

第一版应优先降低低层难度，把 W-R 诊断做干净：

- 柜体先做成简化几何体：三层开放格 + 可隐藏/透明的前挡板。
- 门不必先做真实铰链；可以用 kinematic panel 表示关闭遮挡。
- 目标物体先选稳定、易抓、姿态不敏感的 YCB 物体，例如 `ycb:036_wood_block` 或 `ycb:061_foam_brick`。
- 取回动作可以先不要求真实“开门”，而是接近正确层后由规则轨迹取物；等成功率稳定后再增加门/抽屉交互。
- `target_shelf` 与最终取回区域、干扰动作参数独立随机，避免位置偏置。

### 验收标准

MVP 完成时应满足：

- `memory` 与 `reveal_all` 使用同一批 episode specs 和同一成功条件。
- `reveal_all` 当前帧可恢复 `inside(object, shelf_i)`。
- `memory` 最终当前帧不可恢复 `inside(object, shelf_i)`。
- rule policy 能在 `reveal_all` 或 oracle 信息下稳定完成至少 `val` 前 3 个 episode。
- 录制输出包含 `third / wrist / head` 三视角视频、`trajectory.h5`、`episode.json` 和 attempt log。
