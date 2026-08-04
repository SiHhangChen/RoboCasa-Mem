任务分类原理：
Motivation： 现有具身智能评测常把长任务、长视频或复杂操作等同于记忆能力，但任务长并不意味着真正需要记忆。我们从 POMDP 出发，将 memory demand 定义为：正确决策所需的信息曾经出现，但已从当前观测中消失，智能体必须从历史中恢复它。
我们的 W/T 体系提供了一条清晰的设计逻辑：一个智能体要做出正确动作，本质上必须回答两个问题：
1. 世界现在是什么样？（W-memory）
包括物体在哪里（W-X）、处于什么物理或功能状态（W-A），以及物体之间有什么关系（W-R）。
2. 任务现在要求我做什么？（T-memory）
包括单任务执行到了哪一步（T-S），以及多任务中哪些事项仍然挂起、被中断的任务应从哪里恢复（T-M）。
这形成了 benchmark 的核心故事：W 负责记住世界，T 负责记住任务；只有同时知道“世界是什么”和“接下来该做什么”，智能体才能在长期移动操作中持续做出正确决策。 W/T 不是互斥的任务标签，而是对缺失决策变量的诊断维度，一个真实任务也可能同时需要多类记忆。
World：
- WX01: 把杯子放到清洁工具清洁过的位置
- 流程
  - 1.手持清洁工具导航到杯子前
  - 2.擦拭桌面
  - 3.放下清洁工具
  - 4.夹取杯子并放到刚刚擦拭过的地方
- memory：
  - 空间位置记忆，需要记住刚刚擦拭的位置
- success判定：
  - 杯子底部圆心和target的距离不超过2cm
- 操作演示示例:
This content is only supported in a Feishu Docs

- WR01: 把两个杯子分别放到柜子中，然后根据prompt的指示取出对应的杯子
- 流程
  - 1.两个杯子在桌面，随机拿起一个杯子导航到柜子前随机放入上下两个柜子中的一个并关闭柜门
  - 2.导航到杯子前，重复1
  - 3.打开需要的柜门，并取出杯子
- memory：
  - 多实体空间关系记忆，需要记住刚刚目标杯放入了哪个柜子
- success判定：
  - 取出目标杯子
- 操作演示示例:
This content is only supported in a Feishu Docs

- WR02: 把番茄酱和黄芥末酱放到使用区，然后放回其原来的区域
- 流程
  - 1.一个开放的柜台上左边是黄芥末酱右边是番茄酱，机器人导航到中间位置，随机取下一个酱料瓶，导航到餐桌前，随机放在左边或者右边
  - 2.导航到柜台前，取下剩下的酱料瓶，导航到餐桌前，放在另一个位置
  - 3.把两瓶酱料放回正确的区域即可
- memory：
  - 多实体空间关系记忆，需要记住酱料瓶具体是在哪个区域即可
- success判定：
  - 正确放回区域
- 操作演示示例:
This content is only supported in a Feishu Docs

- WA01: 咖啡加糖任务
- 流程
  - 1.站在咖啡机前，有两杯已经做好的咖啡，选择一个咖啡杯，向其中加入糖块
  - 2.随机选择一个咖啡杯将其放到餐桌上，再去咖啡机前拿剩下的咖啡
  - 3.选出正确的加糖的咖啡
- memory：
  - 需要记住往哪个咖啡中加了糖
- success判定：
  - 正确serve加糖了的杯子
- 操作演示示例:
This content is only supported in a Feishu Docs

- WA02: 冷热加水任务
- 流程：
  - 1.站在水龙头开关前，随机按下热水/冷水按钮
  - 2.导航到杯子处，夹取一个杯子进行接水，然后把杯子放到操作台
  - 3.按下另一个按钮，导航到杯子处，夹取剩下的一个杯子进行接水，然后把杯子放到操作台
  - 4.选出想要的冷水or热水杯子
- memory：
  - 需要记住哪个杯子装的是什么水
- success：
  - 正确serve想要的杯子
- 操作演示示例
This content is only supported in a Feishu Docs

Task
- TS01:热巧克力加糖任务
- 流程：
  - 1.机器人面前有一个锅，旁边不远处有一个盘子，里面有N块方糖
  - 2.根据prompt加入k块方糖（N >= K)
  - 3.操作完成后按下按钮，示意结束
- memory：
  - 需要记住已经加了多少块糖，还要加多少块糖
- Success:
  - 按下按钮，且确实加入了k块方糖
- 操作演示示例：
- 
This content is only supported in a Feishu Docs

- TS02:抽屉检查任务
- 流程：
  - 1.机器人按照随机顺序，打开并关上抽屉（一共三个抽屉）
  - 2.操作完成后，按下按钮，示意结束
- memory：
  - 需要记住自己哪些抽屉是已经打开过的
- success：
  - 按下按钮，所有抽屉都被检查且都只被检查了一遍
- 操作演示示例：
- 
This content is only supported in a Feishu Docs

- TM01: 多任务切换
- A1-A2-A3
- B1-B2
A1-B1-A2-A3-B2
      -B2-A2-A3
Calvin memory

- TM02：多任务切换

Method
以我们目前的任务设计而言，我们期望有一个外挂的高层VLM来维护任务进度以及当前任务所涉及的实体状态，并异步给出我们当前的正确的subgoal，底层VLA则利用keyframe机制来生成足够优秀的action，完成subgoal
核心逻辑：一些任务的记忆需求在于无法分清楚正确的subgoal是什么，比如WA01任务：
实体：两个杯子、糖块
阶段：
1.初始：糖块在盘子里（空间+与其他物体的关系），目前没有加入咖啡（功能状态）；杯子1在做咖啡的桌子上，还没有加糖；杯子2在做咖啡的桌子上，还没有加糖
subgoal：给右边杯子加糖

2.加糖：糖块在咖啡杯子里，融化了；杯子1在做咖啡的桌子上，还没有加糖；杯子2在做咖啡的桌子上，已经加糖
subgoal：搬运左边咖啡到餐桌

3.搬运咖啡1: 糖块在咖啡杯子里，融化了；杯子1在餐桌上，还没有加糖；杯子2在做咖啡的桌子上，已经加糖
subgoal：去咖啡机所在的地方搬运第二杯咖啡

4.搬运咖啡1: 糖块在咖啡杯子里，融化了；杯子1餐桌上放在了右边，还没有加糖；杯子2在餐桌上的左边，已经加糖
subgoal：拿起左边的杯子
让模型来做“拿起左边的咖啡”or“拿起右边的咖啡”是简单的，现有的pi05等VLA模型已经证明了这一点。但是对于我们这个任务，输出这个subgoal是困难的，因为决策需要记忆信息。针对这样的任务我们用语言来描述实体的状态，并给出正确的subgoal。
所以我们的planner部分主要就是：实体状态+任务进度->正确的subgoal
再比如我们的WX01任务：
无论是否给实体状态，subgoal都只能是：“把杯子放到刚刚用清洁工具擦拭过的位置”。所以如果要准确的空间位置，一是subgoal需要精确到具体的空间坐标，二是可以采样keyframe来辅助。
所以我们的VLA部分就是：subgoal+观测+keyframe->正确的action
stage1: 训练数据构建
使用coding agent（LVLM）来帮我们构建，对于每一个任务，我们都要分多个阶段，每一个阶段都设定一个清晰的subgoal，除此之外，每一个阶段设定一份任务进度和实体状态
其中任务进度和subgoal的方式类似于MEM：
[Image]
我们维护另一个输出：实体状态，并设定三个语义字段
1.空间状态
2.与其他物体的关系
3.功能状态和重要历史信息
注意，实体状态应该与任务本身强绑定和强关联，比如TS02任务，检查抽屉，抽屉的功能本身应该是装载，但是在这个任务中应该被记录为是否被检查。我们的实体状态只有语言描述
所以我们在stage1构建了一份可以训练VLM的数据。
stage2: 微调VLM
我们选择一个qwen的小模型（2B or 3B or 4B）来作为我们的外挂planner，这个阶段的训练：
输入为：当前任务阶段，实体状态，视觉输入
输出为：下一步任务阶段，实体状态更新，subgoal
stage3：微调pi05
[Image]
将EventVLA的keyframe模块插入PI05，进行微调
输入为：subgoal，当前视觉，keyframes，state， action_noise
输出为：keyframe，action
空间的表示(heatmap,体素）


## 录制数据合同（v2）

这里的“state”是执行 `action_t` 之前测得的机器人状态；“action”是实际传给
RoboCasa `env.step()` 的单一 13D 控制向量。native HDF5 使用 `actions[T,13]`，
LeRobot 使用 `action[13]`，不再拆成 `action_0` … `action_12`。

### Action

动作 schema 为 `membench.pandaomron_hybrid13.v2`：

| slice | 轴名 | 语义 |
| --- | --- | --- |
| 0:7 | `arm_joint1_target_rad` … `arm_joint7_target_rad` | Panda 7 个绝对关节目标，rad，JOINT_POSITION |
| 7:8 | `gripper_command` | 夹爪命令，open=-1、closed=+1 |
| 8:9 | `base_forward_command` | 底盘局部 forward 归一化速度 |
| 9:10 | `base_side_command` | 底盘局部 side 归一化速度 |
| 10:11 | `base_yaw_command` | 底盘 yaw 归一化速度 |
| 11:12 | `torso_delta_command` | torso 归一化 delta，映射约 `[-0.05,+0.05] m` |
| 12:13 | `base_mode` | `>0` 为 base mode，否则 arm mode |

每个轴的名称、slice、单位、输入范围、controller 以及 absolute/delta/velocity 语义都会写入
HDF5 和 LeRobot metadata。所有 action 都在写入和执行前进行 13D/finiteness/range 校验。

### State

state schema 为 `membench.robot_state37.v2`，顺序固定为：

```text
arm_joint_qpos[0:7], arm_joint_qvel[0:7],
gripper_qpos[0:2], gripper_qvel[0:2],
base_pos_world[0:3], base_quat_world_xyzw[0:4], base_joint_qvel[0:3],
torso_qpos[0:1], torso_qvel[0:1],
eef_pos_in_base[0:3], eef_quat_in_base_xyzw[0:4]
```

state 从 RoboCasa MuJoCo 实际 qpos/qvel 和标准 observation keys 提取，不依赖任务自定义
policy state。所有四元数都是单位 XYZW：episode 首帧 dominant absolute component 为正，
后续帧与前一帧点积小于 0 时整体取反，消除 `q/-q` 跳变。

### Observation profile 与边界

所有任务统一使用 `rgbd_v3`：`robot0_agentview_left`、`robot0_agentview_right`、
`robot0_eye_in_hand` 三相机，224×224，RGB-D，OpenCV/top-left；每帧保存内参和
camera-to-world 外参。

policy observation 只包括 prompt、上述三路图像和 state37。任务目标关系、专家阶段、成功判定
内部量属于 oracle/diagnostics，只能作为独立 episode metadata 保存，不能混入 policy state。
terminal 只在最后一条 transition 设置；stage 边界保存在 diagnostics/event 中，而不是额外的 policy
输入维度。
