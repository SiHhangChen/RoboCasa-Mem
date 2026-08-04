# MobileMemBench

MobileMemBench 是一个基于 RoboCasa、robosuite 和 MPlib 的移动操作记忆基准。
本仓库只保存基准代码、配置、测试和文档；仿真资产、第三方源码、数据集、模型权重与运行输出均不上传。

## 仓库内容与目录组织

主要目录如下：

- `bench/`：共享任务解析/runtime、专家执行、录制、策略无关评测和策略协议代码。
- `configs/`：任务与评测配置。
- `scripts/`：录制、规划、训练、服务和评测入口。
- `tests/`：不依赖运行输出的自动化测试。
- `docs/`：架构、任务定义和实验计划文档。

RoboCasa、robosuite 和策略仓库各自保留独立的 Git 历史，并在本仓库中被 `.gitignore` 忽略。
完整的本地开发目录应组织为：

```text
MobileMemBench/
├── bench/
├── configs/
├── docs/
├── scripts/
├── tests/
├── robocasa/             # 单独克隆，不提交到本仓库
├── robosuite/            # 单独克隆，不提交到本仓库
└── policy/               # 可选；每种策略使用自己的环境
    ├── openpi/
    ├── lerobot/
    └── AC-DiT/
```

以下内容只保留在本地：`outputs/`、`lerobot_data/`、`ckpts/`、`paper_rewriting_output/`、
仿真资产、视频、HDF5、Parquet、模型权重、缓存和日志。

## 运行环境结论

当前基准的仿真、任务执行、数据录制和评测客户端使用 Conda 环境 `robocasa`。
MPlib 规划器按任务配置在独立的 `membench` 环境中作为子进程运行；如果两个依赖已经安装在
同一个环境，规划器会自动使用当前 Python。ACT、OpenPI、LeRobot 或 AC-DiT 的训练与策略服务端
依赖较重，应使用各自独立的环境；评测客户端通过 WebSocket 与策略服务端通信，不要求二者安装在同一个环境或同一台机器上。

当前开发机已经验证的核心版本如下：

| 组件 | 版本 |
| --- | --- |
| Python | 3.11 |
| RoboCasa | 1.0.1 |
| robosuite | 1.5.2 |
| MuJoCo | 3.3.1 |
| NumPy | 2.2.5 |
| MPlib | 0.1.1 |
| LeRobot（RoboCasa 自带依赖） | 0.3.3 |

## 搭建 RoboCasa 主环境

以下命令从一个空目录开始。需要先安装 Git 和 Conda；无显示器的 GPU 服务器还需要可用的
NVIDIA 驱动和 EGL。

### 1. 克隆本仓库

私有仓库的协作者获得 GitHub 访问权限并配置 SSH key 后，可以执行：

```bash
git clone git@github.com:SiHhangChen/MomaMemBench.git MobileMemBench
cd MobileMemBench
```

### 2. 克隆并固定仿真依赖

主代码会优先从仓库根目录下的 `robosuite/` 和 `robocasa/` 导入源码，因此目录名不能改变。
下面两个提交是当前代码所依据的干净上游版本，本地检查点提交不是运行主基准的必要条件。

```bash
git clone https://github.com/ARISE-Initiative/robosuite.git robosuite
git -C robosuite checkout 232ce7d4a6ed89c949a9aba024a05c8c32fdd08b

git clone https://github.com/robocasa/robocasa.git robocasa
git -C robocasa checkout 8f124f0e263de0c583ba5d2acc87137bfd905986
```

### 3. 创建并安装 Conda 环境

```bash
conda create -c conda-forge -n robocasa python=3.11 -y
conda activate robocasa

python -m pip install --upgrade pip
python -m pip install -e ./robosuite
python -m pip install -e ./robocasa
python -m pip install -e ".[dev]"
python -m pip install "mplib==0.1.1"
```

`robocasa` 的安装过程会固定 MuJoCo、NumPy、Numba 等核心版本，并安装其所需的
LeRobot 0.3.3。不要在这个环境中直接升级到 `policy/lerobot/` 的新版 LeRobot；当前新版源码要求
Python 3.12，并且会与仿真环境产生依赖冲突。

### 4. 初始化宏并下载 RoboCasa 资产

```bash
python -m robosuite.scripts.setup_macros
python -m robocasa.scripts.setup_macros
python -m robocasa.scripts.download_kitchen_assets
```

最后一条命令会交互式确认并下载约 10 GB 的厨房纹理、fixture 和物体模型。资产会进入被忽略的
`robocasa/robocasa/models/assets/`，不会随主仓库上传。可以用下面的命令检查当前任务需要的一个典型资源：

```bash
test -f robocasa/robocasa/models/assets/objects/objaverse/cup/cup_2/model.xml
```

### 5. 选择渲染后端

无显示器服务器通常使用 EGL：

```bash
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export PYTHONPATH=.
```

需要桌面窗口时使用 `MUJOCO_GL=glfw`，并取消之前设置的 `PYOPENGL_PLATFORM`。不建议把这些
变量永久写入全局 shell 配置；为每次运行或项目终端单独设置即可。

### 6. 验证安装

下面的检查会确认 Python 确实从当前目录的两个依赖仓库加载源码，并验证 PandaOmron 控制器配置：

```bash
python - <<'PY'
from bench.env.robocasa_runtime import (
    bootstrap_local_robocasa_sources,
    make_joint_position_controller_config,
)

bootstrap_local_robocasa_sources()
import mujoco
import robocasa
import robosuite

config = make_joint_position_controller_config()
print("RoboCasa:", list(robocasa.__path__))
print("robosuite:", list(robosuite.__path__))
print("MuJoCo:", mujoco.__version__)
print("right arm controller:", config["body_parts"]["right"]["type"])
PY
```

预期路径分别指向当前项目的 `robocasa/robocasa` 和 `robosuite/robosuite`，控制器类型为
`JOINT_POSITION`。随后可运行不需要下载输出或模型权重的测试：

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q
```

## 策略环境与远程评测

策略仓库不是运行基础仿真所必需的，不要将它们整体安装进 `robocasa` 环境。

- OpenPI / π0.5：在 GPU 机器上使用 `policy/openpi/` 自己的 `uv` 环境启动服务端；
  `robocasa` 环境只运行评测客户端。
- LeRobot / ACT：训练、数据转换和纯策略服务建议使用独立的 Python 3.12 环境，并遵循
  `policy/lerobot/README.md`。
- AC-DiT：依赖 ManiSkill、PyTorch 3D 和指定 CUDA 版本，应完全使用其独立环境。

当前 MomaMemBench 版 OpenPI 可以按以下方式放到约定目录；它已经包含本项目的训练适配：

```bash
mkdir -p policy
git clone --recurse-submodules --branch membench-openpi \
    git@github.com:SiHhangChen/openpi.git policy/openpi
git -C policy/openpi checkout 986ddd4e26561c1d488813f1b72d159d8e5435bf
```

正式评测客户端通过版本化的 WebSocket 协议连接策略服务。完整 OpenPI 服务端的安装、
训练和启动应在 `policy/openpi/` 中按它自己的 README 进行。13 个 builtin task 都有
`configs/eval/suites/*_smoke_v1.yaml`；正式 validation/test seed 目前已为 WX-01 冻结。

先做只读 preflight，可以在创建 RoboCasa 环境前检查 task config hash、state/action schema、
图像尺寸、相机、20 Hz 控制频率和 action horizon：

```bash
python scripts/eval/run_remote.py \
    --endpoint ws://<POLICY_SERVER_IP>:8000 \
    --suite configs/eval/suites/wx01_smoke_v1.yaml \
    --preflight-only
```

确认通过后运行闭环评测：

```bash
PYTHONPATH=. MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
python scripts/eval/run_remote.py \
    --endpoint ws://<POLICY_SERVER_IP>:8000 \
    --suite configs/eval/suites/wx01_smoke_v1.yaml
```

每个 episode 原子提交并可按 seed 恢复。`summary.json` 同时报告 planned、recorded、pending、
error/cancelled 和 publishable；中断后可独立重建与校验：

```bash
python scripts/eval/summarize_run.py outputs/eval/<run_id>
```

14 个任务均通过录制/评测共享的 canonical runtime 创建环境，并从同一个
`ResolvedEpisode` 获得初始化和成功语义。当前 v2 policy observation 固定为三路 RGB-D、state37
和 prompt；策略声明不匹配的相机、分辨率、深度或 state/action schema 时会在 preflight 明确失败。

依赖仓库中的实验修改应先在各自仓库建立分支并提交。若某个修改将来变成主基准的必要条件，
应将对应 fork 推送到 GitHub 并在这里更新仓库地址和固定提交，而不是把整个第三方目录复制进主仓库。

当前任务实现及其任务内资源位于 `bench/tasks/<task_id>/`；14 个叶子任务统一暴露
`TASK`、`resolve_episode()`、任务语义和 `expert/` 出口。

### 重构后的两个主入口

```bash
# 只展开专家录制矩阵，不创建模拟器
python scripts/record_entrypoint.py --task-id TM01 --seeds 0-2 --dry-run

# 只做评测 suite 的任务/catalog preflight
python scripts/evaluate.py --suite configs/eval/suites/wx01_smoke_v1.yaml --preflight-only
```

实际 RoboCasa 录制通过 `scripts/record_entrypoint.py` 运行；远程策略闭环仍通过
`scripts/eval/run_remote.py` 运行。两者共享 canonical episode identity 和 RoboCasa runtime
生命周期，不依赖任何旧数据 bundle。

架构冻结接口位于：

- `bench/tasks/contracts.py`：`TaskDefinition`、`EpisodeRequest`、`ResolvedEpisode`。
- `bench/runtime/contracts.py`：`EpisodeRuntime`、`RuntimeStep`、诊断和初始化合同。
- `bench/execution/contracts.py`：专家 `ExpertPlan`、导航/操作阶段和目标合同。
- `bench/policy/protocol/`：唯一 policy protocol 的稳定导入边界。

RoboCasa 主环境仍是 Conda 环境 `robocasa`；任务解析和协议合同测试不需要启动模拟器。

## 录制数据合同（v2）

RoboCasa native episode 和 LeRobot 数据集只保留一个 action 字段：
`actions[T,13]` / `action[13]`。state 使用 `membench.robot_state37.v2`，动作使用
`membench.pandaomron_hybrid13.v2`。每条 transition 保存执行前的 state、三路 RGB-D、时间戳和
action；最后一条 transition 才设置 `terminal=true`。相机还保存每帧内参和 camera-to-world 外参。

固定 observation profile 为 `rgbd_v3`：`robot0_agentview_left`、`robot0_agentview_right`、
`robot0_eye_in_hand`，OpenCV/top-left，RGB-D，224×224。

### Action（单一 13D 向量）

| slice | 轴名 | 语义 | 单位 / 输入范围 | controller / 类型 |
| --- | --- | --- | --- | --- |
| 0:1 … 6:7 | `arm_joint1_target_rad` … `arm_joint7_target_rad` | Panda 绝对关节目标 | rad；由 Panda joint limit 约束 | `right.JOINT_POSITION` / absolute position |
| 7:8 | `gripper_command` | 夹爪开合命令 | normalized `[-1,1]`；open=-1、closed=+1 | `right.GRIP` / command |
| 8:9 | `base_forward_command` | 底盘局部 forward 速度 | normalized `[-1,1]` | `base` / velocity |
| 9:10 | `base_side_command` | 底盘局部 side 速度 | normalized `[-1,1]` | `base` / velocity |
| 10:11 | `base_yaw_command` | 底盘 yaw 速度 | normalized `[-1,1]` | `base` / velocity |
| 11:12 | `torso_delta_command` | torso 相对位移命令 | normalized `[-1,1]`，映射约 `[-0.05,+0.05] m` | `torso` / delta position |
| 12:13 | `base_mode` | 混合控制器模式 | scalar；`>0` 为 base，否则 arm | `hybrid` / controller mode |

轴名、slice、单位、输入范围、controller 和 absolute/delta/velocity 语义都写入 HDF5 与
LeRobot 的 schema metadata；不使用 `action_0` … `action_12` 作为公开字段名。

### State（37D）

```text
arm_joint_qpos[0:7], arm_joint_qvel[0:7],
gripper_qpos[0:2], gripper_qvel[0:2],
base_pos_world[0:3], base_quat_world_xyzw[0:4], base_joint_qvel[0:3],
torso_qpos[0:1], torso_qvel[0:1],
eef_pos_in_base[0:3], eef_quat_in_base_xyzw[0:4]
```

机械臂、夹爪、底盘关节和 torso 从 MuJoCo 实际 qpos/qvel 读取；EEF 位置来自
`robot0_base_to_eef_pos`，EEF 姿态来自 `robot0_base_to_eef_quat_site`。四元数统一为单位
`xyzw`：episode 首帧 dominant absolute component 为正，后续帧若与上一帧点积小于 0 则整体取反。
这只消除 `q/-q` 的表示跳变，不改变实际姿态。

### Policy 与 diagnostics 的边界

策略只能收到 prompt、profile 声明的三路 RGB-D 和 state37；任务目标关系、专家阶段、成功判定
内部状态等 oracle/diagnostic 字段不会进入 policy observation。原生 episode 可以单独保存
任务专属 `oracle` 和结构化 `diagnostics`，用于审核和评测，但 exporter 不会把它们伪装成 policy state。
