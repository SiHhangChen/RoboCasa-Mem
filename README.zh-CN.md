# RoboCasa-Mem

[English](README.md) | [简体中文](README.zh-CN.md)

RoboCasa-Mem 基于 RoboCasa 提供九个需要记忆能力的移动操作任务，支持 Pi0.5、ACT、Diffusion Policy，以及兼容协议的自定义策略服务。

每个已记录的评测配置覆盖 50 个随机种子（1050–1099），评测会保存各任务和模型的逐回合结果、阶段报告及汇总。本仓库包含仿真评测代码和配置；仿真器源码、资产、模型实现及检查点需要单独安装或获取。

ModelScope 下载地址：

- [仿真资产](https://modelscope.cn/datasets/ouhebai/RoboCasa-Mem/tree/master/sim-asset)
- [数据集](https://modelscope.cn/datasets/ouhebai/RoboCasa-Mem/tree/master/data)

**发布状态：** 当前版本为 `0.1.0rc1`。移除真机集成后，当前安装包仍需重新完成 wheel 安装验收，详见[验证与发布状态](#验证与发布状态)。

## 任务

| 任务  | 描述                         | 评测配置后缀 |
| ----- | ---------------------------- | ------------ |
| WX-01 | 将物体放回原位               | `wx01`     |
| WA-01 | 将加糖和未加糖的咖啡分类     | `wa01`     |
| WA-05 | 装满一个杯子并匹配温度按钮   | `wa05`     |
| WR-01 | 取回记忆中的罐子             | `wr01`     |
| WR-08 | 将牛奶盒与对应颜色的糖盘匹配 | `wr08`     |
| TS02  | 检查三个柜子                 | `ts02`     |
| TS03  | 向热巧克力中加入指定量的糖   | `ts03`     |
| TM01  | 烧水并检查两个抽屉           | `tm01`     |
| TM02  | 将锅装满并取回罐子           | `tm02`     |

通过 `<模型系列>:<任务后缀>` 选择评测配置，例如 `pi05:wx01`、`act:wa01` 或 `diffusion:ts03`。发布范围仅包含三个模型系列各九个任务，共 27 份正式评测配置，分别放在 `configs/eval/act/`、`configs/eval/diffusion/`、`configs/eval/pi05/`，统一使用 `wa01.yaml` 等任务名。完整映射见 [`evaluation_suite_sources.json`](configs/benchmark/evaluation_suite_sources.json)。

## 环境与终端分工

**`.venv` 是 RoboCasa-Mem 项目运行所需的主 Python 环境。** 本项目的安装、任务仿真、模型评测和结果汇总都在这个环境中执行。评测时打开两个终端，分别激活项目运行环境和 policy 环境：

- **终端 A，项目运行环境（`.venv`，Python 3.11）：** 安装 RoboCasa、robosuite、仿真资产和本仓库。在这里运行 `membench-eval` 和 `membench-summarize`，负责创建任务、渲染观测、执行动作和保存结果。
- **终端 B，policy 环境：** 安装所选模型的实现和本仓库，在这里加载检查点并执行推理。ACT 和 DP 使用 `membench-serve-lerobot`，Pi0.5 使用 `membench-serve-openpi`。Python 和依赖版本按对应模型实现的要求配置。

终端 A 通过 WebSocket 将观测发送给终端 B，接收模型返回的动作。评测期间保持模型服务运行。显存允许时，两个环境可以共用一张 GPU，不要求两张 GPU。仿真端需要 Linux 和支持 EGL 渲染的 NVIDIA GPU。

## 安装项目运行环境（终端 A）

### 1. 创建项目运行环境

```sh
git clone https://github.com/SiHhangChen/RoboCasa-Mem.git
cd RoboCasa-Mem
export ROBOCASA_MEM_ROOT="$PWD"
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

### 2. 安装指定版本的仿真器源码

将 `/path/to/simulator-sources` 替换为本仓库之外的目录。把 `robosuite/` 和 `robocasa/` 放在同一父目录下，以便将资产压缩包恢复到两个源码目录中。

```sh
mkdir -p /path/to/simulator-sources
git clone https://github.com/ARISE-Initiative/robosuite.git /path/to/simulator-sources/robosuite
git -C /path/to/simulator-sources/robosuite checkout 232ce7d4a6ed89c949a9aba024a05c8c32fdd08b
git clone https://github.com/robocasa/robocasa.git /path/to/simulator-sources/robocasa
git -C /path/to/simulator-sources/robocasa checkout 8f124f0e263de0c583ba5d2acc87137bfd905986
python -m pip install -e /path/to/simulator-sources/robosuite -e /path/to/simulator-sources/robocasa
python -m pip install 'mujoco==3.3.1' 'numpy==2.2.5' 'numba==0.61.2' 'scipy==1.15.3' 'Pillow==12.3.0' 'PyOpenGL==3.1.10'
```

在 RoboCasa-Mem 仓库根目录安装评测包：

```sh
python -m pip install -e '.[video]'
```

`video` 可选依赖提供用于视频输出的 OpenCV。指定的仿真器源码提交和已验证的依赖版本记录在 [`simulator_environment.json`](configs/benchmark/simulator_environment.json) 中。MuJoCo 和 NumPy 版本会影响初始化回放的等价性校验。

### 3. 恢复仿真资产

运行仿真前，请从 [ModelScope 仿真资产目录](https://modelscope.cn/datasets/ouhebai/RoboCasa-Mem/tree/master/sim-asset) 下载 `membench-sim-assets.tar.gz`。指定的资产压缩包包含 3035 个文件，大小为 270484609 字节（约 258 MiB），SHA-256 为：

```text
140176b06cf220f6d11440fe3cede54cca7577edf13473d62fb210bd03ca68e3
```

在仓库根目录校验压缩包，解压到仿真器源码的父目录，再校验恢复后的文件：

```sh
python scripts/tools/verify_assets.py --archive /path/to/membench-sim-assets.tar.gz
tar -xzf /path/to/membench-sim-assets.tar.gz -C /path/to/simulator-sources
python scripts/tools/verify_assets.py --root /path/to/simulator-sources
```

两次校验都必须返回 `passed: true`；恢复后的文件校验还必须返回 `checked_files: 3035`。资产清单见 [`simulator_assets.json`](configs/benchmark/simulator_assets.json)。资产目录需要保持可写，因为 RoboCasa 会在其中创建临时物体 XML 文件。

### 4. 配置无窗口渲染

在运行仿真的终端中设置以下变量，并将 EGL 设备编号改为本机可用的编号：

```sh
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export MUJOCO_EGL_DEVICE_ID=0
```

## 启动 policy 服务（终端 B）

### 准备 policy 环境

policy 环境负责提供所选模型的实现和依赖，环境和模型源码统一放在本项目的 `policy/` 目录下。ACT 和 DP 共用 `policy/.venv` 和 `policy/lerobot`；Pi0.5 使用 `policy/openpi/` 下的 OpenPI 环境。每个 policy 环境都需要安装本仓库，以使用对应的服务命令。`policy/` 已被 Git 忽略，不纳入发布包。

对于 ACT 和 DP（Diffusion Policy），下面给出 LeRobot 的参考安装配置，需要预先安装 Python 3.12。将 `ROBOCASA_MEM_ROOT` 设置为终端 A 使用的同一个仓库的绝对路径。固定的 LeRobot 提交来自历史 ACT 评测记录，`diffusers-dep` 可选依赖用于 DP。

```sh
export ROBOCASA_MEM_ROOT=/absolute/path/to/RoboCasa-Mem
cd "$ROBOCASA_MEM_ROOT"
mkdir -p policy
python3.12 -m venv policy/.venv
source policy/.venv/bin/activate
export TORCH_HOME="$ROBOCASA_MEM_ROOT/policy/.cache/torch"
export HF_HOME="$ROBOCASA_MEM_ROOT/policy/.cache/huggingface"
python -m pip install --upgrade pip
git clone https://github.com/huggingface/lerobot.git policy/lerobot
git -C policy/lerobot checkout 64b23178d5348609c266250d3e1f511eba4c33ff
python -m pip install 'torch==2.7.1' 'torchvision==0.22.1' 'numpy==2.2.5' -e 'policy/lerobot[diffusers-dep]'
python -m pip install -e "$ROBOCASA_MEM_ROOT"
python -m pip check
python -c 'from lerobot.policies.factory import make_pre_post_processors; from lerobot.policies.act.modeling_act import ACTPolicy; from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy; print("ACT and DP imports OK")'
```

如果 Git 克隆超时，将上面的克隆及 checkout 命令替换为以下同一固定提交的官方源码下载方式，然后继续执行 pip 安装命令：

```sh
curl -fL --connect-timeout 15 --max-time 90 --retry 1 -o policy/lerobot-64b23178.tar.gz https://codeload.github.com/huggingface/lerobot/tar.gz/64b23178d5348609c266250d3e1f511eba4c33ff
mkdir -p policy/lerobot
tar -xzf policy/lerobot-64b23178.tar.gz -C policy/lerobot --strip-components=1
```

2026-10-26，此共享环境已通过全新安装、依赖一致性和接口导入检查，并使用已训练的 WA-01 检查点和合成观测完成 ACT、DP 的 CUDA 前向推理。仿真试跑情况见[验证与发布状态](#验证与发布状态)。首次加载模型可能会将 ResNet18 骨干网络权重下载到 `TORCH_HOME`。项目运行环境中的 LeRobot 0.3.3 依赖缺少 `make_pre_post_processors`，LeRobot policy 服务应在 policy 环境中运行。

根据待评测的模型选择下面一个示例，示例均使用端口 8000 和 WA-01 检查点。保持所选服务运行，再到终端 A 执行匹配的评测配置。

### ACT 示例

在 policy 环境中运行，将检查点路径替换为实际的 ACT 模型目录：

```sh
membench-serve-lerobot \
  --checkpoint /path/to/act-wa01/checkpoints/040000/pretrained_model \
  --device cuda \
  --port 8000
```

在终端 A 使用 `--suite act:wa01`。

### DP 示例

在已安装 Diffusion 依赖的 policy 环境中运行，将检查点路径替换为实际的 DP 模型目录：

```sh
membench-serve-lerobot \
  --checkpoint /path/to/dp-wa01/checkpoints/040000/pretrained_model \
  --device cuda \
  --port 8000
```

在终端 A 使用 `--suite diffusion:wa01`。服务端根据检查点的 `config.json` 自动选择 ACT 或 DP 后端。

两种 LeRobot 检查点目录都需要包含 `config.json`、模型权重、已保存的预处理及后处理配置和统计量，以及本基准的 `membench_policy.json` 元数据。该元数据提供仿真状态格式及所需的相机映射，服务端会在加载时检查接口约定。

### Pi0.5 示例

在终端 B 激活已安装兼容 OpenPI 实现的 policy 环境，将 `YOUR_OPENPI_CONFIG` 替换为该环境中已注册、且与 WA-01 检查点匹配的配置名称：

```sh
export ROBOCASA_MEM_ROOT=/absolute/path/to/RoboCasa-Mem
source "$ROBOCASA_MEM_ROOT/policy/openpi/.venv/bin/activate"
python -m pip install -e "$ROBOCASA_MEM_ROOT"
membench-serve-openpi \
  --config YOUR_OPENPI_CONFIG \
  --checkpoint /path/to/pi05-wa01/checkpoint \
  --image-size 224 \
  --seed-by-episode \
  --port 8000
```

在终端 A 使用 `--suite pi05:wa01`。模型包、任务专用的 OpenPI 配置和检查点均需单独准备，上面的 LeRobot 安装步骤不负责安装 OpenPI。

以上示例默认监听 `127.0.0.1`。如果模型服务位于另一台机器，请通过 `--host` 指定监听地址，并在评测端使用该机器的地址连接。更多选项可通过对应命令的 `--help` 查看。

自定义服务需要实现 `membench-policy/v2` 协议，并使用仿真运行配置：37 维状态、13 维动作、三路 224×224 RGB 图像（`agentview_left`、`agentview_right`、`eye_in_hand`），控制频率为 20 Hz。具体定义见[协议代码](bench/policy/protocol/contracts.py)和[仿真数据约定](bench/runtime/data_contract.py)。

## 运行评测（终端 A）

回到终端 A，激活项目运行环境 `.venv`。下面的示例连接前面启动的 ACT WA-01 服务，先检查评测配置与模型服务是否兼容：

```sh
cd /absolute/path/to/RoboCasa-Mem
source .venv/bin/activate
membench-eval \
  --endpoint ws://localhost:8000 \
  --suite act:wa01 \
  --preflight-only
```

预检查会验证任务、评测配置和远程策略接口，不会创建仿真任务实例，也不会验证资产恢复或渲染。随后运行完整的 50 个种子评测：

```sh
membench-eval \
  --endpoint ws://localhost:8000 \
  --suite act:wa01 \
  --output-root ./outputs/eval \
  --run-id wa01-act-round1
```

评测其他模型或任务时，在终端 B 加载对应检查点，在终端 A 选择匹配的评测配置，例如 `pi05:wx01` 或 `diffusion:ts03`，并使用不同的运行 ID。自定义实验也可以通过 `--suite` 直接指定 YAML 文件路径。

已记录的评测配置保留了各模型系列的执行设置：

| 模型系列      | 执行设置                                                                     |
| ------------- | ---------------------------------------------------------------------------- |
| `pi05`      | 每执行 40 个动作后重新规划（`r40`）                                        |
| `act`       | 使用各任务已记录的动作截断间隔                                               |
| `diffusion` | 每次请求一个动作（`r1`），保留服务端的 47 动作队列，并在每次请求时更新观测 |

WA-01 的 ACT、DP、Pi0.5 正式配置均显式使用 2000 步预算，任务默认上限也为 2000 步。历史 ACT、DP 的 WA-01 结果使用 5000 步，不能与新预算下的结果直接比较。其余八个任务保持原有的各模型预算及重规划间隔。

WX-01 初始化时，机器人已实际握住海绵。仓库提供覆盖全部 50 个种子的初始化动作记录，用于替代重置阶段的规划。初始化动作在模型控制前执行，不占用评测步数预算。记录缺失或仿真状态不匹配会报错，评测不需要 MPlib。

## 结果与断点续跑

结果处理命令在终端 A 执行。上述示例将结果写入 `outputs/eval/wa01-act-round1/`。其中，`run_manifest.json` 记录评测配置、种子、模型接口约定和运行环境标识；`summary.json` 汇总已写入的有效回合产物，包括任务成功情况和阶段结果。评测配置启用视频时，还会保存视频。

使用相同的运行 ID、评测配置和检查点重复执行原命令，即可断点续跑；该功能默认开启。添加 `--retry-errors` 可以重试发生错误的回合。更换模型或配置时，请使用新的运行 ID。

根据已有运行结果重建并校验汇总：

```sh
membench-summarize ./outputs/eval/wa01-act-round1
```

当结果满足完整性和产物校验要求时，评测及汇总命令返回退出码 `0`，否则返回 `2`。有效运行可以包含任务失败的回合，结果完整不代表每个回合都成功。

## 验证与发布状态

2026-10-25，使用独立的本地发布源码快照创建的全新 Python 环境已通过全部 450 项仿真检查，覆盖九任务各 50 个种子（1050–1099）的三步保持动作、三路相机渲染、阶段报告及结果产物。这是源码可编辑安装验收，未执行真实模型推理，也不替代 wheel 安装验收。2026-10-26，共享 policy 环境另已通过依赖一致性和接口导入检查，并使用已训练的 WA-01 检查点和合成观测完成 ACT、DP 的 CUDA 推理。随后 ACT 的 WA-01、seed 1050 试跑无运行错误地完成了 2000 步、40 次模型请求，保存了 2000 条动作、三路各 2000 帧的视频及完整阶段报告。本回合通过了加糖阶段，但未在预算内完成任务。这验证了单条执行流程，不代表完整基准性能；DP 的真实模型仿真回合仍待验证。

此前的 wheel 已在独立 Python 环境中完成安装验证，使用指定版本的仿真器源码，并恢复了全部 3035 个资产文件。验证覆盖全部 450 个正式任务与种子组合（九个任务，每个任务 50 个种子），每个组合执行初始化、渲染和三步保持动作。该验证使用同一台主机及 GPU 驱动，没有重新执行模型推理或完整的模型评测轨迹。

当前安装包移除了真机集成、调整了策略运行文件、整理了评测配置，并将 WA-01 预算统一为 2000 步。此前移除真机代码后的 wheel 已通过 47 项软件回归检查，包括本机 WebSocket 评测流程，以及 OpenPI 仿真接口和全部 27 个评测配置选择器的检查。本次整理后的版本已通过全部 27 个入口和 450 个回合输入身份的解析检查，wheel 与源码包也均确认只包含 27 份正式评测配置。当前安装包仍需重新执行 450 项 wheel 安装验收；最终源码版本冻结也尚未完成。具体状态见 [`release_candidate.json`](release_candidate.json)。

历史 TS03 Pi0.5 的种子 1056 曾将阶段报告保存在空 JSON 字段名下。经过验证的修复已恢复 `report_stages`，全部 50 份历史阶段报告均完整，总体成功率仍为 14/50（28%）。修正后的汇总和修复来源记录见[历史修复归档](https://github.com/SiHhangChen/RoboCasa-Mem/tree/e020312e8146c646c9013d35e9a7fc3523f0cdc1/docs/historical_result_repairs/ts03_pi05_r40)。

维护文档、验证材料、发布维护脚本、测试及真机代码均不包含在发布目录和分发包中。历史验证材料保存在[文档归档](https://github.com/SiHhangChen/RoboCasa-Mem/tree/e020312e8146c646c9013d35e9a7fc3523f0cdc1/docs)中。发布元数据中的 `docs/` 路径指向该归档版本或本地维护文件；验证材料校验和候选版本整理工具需要这些文件，正常评测不需要。测试源码保存在 Git 提交 `953b8ef` 中。

## 许可证

评测代码采用 [MIT 许可证](LICENSE)。RoboCasa 和 robosuite 是独立依赖，适用各自的 MIT 声明及第三方条款；MuJoCo 采用 Apache 2.0 许可证。仿真资产和模型检查点适用各自的条款。本项目许可证不授予这些材料的额外使用或分发权利，重新分发前请查阅其来源声明。
