# 独立评测候选版实施结果

当前仓库根目录即为评测候选版，版本 `0.1.0rc1`。它只注册 WX-01、WA-01、WA-05、
WR-01、WR-08、TS02、TS03、TM01、TM02，顺序与发布任务清单一致。

## 已完成

- 27 份实际使用的 suite 原样保留，逐文件 SHA256 与规范化 hash 均核验。
  pi05、ACT、Diffusion 分别保留已有执行设置；DP 的 wire r1 与服务端原生动作缓存保持原样。
- 当前任务输入共 450 个（9 个任务 × 50 个 seed）逐项对比。物理配置、随机化事实、
  环境参数、prompt、观测及动作配置一致。移除 planner_script 后产生的新身份 hash
  及原身份映射保存在候选版 `configs/benchmark/evaluation_input_lock.json` 中。
  TS02 固定为当前 1.1.0 和各实测 suite 的 2800 步。
- 开发版专家 handler 延迟加载，保留原录制入口；候选版不包含专家策略、录制流程、
  训练脚本或 MPlib，只保留初始化和判定需要的公共工具。
- WX-01 的 50 个 seed（1050–1099）全部生成初始化记录。每份包含初始化动作、显式
  观测更新事件、终态 qpos/qvel/time、初始观测 hash、任务与资源身份、环境版本和验证结果。
  初始化步骤范围为 249–265，发生在模型接管之前，不占用 suite 的模型评测预算。
- 每个 WX-01 seed 均与当次原实现初始化比较，并执行相同的 20 步检查动作；初始观测、
  state37、轨迹图像、诊断与阶段判定一致。候选版按 seed 重放，缺失记录、资源变更或终态
  不匹配会明确报错。记录清单为 `configs/benchmark/setup_replays/wx01/manifest.json`。
- 9 个任务均在独立进程中通过 seed 1050 的真实仿真对比：初始化与随后相同的 20 步动作
  保持一致；候选进程禁止导入专家、录制和 MPlib。证据摘要见
  `docs/simulation_equivalence.json`。
- 开发源码定向测试 102 项通过；候选版软件测试 109 项通过，另行运行的 localhost
  WebSocket 集成测试 1 项通过。wheel 构建、仓库外安装、
  任务发现、suite family 解析和全部 27 个配置加载通过；安装后仍只发现指定的 9 个任务。
- 已复用 `membench` 现有资产包，记录 3035 个文件的相对路径、大小及 SHA-256，
  清单为 `configs/benchmark/simulator_assets.json`。当前包压缩大小 270484609 字节，
  SHA-256 为 `140176b06cf220f6d11440fe3cede54cca7577edf13473d62fb210bd03ca68e3`。
  全部文件与本机已安装资产逐项一致；校验工具为 `scripts/release/verify_assets.py`。
  既有环境恢复及 WA-01 冒烟测试证据已保存，当前包与历史旧包分别记录。

## 交接入口

```sh
# 从仓库根目录检查验收证据和发布文件哈希。
python scripts/release/validate_installation_evidence.py .

# 构建可安装的 wheel。
python -m pip wheel --no-deps --no-build-isolation . -w dist
```

原开发树及其导出、源码对比、初始化记录生成脚本保存在 Git 检查点
`a10358f`。当前根目录直接维护发布内容，不再需要嵌套导出步骤。WX-01 已附带
50 份初始化记录；如需改变初始化或任务资源，应重新生成并验收相关证据。
维护发布元数据可使用 `python scripts/release/finalize_candidate.py .`；该工具会校验
现有验收证据，不会把修改后的运行代码自动视为已验收，也不会批准公开发布。

维护测试源码已移出当前发布范围，最后一版保存在 Git 提交 `953b8ef`。
本文中的测试通过数量及 `docs/installation_evidence/` 中的日志属于既有验收记录；
运行任务和模型评测不需要测试源码或 pytest。

候选版安装后可从任意目录使用：

```sh
membench-eval --endpoint ws://localhost:8000 --suite pi05:wx01
membench-eval --endpoint ws://localhost:8000 --suite act:wa01
membench-eval --endpoint ws://localhost:8000 --suite diffusion:wa01
```

每次新评测输出记录候选版本、实现内容 hash、配置/初始化清单身份和模拟器包版本。
完整安装说明见候选版 `docs/INSTALL.md`。

## 当前验证范围与剩余发布工作

上述验证比较当前源码与拆分后实现，不是重新执行全部历史模型实验；历史运行没有
提供全部初始仿真状态，因此不能声称与历史实验逐像素核验。实际 suite 设置已与历史
运行记录核验一致。短检查动作也不等同于完整任务成功轨迹的验收。
2026-10-20 已在新建 Python 3.11.15 环境中安装 wheel，独立恢复固定版本模拟器源码
及当前资产包。9 个任务 × seed 1050–1099，共 450/450 组均通过真实初始化、三路
224×224 RGB 渲染、state37/action13、三步保持动作和阶段报告检查；没有访问旧目录
或开发目录中的运行资源，也未导入专家、录制或 MPlib。软件测试 97 项及单独的
WebSocket 测试 1 项通过，27 个 suite family selector 均验证通过。
这是同一主机上的独立环境，共用基础 Python 与 GPU 驱动；没有重跑模型推理或完整
任务成功轨迹。证据位于 `docs/installation_evidence/`，步骤见候选版
`docs/ISOLATED_INSTALL.md`。运行文件与实际验收 wheel 逐字节一致；更改运行文件
会使 finalizer 拒绝沿用本次验收。

第三方模拟器和资产压缩包没有打入候选 Python 包，候选目录包含资产清单、校验工具、
安装说明和验证证据。现有资产包的 3035 文件已在独立环境恢复校验，9 个候选任务
的全部 450 组正式输入均完成短步验收。过去的恢复测试
使用 2835 文件旧包，不能将它的文件数和哈希用于当前包。详见候选版 `docs/ASSETS.md`。
历史 TS03 pi05 seed 1056 的完整阶段报告实际存于 JSON 空字段名，
已通过字段重命名恢复；50 份报告均与原始诊断交叉核验，重新汇总后覆盖 50/50、
`publishable=true`，总成功率保持 14/50（28%）。这次处理没有重跑模型或仿真。
原始文件不变；修复副本、原始结果、原始汇总与哈希证据位于
`docs/historical_result_repairs/ts03_pi05_r40/`，发布结果应使用其中的 `summary.json`。
环境与资产技术验收已通过。候选版仍保留 `release_ready=false`：尚需冻结最终发布
源码版本并向使用者提供固定资产包；尚未发布到外部仓库或包平台。
