# RoboCasa / RoboSuite 最小资产包

当前评测候选版复用 `membench` 已有的 **3035 文件**资产包，压缩大小为
270484609 字节，SHA-256 为
`140176b06cf220f6d11440fe3cede54cca7577edf13473d62fb210bd03ca68e3`。
完整清单见 [simulator_assets.json](../configs/benchmark/simulator_assets.json)，
恢复命令、校验工具和验收范围见[候选版资产说明](../release/evaluation/docs/ASSETS.md)。
下文保留旧收集器的使用说明和当时的 2748 文件统计，不能用作当前资产包的身份信息。

本仓库的任务不是使用 RoboCasa 的全部厨房资产。当前固定任务配置只用到以下
layout/style 对：

```text
(15,34)  (24,24)  (29,29)  (32,32)  (38,38)  (39,24)  (39,37)
```

任务代码还固定引用了 15 个物体目录：

```text
objects/objaverse/canned_food/canned_food_11
objects/objaverse/canned_food/canned_food_15
objects/objaverse/cup/cup_2
objects/objaverse/cup/cup_5
objects/objaverse/plate/plate_3
objects/objaverse/plate/plate_14
objects/objaverse/sponge/sponge_0
objects/objaverse/sponge/sponge_3
objects/objaverse/waffle/waffle_0
objects/aigen_objs/bar_soap/bar_soap_4
objects/aigen_objs/ketchup/ketchup_3
objects/lightwheel/mustard/Mustard003
objects/lightwheel/oven_tray/OvenTray001
objects/lightwheel/saucepan/Saucepan002
objects/lightwheel/sugar_cube/SugarCube003
```

这些目录之外，运行时还需要上述场景实际选中的 fixture model、fixture registry、
纹理、generative texture，以及 RoboSuite 的 Panda、Omron mobile base 和 Panda
gripper 资源。部分 RoboCasa fixture（例如 stool、dish rack、paper towel）本身
也由 registry 指向 `objects/lightwheel/` 下的模型；收集器会自动加入这些目录。
不要根据上面的 15 个目录手工复制，因为 fixture 的 mesh 和纹理依赖也必须保留。

## 生成清单或压缩包

在已经下载完整 RoboCasa 资产、并且仓库根目录包含 `robocasa/` 和 `robosuite/`
源码的机器上执行：

```bash
python scripts/tools/collect_sim_assets.py \
  --manifest /tmp/membench-sim-assets.json

python scripts/tools/collect_sim_assets.py \
  --tar /tmp/membench-sim-assets.tar.gz
```

传输时请同时带上 `.tar.gz` 和 `.json` 清单；后者用于在目标机器核对每个文件的
SHA-256。

脚本不创建仿真环境，不需要显示器；只读取任务 YAML、RoboCasa 的 layout/style
和 fixture registry，并为每个文件写入大小与 SHA-256。当前仓库的实际结果约为
`0.489 GiB`（当前 gzip 压缩包约 225 MiB），明显小于 RoboCasa 全量 `objects/` 下载。

收集器也会扫描 `bench/tasks/*/resources/assets/*.xml` 的 `file`/`meshfile`
引用；因此 WR03、WR01、TM02 自定义柜子使用的 `wood_planks_2.png` 和
`metals/metal.png` 会被纳入。RoboCasa 墙体 backing 默认使用的
`textures/flat/light_gray.png` 也会显式纳入，避免只在部分机器上依赖兼容链接。

旧版 2745 文件清单遗漏了上述三张纹理，不应继续用于场景或视觉策略复现。当前
manifest schema 为 2，共 2748 个文件。

如果不想生成压缩包，也可以复制成一个保持仓库相对路径的目录：

```bash
python scripts/tools/collect_sim_assets.py \
  --copy-to /mnt/transfer/membench-sim-assets
```

## 在另一台机器恢复

目标机器仍需先按 [README.md](../README.md) 克隆本仓库，以及固定版本的 RoboCasa
和 RoboSuite 源码。脚本会把生成机器实际使用的两个 Git commit 写入 manifest；
恢复后应优先核对这两个 commit。如果当前机器的 `robocasa/` 或 `robosuite/`
包含仓库外的本地修改/私有 commit，仅复制资产还不够，还需要另外同步对应源码
（或先把这些修改提交到目标机器可访问的 Git 仓库）。然后在目标仓库根目录解压：

```bash
tar -xzf /mnt/transfer/membench-sim-assets.tar.gz -C /path/to/membench
```

这会把文件放回 `robocasa/robocasa/models/assets/` 和
`robosuite/robosuite/models/assets/`。恢复后可用 manifest 校验：

```bash
python - <<'PY'
import hashlib, json
from pathlib import Path

manifest = json.loads(Path('/tmp/membench-sim-assets.json').read_text())
root = Path('.')
bad = []
for item in manifest['files']:
    path = root / item['path']
    if not path.is_file():
        bad.append((item['path'], 'missing'))
        continue
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != item['sha256']:
        bad.append((item['path'], 'sha256'))
print('OK' if not bad else f'{len(bad)} files failed')
if bad:
    print(*bad[:20], sep='\n')
PY
```

资产包只覆盖外部 RoboCasa/RoboSuite 资源；任务专用 XML（例如 TM02 的
`resources/assets/`）和其它配置仍由本仓库提供。若目标机器的任务配置、第三方
commit 或资产内容发生变化，应重新运行收集器，不要继续复用旧 manifest。
