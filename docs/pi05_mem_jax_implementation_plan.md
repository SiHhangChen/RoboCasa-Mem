# π0.5-MEM 原生 JAX 完整短期记忆实施计划

## 1. 文档目的

本文档定义如何在 `policy/openpi-mem` 中实现原生 JAX 版本的 π0.5-MEM，并将其接入 MemBench 的训练、WebSocket 推理和闭环评测链路。

目标不是复现论文的长期语言记忆，而是一次性实现论文中的完整短期 observation memory：

```text
多帧 RGB history
    ↓
Space-Time SigLIP
    ↓
当前帧视觉 tokens

历史 proprio states
    ↓
共享 Linear Projection
    ↓
T 个 continuous state tokens

视觉 tokens + prompt tokens + state-history tokens
    ↓
π0.5 PaliGemma + Action Expert
    ↓
action chunk
```

第一阶段不实现：

- 长期语言 memory；
- high-level policy；
- language memory update；
- No-History、Pool-Memory、Proprio-only 等消融训练；
- OpenTau 的 motion module。

消融以后从同一套实现中通过配置关闭视觉历史或状态历史生成，不需要预先开发多套模型。

## 2. 实现定位与参考来源

本实现应命名为：

```text
MEM-style π0.5
或 π0.5-MEM
```

不能声称复现论文中的 π0.6-MEM 数值结果，因为当前基础模型和论文存在以下差异：

| 项目 | MEM 论文 | 本实现 |
|---|---|---|
| VLA | π0.6 | π0.5 |
| 图像分辨率 | 448×448 | 224×224 |
| 相机数 | 最多四路 | 两路有效相机 |
| 预训练数据 | 大规模机器人、video-language 混合数据 | MemBench 专家轨迹为主 |
| 长期语言 memory | 有 | 第一阶段不实现 |

权威程度从高到低的参考为：

1. MEM 论文正文与附录：决定架构必须满足的性质。
2. OpenTau 固定 commit `5997783a93f48455225e6d532609371e650d9016`：作为论文公式的可执行解释和测试参考。
3. 当前 `policy/openpi-mem`：作为实际 JAX/Flax、checkpoint、LoRA 和训练基础。

OpenTau 不是 MEM 官方实现。特别是 temporal sinusoidal encoding 的具体频率范围，以及论文附录中嵌套 attention 公式的程序解释，属于 OpenTau 的实现选择，必须在代码和实验记录中固定并说明。

## 3. 第一版固定规格

```yaml
model_type: pi05_mem
base_checkpoint: pi05_base

n_obs_steps: 6
history_stride_steps: 20       # 20 Hz 下等于 1 秒
spacetime_layer_stride: 4
temporal_min_period: 0.004
temporal_max_period: 40.0

use_video_history: true
use_state_history: true
use_motion: false

cameras:
  - agentview_right
  - eye_in_hand
image_resolution: [224, 224]

state_raw_dim: 37
state_keep_dim: 30
state_model_dim: 32

action_raw_dim: 13
action_model_dim: 32
action_horizon: 50
```

时间序列约定：

```text
index:        0    1    2    3    4    5
relative s:  -5   -4   -3   -2   -1    0
role:       past past past past past current
```

当前帧始终位于最后一个时间位置。

## 4. 计划中的文件变更

### 4.1 `policy/openpi-mem` 新增文件

```text
src/openpi/models/siglip_mem.py
src/openpi/models/siglip_mem_test.py
src/openpi/models/pi0_mem_test.py

src/openpi/policies/membench_policy.py
src/openpi/policies/membench_policy_test.py

src/openpi/training/membench_v3_dataset.py
src/openpi/training/membench_v3_dataset_test.py
src/openpi/training/weight_loaders_test.py

docs/pi05_mem.md
```

### 4.2 `policy/openpi-mem` 修改文件

```text
src/openpi/models/model.py
src/openpi/models/pi0.py
src/openpi/models/pi0_config.py

src/openpi/training/config.py
src/openpi/training/data_loader.py
src/openpi/training/weight_loaders.py

scripts/compute_norm_stats.py
```

### 4.3 MemBench 主仓库新增文件

```text
bench/policy/backends/openpi_mem.py
scripts/serve/serve_openpi_mem.py

tests/serving/test_openpi_mem_backend.py

configs/eval/suites/wa01_smoke_pi05_mem_v1.yaml
configs/eval/suites/wr03_smoke_pi05_mem_v1.yaml

scripts/eval/eval_pi05_mem.sh
scripts/train/train_pi05_mem.sh
```

### 4.4 MemBench 主仓库保持不变的文件

```text
bench/policy/backends/openpi.py
bench/evaluation/rollout.py
bench/evaluation/client.py
scripts/serve/serve_openpi.py
```

第一版不给现有 OpenPI baseline 增加任何 MEM 分支。MEM 使用独立 backend 和独立 server，历史缓存在 MEM backend 内。通用 rollout、client、协议和 vanilla server 均保持原样。

## 5. JAX Space-Time SigLIP

### 5.1 新增 `src/openpi/models/siglip_mem.py`

该文件实现完整 JAX Space-Time SigLIP。现有 `siglip.py` 保持为 vanilla 实现，`siglip_mem.py` 复用其中的以下组件：

```python
MlpBlock
MAPHead
get_posemb
decode_variant
```

这样可以避免改变无记忆 π0.5 的行为。

### 5.2 `temporal_sincos_position_embedding`

建议签名：

```python
def temporal_sincos_position_embedding(
    num_frames: int,
    embedding_dim: int,
    *,
    min_period: float = 4e-3,
    max_period: float = 40.0,
    dtype: jnp.dtype = jnp.float32,
) -> jax.Array:
    ...
```

输入输出：

```text
input:  num_frames=T, embedding_dim=D
output: float[T,D]
```

时间坐标：

```text
[-T+1, ..., -2, -1, 0]
```

必须满足：

```python
pe[-1] == 0
```

实现方式：

```python
pe = concatenate(sin(phase), cos(phase))
pe = pe - pe[-1:]
```

作用：

- 为历史帧提供相对时间位置；
- 保证当前帧 `e(0)=0`；
- 保证 T=1 初始化时与原 SigLIP 一致；
- 不引入可学习参数。

### 5.3 `make_temporal_attention_mask`

建议签名：

```python
def make_temporal_attention_mask(
    history_is_pad: jax.Array,
) -> jax.Array:
    ...
```

输入：

```text
history_is_pad: bool[B,T]
True 表示该时间位置为 padding
```

输出：

```text
bool[B,1,1,T,T]
```

mask 规则：

```text
query t_i 只能关注 key t_j，其中：
j <= i
且 history_is_pad[:,j] == False
```

当前帧必须强制有效：

```python
key_valid = ~history_is_pad
key_valid = key_valid.at[:, -1].set(True)
```

该 mask 同时解决：

- temporal causality；
- episode 开头 padding；
- zero image 经过带 bias 的 patch embedding 后产生非零特征的问题；
- padding temporal PE 非零导致的历史泄漏。

### 5.4 `space_time_dot_product_attention`

建议签名：

```python
def space_time_dot_product_attention(
    query,
    key,
    value,
    *,
    num_frames: int,
    history_is_pad,
    layer_index,
    spacetime_layer_stride: int,
    mask=None,
    dropout_rng=None,
    dropout_rate=0.0,
    broadcast_dropout=True,
    deterministic=True,
    dtype=None,
    precision=None,
    force_fp32_for_softmax=False,
    **kwargs,
):
    ...
```

Flax `MultiHeadDotProductAttention` 在调用该函数前已经完成 Q/K/V 投影，因此输入形状为：

```text
[B*T,N,H,Dh]
```

普通层直接执行：

```python
nn.dot_product_attention(query, key, value)
```

ST layer 执行：

```text
[B*T,N,H,Dh]
→ [B,N,T,H,Dh]
→ temporal causal attention over T
→ [B,T,N,H,Dh]
→ spatial bidirectional attention over N
→ [B*T,N,H,Dh]
```

核心计算：

```python
v_temporal = temporal_attention(
    q_temporal,
    k_temporal,
    v_temporal,
    mask=temporal_mask,
)

output = spatial_attention(
    q_spatial,
    k_spatial,
    v_temporal,
)
```

必须保持以下约束：

- Q/K/V 各投影一次；
- temporal 和 spatial pass 复用同一组 Q/K；
- temporal 输出作为 spatial pass 的 V；
- output projection 只执行一次；
- residual 只执行一次；
- 不新增视觉可学习参数。

这是 OpenTau 对论文附录公式的 Reading B 解释。

### 5.5 `SpaceTimeEncoder1DBlock`

建议接口：

```python
class SpaceTimeEncoder1DBlock(nn.Module):
    mlp_dim: int | None
    num_heads: int
    num_frames: int
    spacetime_layer_stride: int
    temporal_min_period: float
    temporal_max_period: float
    dropout: float
    dtype_mm: str

    @nn.compact
    def __call__(
        self,
        x,
        layer_index,
        history_is_pad,
        deterministic=True,
    ):
        ...
```

执行顺序：

```text
residual = x

如果是 ST layer：
    attention_input = x + temporal_pe
否则：
    attention_input = x

y = LayerNorm_0(attention_input)
y = MultiHeadDotProductAttention_0(
        attention_fn=space_time_dot_product_attention
    )(y, y)

x = residual + y
x = x + MlpBlock_0(LayerNorm_1(x))
```

temporal PE 只参与 attention 输入，不进入 residual。

参数名字必须继续保持：

```text
LayerNorm_0
MultiHeadDotProductAttention_0/query
MultiHeadDotProductAttention_0/key
MultiHeadDotProductAttention_0/value
MultiHeadDotProductAttention_0/out
LayerNorm_1
MlpBlock_0
```

这是 `pi05_base` 视觉 checkpoint 原样加载的基础。

### 5.6 `SpaceTimeEncoder`

该类对应现有 `siglip.Encoder`，继续使用：

```python
nn.scan(
    variable_axes={"params": 0},
    ...,
)
```

新增被 scan 的 layer index：

```python
layer_indices = jnp.arange(self.depth)
```

启用 ST 的 0-based layer index：

```text
[3, 7, 11, 15, 19, 23]
```

即 1-based 的：

```text
4、8、12、16、20、24 层
```

不能为实现方便把 27 层改成独立 named blocks，因为当前 checkpoint 的视觉参数沿 scan axis 堆叠：

```text
Transformer/encoderblock/... shape=[27,...]
```

### 5.7 `SpaceTimeVisionTransformer`

建议接口：

```python
class SpaceTimeVisionTransformer(nn.Module):
    ...

    @nn.compact
    def __call__(
        self,
        video,
        history_is_pad,
        *,
        train=False,
    ):
        ...
```

输入：

```text
video: float[B,T,H,W,3]
history_is_pad: bool[B,T]
```

处理：

```text
[B,T,H,W,3]
→ [B*T,H,W,3]
→ patch embedding
→ 2D spatial position embedding
→ SpaceTimeEncoder
→ encoder norm
→ [B,T,N,D]
→ 只保留 [:,-1]
→ projection head
```

输出：

```text
[B,N,2048]
```

过去帧视觉 tokens 不进入 PaliGemma backbone。

### 5.8 `Module` factory

保持与 vanilla SigLIP 相同的使用方式：

```python
def Module(num_classes=None, *, variant=None, **kwargs):
    return SpaceTimeVisionTransformer(...)
```

## 6. Observation 数据结构

### 6.1 修改 `src/openpi/models/model.py`

#### `ModelType`

新增：

```python
PI05_MEM = "pi05_mem"
```

#### `Observation`

新增字段：

```python
history_is_pad: at.Bool[ArrayT, "*b t"] | None = None
```

vanilla 模型：

```text
images[name]: [B,H,W,C]
image_masks[name]: [B]
state: [B,D]
history_is_pad: None
```

PI05_MEM：

```text
images[name]: [B,T,H,W,C]
image_masks[name]: [B,T]
state: [B,T,D]
history_is_pad: [B,T]
```

直接让 `images` 和 `state` 携带时间维，可以复用当前 Normalize 和 PadStatesAndActions 的最后一维语义。

#### `Observation.from_dict`

增加：

```python
history_is_pad=data.get("history_is_pad")
```

NumPy uint8 图像转换应支持 `[B,T,H,W,C]`。

PyTorch 分支要区分：

```text
[B,H,W,C]   → [B,C,H,W]
[B,T,H,W,C] → [B,T,C,H,W]
```

#### `Observation.to_dict`

将 `history_is_pad` 正常写回字典。

#### `preprocess_observation`

增加：

```python
is_memory = observation.history_is_pad is not None
```

vanilla 分支保持原逻辑。

memory 分支检查：

```text
image.ndim == 5
state.ndim == 3
history_is_pad.shape == state.shape[:2]
image.shape[:2] == state.shape[:2]
image_masks[name].shape == state.shape[:2]
history_is_pad[:,-1] == False
```

图像尺寸检查使用：

```python
image.shape[-3:-1]
```

resize 时：

```text
[B,T,H,W,C] → [B*T,H,W,C] → resize → [B,T,H',W',C]
```

建议新增私有函数：

```python
def _augment_image_batch(...): ...
def _augment_video_batch(...): ...
```

`_augment_video_batch` 对每个 batch sample 只生成一组几何增强参数，并将其应用到同一 clip 的全部 T 帧，防止随机 crop/rotation 制造虚假运动。

## 7. π0.5-MEM 配置

### 7.1 修改 `src/openpi/models/pi0_config.py`

在 `Pi0Config` 增加：

```python
short_term_memory: bool = False
n_obs_steps: int = 6
spacetime_layer_stride: int = 4
temporal_min_period: float = 4e-3
temporal_max_period: float = 40.0

image_keys: tuple[str, ...] = (
    "base_0_rgb",
    "left_wrist_0_rgb",
    "right_wrist_0_rgb",
)
```

MEM 配置改为：

```python
image_keys=("base_0_rgb", "left_wrist_0_rgb")
```

避免为 dummy right-wrist camera 编码 6 帧。

#### `__post_init__`

新增验证：

```python
if short_term_memory and not pi05:
    raise ValueError("short_term_memory requires pi05=True")

if short_term_memory and discrete_state_input:
    raise ValueError("PI05_MEM uses continuous state-history tokens")

if n_obs_steps < 1:
    raise ValueError(...)

if spacetime_layer_stride < 1:
    raise ValueError(...)
```

当 `short_term_memory=True` 时自动设置：

```python
discrete_state_input=False
```

#### `model_type`

返回：

```python
ModelType.PI05_MEM
```

#### `inputs_spec`

MEM 规格：

```text
image: [B,T,224,224,3]
image_mask: [B,T]
state: [B,T,32]
history_is_pad: [B,T]
```

vanilla 规格不变。

#### `get_freeze_filter`

第一版训练策略：

- Gemma 非 LoRA 参数冻结；
- Gemma LoRA 可训练；
- action expert LoRA 可训练；
- 整个 SigLIP 视觉塔可训练；
- visual projection head 可训练；
- state-history projector 可训练；
- action input/output/time projections 可训练。

当前路径过滤默认不会冻结 `PaliGemma.img`。第一版先微调整个视觉塔，避免在 `nn.scan` 堆叠参数上引入 element-level optimizer mask。ST-only 视觉更新可作为后续资源优化。

## 8. π0.5 模型接入

### 8.1 修改 `src/openpi/models/pi0.py`

#### `Pi0.__init__`

保存：

```python
self.short_term_memory = config.short_term_memory
self.image_keys = config.image_keys
```

视觉模型分支：

```python
if config.short_term_memory:
    img = siglip_mem.Module(...)
else:
    img = siglip.Module(...)
```

MEM lazy init 输入：

```python
img.lazy_init(
    fake_video,
    fake_history_is_pad,
    train=False,
    rngs=rngs,
)
```

新增：

```python
self.state_history_proj = nnx.Linear(
    config.action_dim,
    paligemma_config.width,
    rngs=rngs,
)
```

即：

```text
32 → 2048
```

state history 必须进入 VLM prefix，因此不能投影到 action expert 的 1024 维。

#### `embed_prefix`

修改签名：

```python
def embed_prefix(
    self,
    obs: Observation,
    *,
    train: bool = False,
):
    ...
```

vanilla 路径保持原行为。

MEM 视觉路径：

```python
camera_pad = obs.history_is_pad | ~obs.image_masks[name]

image_tokens, _ = self.PaliGemma.img(
    obs.images[name],
    camera_pad,
    train=train,
)
```

视觉 prefix mask 使用当前帧相机有效性：

```python
current_camera_valid = ~camera_pad[:, -1]
```

语言 tokens 保持现有 embedding，但 prompt 中不再包含离散 state。

state-history tokens：

```python
state_mask = ~obs.history_is_pad
state_mask = state_mask.at[:, -1].set(True)

masked_state = jnp.where(
    state_mask[..., None],
    obs.state,
    0.0,
)

state_tokens = self.state_history_proj(masked_state)
```

必须在 Normalize 后、projection 前清零 padding state，避免 raw zero 归一化成 `-mean/std` 后泄漏。

state tokens 加入 prefix：

```python
tokens.append(state_tokens)
input_mask.append(state_mask)
ar_mask += [False] * T
```

最终 prefix 顺序：

```text
base video tokens
wrist video tokens
language tokens
state-history tokens
```

#### `compute_loss`

调用：

```python
prefix_tokens = self.embed_prefix(
    observation,
    train=train,
)
```

确保 action loss 可以反向传播到 ST SigLIP。

#### `sample_actions`

调用：

```python
self.embed_prefix(observation, train=False)
```

batch size 仍取：

```python
observation.state.shape[0]
```

## 9. checkpoint 热启动

### 9.1 修改 `src/openpi/training/weight_loaders.py`

视频编码器不应产生新视觉参数。`pi05_base` 中全部 `PaliGemma/img/...` 权重必须原样加载。

新增的必需参数只有：

```text
state_history_proj/kernel
state_history_proj/bias
```

LoRA 参数也可能不在 base checkpoint 中。

### 9.2 `Pi05MemCheckpointWeightLoader`

```python
@dataclasses.dataclass(frozen=True)
class Pi05MemCheckpointWeightLoader:
    params_path: str

    def load(self, params):
        ...
```

允许保留初始化值的 missing 参数只匹配：

```text
.*lora.*
state_history_proj/.*
```

其他 missing key 一律报错。

### 9.3 `_merge_params_strict`

```python
def _merge_params_strict(
    loaded_params,
    reference_params,
    *,
    allowed_missing_patterns: tuple[str, ...],
):
    ...
```

功能：

- 检查 loaded key 是否存在于目标模型；
- 检查 shape；
- 转换 dtype；
- 只补充白名单参数；
- 报告 missing、unexpected 和 shape mismatch。

不能使用过宽的 `missing_regex=".*"`，否则视觉参数树不兼容时可能被静默忽略。

## 10. MemBench 时序数据集

### 10.1 新增 `src/openpi/training/membench_v3_dataset.py`

从现有 `policy/openpi` 的单帧 MemBench v3 reader 移植，并增加历史读取。

#### `MemBenchV3Dataset.__init__`

```python
class MemBenchV3Dataset:
    def __init__(
        self,
        repo_id: str,
        *,
        action_horizon: int,
        n_obs_steps: int,
        history_stride_steps: int,
        root=None,
        tolerance_s=1e-4,
        video_backend=None,
    ):
        ...
```

#### `_history_indices`

```python
def _history_indices(
    self,
    current_index: int,
    episode_index: int,
) -> tuple[np.ndarray, np.ndarray]:
    ...
```

返回：

```text
indices: int64[T]
is_pad: bool[T]
```

20 Hz、T=6 的目标 offsets：

```text
[-100, -80, -60, -40, -20, 0]
```

绝不能越过 episode boundary。

episode 开头：

- invalid index 使用 episode 首帧索引占位；
- `is_pad=True`；
- 返回 item 前将 padded state/image 显式清零。

#### `_get_video_clip`

```python
def _get_video_clip(
    self,
    key: str,
    episode_index: int,
    timestamps: np.ndarray,
    is_pad: np.ndarray,
) -> np.ndarray:
    ...
```

输出：

```text
uint8[T,H,W,3]
```

尽量一次调用 `decode_video_frames` 解码多个 timestamps，避免每个历史帧单独启动一次解码。

#### `__getitem__`

输出：

```python
{
    "observation.state": state_history,             # [T,37]
    "observation.history_is_pad": history_is_pad,   # [T]
    "observation.images....": video_clip,           # [T,H,W,3]
    "action": actions,                              # [50,13]
    "timestamp": current_timestamp,
    "episode_index": ...,
    "task_index": ...,
    "index": ...,
}
```

action chunk 必须从 current 开始，而不是从最早历史帧开始。

## 11. MemBench transforms

### 11.1 新增 `src/openpi/policies/membench_policy.py`

#### `MemBenchMemoryInputs`

输入：

```text
images.agentview_right: [T,H,W,3]
images.eye_in_hand: [T,H,W,3]
state: [T,37]
history_is_pad: [T]
actions: [50,13]
prompt: str
```

输出：

```python
{
    "image": {
        "base_0_rgb": ...,
        "left_wrist_0_rgb": ...,
    },
    "image_mask": {
        "base_0_rgb": ~history_is_pad,
        "left_wrist_0_rgb": ~history_is_pad,
    },
    "state": ...,
    "history_is_pad": ...,
    "actions": ...,
    "prompt": ...,
}
```

#### `SliceState`

沿最后一维执行：

```python
state[..., :30]
```

同时支持 `[D]`、`[T,D]` 和 `[B,T,D]`。

#### `MemBenchOutputs`

```text
[50,32] → [50,13]
```

## 12. 数据配置与训练配置

### 12.1 修改 `src/openpi/training/config.py`

#### `ModelTransformFactory.__call__`

新增：

```python
case ModelType.PI05_MEM:
```

transform 顺序：

```text
InjectDefaultPrompt
ResizeImages(224,224)
TokenizePrompt(discrete_state_input=False)
PadStatesAndActions(32)
```

客户端 `ResizeImages` 已支持任意 leading dimensions，因此 `[T,H,W,C]` 可直接 resize。

#### `LeRobotMemBenchMemoryDataConfig`

字段：

```python
repo_id
default_prompt
state_keep_dim = 30
n_obs_steps = 6
history_stride_steps = 20
```

repack：

```python
{
    "images": {
        "agentview_right": "...robot0_agentview_right",
        "eye_in_hand": "...robot0_eye_in_hand",
    },
    "state": "observation.state",
    "history_is_pad": "observation.history_is_pad",
    "actions": "action",
    "prompt": "prompt",
}
```

#### 训练配置

至少增加：

```text
pi05_mem_membench_wa01_lora
pi05_mem_membench_wr03_lora
```

模型配置：

```python
Pi0Config(
    pi05=True,
    short_term_memory=True,
    discrete_state_input=False,
    n_obs_steps=6,
    spacetime_layer_stride=4,
    image_keys=("base_0_rgb", "left_wrist_0_rgb"),
    paligemma_variant="gemma_2b_lora",
    action_expert_variant="gemma_300m_lora",
)
```

weight loader：

```python
Pi05MemCheckpointWeightLoader(pi05_base)
```

初始训练参数建议：

```yaml
batch_size: 2 或 4
peak_lr: 2.5e-5
num_train_steps: 60000
ema_decay: null
num_workers: 8
fsdp_devices: 根据 GPU 数量配置
```

第一版使用统一学习率，先验证训练稳定。多学习率不作为首个实现的阻塞项。

## 13. DataLoader

### 13.1 修改 `src/openpi/training/data_loader.py`

在 `create_torch_dataset` 中增加：

```python
if membench_v3_dataset.is_membench_v3_dataset(repo_id):
    dataset = MemBenchV3Dataset(
        repo_id,
        action_horizon=action_horizon,
        n_obs_steps=model_config.n_obs_steps,
        history_stride_steps=data_config.history_stride_steps,
    )
```

避免普通 LeRobot loader 尝试读取 MemBench v3 metadata。

## 14. normalization statistics

### 14.1 修改 `scripts/compute_norm_stats.py`

不能把 episode 开头的 padded state history zeros 加入 state statistics。

当 batch state 为 `[B,T,D]` 时，只使用当前 state：

```python
state_for_stats = batch["state"][:, -1]
```

actions 保持完整：

```python
batch["actions"]
```

最终 state mean/std 基于真实当前帧分布，Normalize 时广播到全部历史 state。

## 15. 推理 history 的归属

第一版将 observation history 放在独立 MEM backend 中维护，而不是修改通用 rollout、client 或 WS 协议：

```text
rollout 按 replan_steps=20 调用 MEM policy
→ OpenPIMemBackend 收到当前图像和当前 state
→ backend 将当前 observation 加入 episode-local ring buffer
→ backend 组装 5 个历史 observation + 当前 observation
→ JAX π0.5-MEM 接收显式 [T,...] 输入
```

这种设计中：

- 核心 JAX model 仍然是无状态函数；
- MEM backend 是 episode-local 有状态组件；
- vanilla OpenPI backend 完全不变；
- `membench-policy/v2` 线协议完全不变；
- 通用 rollout 和 client 完全不变；
- MEM suite 必须使用 `replan_steps=20`，使相邻 policy observation 的间隔严格为 1 秒。

如果未来需要在 `replan_steps!=20` 时仍使用 1 秒历史，再单独设计逐环境步 observation streaming 协议；该扩展不属于第一版。

## 16. 独立 OpenPI MEM backend

### 16.1 新增 `bench/policy/backends/openpi_mem.py`

该文件不得在现有 `openpi.py` 中添加条件分支。它实现独立的：

```python
class OpenPIMemBackend:
    ...
```

该 backend 只复用主仓库已有的通用协议、checkpoint identity、action validation 和 data-contract 类型，不继承 `OpenPIBackend`，也不从 `bench.policy.backends.openpi` 导入实现细节。文件内应独立提供：

```python
_DEFAULT_MEM_CAMERAS = ("agentview_right", "eye_in_hand")

def _mem_policy_cameras_from_config(train_config: Any) -> tuple[str, ...]: ...
def _validate_mem_train_config(train_config: Any) -> None: ...
```

`_mem_policy_cameras_from_config()` 从 MEM data repack 的 `images` mapping 推导实际时序相机；`_validate_mem_train_config()` 集中验证 `PI05_MEM`、`short_term_memory=True`、`n_obs_steps=6`、`history_stride_steps=20` 以及 action horizon，避免这些检查散落在 server 脚本中。

### 16.2 `_ObservationHistoryBuffer`

建议在同一文件中新增私有类：

```python
class _ObservationHistoryBuffer:
    def __init__(
        self,
        *,
        n_obs_steps: int,
        history_stride_steps: int,
        cameras: tuple[str, ...],
        image_size: tuple[int, int],
        state_dim: int,
    ): ...

    def reset(self) -> None: ...

    def append(
        self,
        *,
        step: int,
        images: Mapping[str, np.ndarray],
        state: np.ndarray,
    ) -> None: ...

    def stack(self) -> tuple[
        dict[str, np.ndarray],
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ]: ...
```

内部使用最大长度为 `n_obs_steps` 的 deque，每次 policy call 保存一个 observation。由于 MEM suite 固定 `replan_steps=20`，这些 observation 对应：

```text
t-100, t-80, t-60, t-40, t-20, t
```

`append()` 必须验证：

- 第一次请求必须是 step 0；
- 后续 step 严格单调递增；
- `step - previous_step == history_stride_steps`；
- 图像 shape/dtype 满足当前 canonical contract；
- state 为 `float32[37]`；
- 保存数组副本，避免 simulator observation buffer 后续原地变化。

`stack()` 输出：

```text
images.<camera>: uint8[T,224,224,3]
state: float32[T,37]
history_is_pad: bool[T]
history_steps: int64[T]
```

episode 开头不足 T 个 observation 时向左 padding：

- image 全零；
- state 全零；
- `history_is_pad=True`；
- `history_steps` 使用对应的负相对 step，便于日志审计；
- 当前帧始终位于最后且 `history_is_pad[-1]=False`。

### 16.3 `OpenPIMemBackend.__init__`

保存：

```python
self.policy
self.config_name
self._policy_cameras
self._history
self._active_episode_id
self._contract
```

构造独立 contract：

```python
policy={
    "family": "openpi_mem",
    "name": config_name,
    "config": config_name,
    "short_term_memory": {
        "name": "mem_short_term_v1",
        "n_obs_steps": 6,
        "history_stride_steps": 20,
        "cameras": ["agentview_right", "eye_in_hand"],
        "include_proprio": True,
    },
}
```

server id 使用：

```text
openpi-mem-<config_name>
```

线协议仍然要求 canonical 当前帧的三路相机、37D state 和 prompt。backend 只把配置声明的两路相机写入历史。

### 16.4 `OpenPIMemBackend.from_checkpoint`

职责：

1. 从 `policy/openpi-mem` 导入 OpenPI。
2. 加载 `train_config`。
3. 调用 `_validate_mem_train_config(train_config)`，验证：

```python
train_config.model.model_type == ModelType.PI05_MEM
train_config.model.short_term_memory is True
train_config.model.n_obs_steps == 6
```

4. 创建 trained policy。
5. 从 config 读取实际相机、action horizon 和 memory 参数。
6. 构造 `OpenPIMemBackend`。

如果传入 vanilla π0.5 config，应立即拒绝，防止错误 server 加载错误 checkpoint。

同时实现与 `PolicyBackend` 接口一致的只读属性：

```python
@property
def contract(self) -> PolicyContract:
    return self._contract
```

### 16.5 `OpenPIMemBackend.infer`

执行顺序：

1. 验证已经调用 `reset(episode_id)`，并且 `request.episode_id == self._active_episode_id`。
2. 从 `InferenceRequest` 读取当前 canonical observation。
3. 调用 `_history.append(step=..., images=..., state=...)`。
4. 调用 `_history.stack()`。
5. 构造：

```python
policy_input = {
    "images": stacked_images,
    "state": stacked_state,
    "history_is_pad": history_is_pad,
    "prompt": observation.prompt,
}
```

6. 调用 `self.policy.infer(policy_input)`。
7. 验证并返回 `actions[50,13]`。
8. timing 中可额外记录数值字段：

```text
history_valid_steps
history_span_steps
model_ms
```

### 16.6 `OpenPIMemBackend.reset`

签名必须满足现有 `PolicyBackend` 协议：

```python
def reset(self, episode_id: str) -> None:
    self._history.reset()
    self._active_episode_id = episode_id

    reset = getattr(self.policy, "reset", None)
    if callable(reset):
        reset()
```

reset 测试必须证明前一个 episode 的图像和 state 不会进入新 episode。

### 16.7 `OpenPIMemBackend.close`

保持与其他 backend 一致：

```python
def close(self) -> None:
    close = getattr(self.policy, "close", None)
    if callable(close):
        close()
```

`close()` 不承担 episode reset；其职责只是释放 policy/JAX runtime 可能持有的资源。

## 17. 独立 MEM server

### 17.1 新增 `scripts/serve/serve_openpi_mem.py`

该文件独立于现有 `serve_openpi.py`，固定使用：

```text
policy/openpi-mem/src
policy/openpi-mem/packages/openpi-client/src
```

导入：

```python
from bench.policy.backends.openpi_mem import OpenPIMemBackend
```

命令行参数至少包括：

```text
--config
--checkpoint
--checkpoint-uri
--host
--port
--image-size
--state-schema
--default-prompt
--pytorch-device
--api-key-env
--inference-timeout-s
```

默认 server port 可以继续为 8000，但实际评测脚本应使用单独端口，例如 8010，避免与 vanilla server 冲突。

`code_revision` 必须分别记录：

```text
membench
policy/openpi-mem
```

现有 `scripts/serve/serve_openpi.py` 不修改。

## 18. MEM suite 的调用频率约束

新增的：

```text
wa01_smoke_pi05_mem_v1.yaml
wr03_smoke_pi05_mem_v1.yaml
```

必须设置：

```yaml
rollout:
  replan_steps: 20
  action_repeat: 1
```

π0.5-MEM 仍然输出 50-step action chunk，但 rollout 每次只消费前 20 个动作，然后重新调用 policy。这样 backend ring buffer 的相邻 observation 严格间隔 1 秒。

backend 还要验证 request step 间隔，避免 suite 配置错误时静默使用 2.5 秒历史。

## 19. 现有链路的隔离保证

第一版明确不修改：

```text
bench/policy/backends/openpi.py
bench/evaluation/rollout.py
bench/evaluation/client.py
scripts/serve/serve_openpi.py
bench/policy/protocol/*
```

因此：

- 现有 π0.5 No-History checkpoint 和评测脚本行为不变；
- vanilla backend 不出现 `if short_term_memory` 分支；
- vanilla server 不依赖 `policy/openpi-mem`；
- MEM 的 reset、history 和 step-frequency 错误只影响独立 MEM backend；
- 不需要升级协议版本或增加 extra modalities。

## 20. 单元测试要求

### 20.1 `siglip_mem_test.py`

必须覆盖：

1. temporal PE 当前帧严格为零。
2. 过去帧 PE 非零。
3. 第 4/8/12/16/20/24 层启用 ST。
4. MEM 与 vanilla SigLIP 参数树完全一致。
5. T=1 输出与 vanilla SigLIP 一致。
6. 输出只包含当前帧 tokens。
7. 修改 masked frame 不影响当前输出。
8. 修改有效历史帧可以影响当前输出。
9. causal mask 方向正确。
10. T 超过配置时报错。
11. current frame 永远有效。
12. FP32/BF16 forward 不产生 NaN。

### 20.2 `pi0_mem_test.py`

覆盖：

1. `PI05_MEM` 输入规格。
2. continuous state projection 为 `32→2048`。
3. prefix 中恰好 T 个 state tokens。
4. prompt tokenizer 不编码 state。
5. masked state 在 projection 前为零。
6. 修改 masked state 不影响 prefix。
7. `compute_loss` shape 正确。
8. `sample_actions` 输出 `[B,50,32]`。
9. SigLIP、state projector 和 action expert 梯度有限且非零。
10. vanilla PI0/PI05 测试继续通过。

### 20.3 数据集测试

覆盖：

1. episode 中段 offsets 正确。
2. episode 开头 padding 正确。
3. 历史绝不跨 episode。
4. action chunk 从 current 开始。
5. image/state history timestamp 对齐。
6. padded image/state 为零。
7. `history_is_pad[-1] == False`。
8. 视频可一次解码多个 timestamps。

### 20.4 checkpoint 测试

覆盖：

1. vanilla `pi05_base` 所有视觉权重成功加载。
2. 只允许缺少 `state_history_proj` 和 LoRA。
3. 随机删除一个 SigLIP 参数必须报错。
4. state projector 使用初始化值。
5. 保存后的 MEM checkpoint 可严格完整加载。

### 20.5 独立 MEM backend 测试

新增 `tests/serving/test_openpi_mem_backend.py`，覆盖：

1. 第一次 `step=0` 请求后，buffer 左侧有 5 个 padding slot，最后一个 slot 是当前 observation。
2. 请求 step 依次为 `0,20,40,60,80,100` 时，`stack()` 恰好得到 6 个有效 observation。
3. 完整窗口的 `history_steps` 恰好为 `[0,20,40,60,80,100]`，图像和 state 顺序一致。
4. 第二次请求为 `step=50` 时立即报错，明确提示 MEM suite 必须使用 `replan_steps=20` 和 `action_repeat=1`。
5. step 重复、倒退、跳步以及首个请求不是 step 0 时均立即报错。
6. episode 开头 padded image/state 为零，`history_is_pad` 正确，且当前 slot 永远有效。
7. buffer 保存 observation 的副本；请求返回后原地修改输入数组不能改变已保存的历史。
8. `reset()` 后 buffer 恢复为空，下一次请求必须重新从 step 0 开始，且不含前一个 episode 的数据。
9. `OpenPIMemBackend.reset()` 同时调用可用的 `policy.reset()`；policy 没有该方法时也能正常工作。
10. backend 传给 JAX policy 的最后一帧与当前 WS request 完全一致。
11. backend 只缓存配置声明的 MEM cameras，不把未使用的 canonical camera 写入时序视觉输入。
12. `from_checkpoint()` 拒绝 vanilla `PI05` config，只接受 `PI05_MEM` 且核验 `n_obs_steps=6`。
13. contract 的 family、server id 和 short-term-memory metadata 分别为 `openpi_mem`、`openpi-mem-<config>` 和实际配置值。
14. 使用 mock policy 做一次完整 infer，确认输出仍是协议要求的 `actions[50,13]`，不在 response 中暴露内部 history。
15. 现有 `OpenPIBackend`、`serve_openpi.py` 及其测试完全不依赖或导入 MEM backend。

## 21. 训练和验收顺序

### 21.1 阶段 A：数学正确性

不训练真实模型，完成：

```text
parameter-tree parity
T=1 parity
causal test
padding test
state-token test
checkpoint hot-start
```

这些检查失败时不能开始正式训练。

### 21.2 阶段 B：小数据过拟合

选择 8–16 个 episode：

```text
T=6
两路相机
完整 state history
```

验收：

- loss 明显下降；
- ST visual gradients 非零；
- state projector gradients 非零；
- 同一当前帧配不同有效历史时输出发生变化；
- masked history 不引起输出变化。

### 21.3 阶段 C：WA-01 工程 smoke

验证：

- 完整数据加载；
- checkpoint 保存/恢复；
- 普通 current-observation WS request 可进入独立 MEM server；
- MEM backend 在 server 内正确组装 T=6 history；
- 闭环动作；
- reset 清空 backend history；
- latency；
- peak GPU memory。

WA-01 只作为工程验收，不作为最终记忆能力结论。

### 21.4 阶段 D：WR-03/TM04 正式训练

记录：

```text
success rate
stage completion rate
history-dependent decision accuracy
inference latency
peak memory
history ablation sensitivity
```

虽然第一阶段不训练完整消融基线，但应做推理期 sanity check：

```text
将历史全部 mask 后输出是否变化
随机打乱历史后性能是否退化
```

这用于确认模型实际使用了历史，不作为独立 baseline。

## 22. PR 拆分

### PR 1：JAX ST SigLIP

文件：

```text
siglip_mem.py
siglip_mem_test.py
model.py
pi0_config.py
```

完成标准：

```text
T=1 parity
parameter-tree parity
causal/padding tests
```

预计 5–7 个工作日。

### PR 2：完整 π0.5-MEM 模型

文件：

```text
pi0.py
pi0_config.py
pi0_mem_test.py
weight_loaders.py
weight_loaders_test.py
```

完成标准：

```text
state-history tokens
pi05_base 热启动
loss/sample JIT
梯度测试
```

预计 4–6 个工作日。

### PR 3：MemBench 时序训练数据

文件：

```text
membench_v3_dataset.py
membench_policy.py
training/config.py
training/data_loader.py
compute_norm_stats.py
相关测试
```

完成标准：

```text
T=6 数据对齐
episode padding
norm stats 正确
小数据过拟合
```

预计 4–6 个工作日。

### PR 4：独立 MEM backend 和 server

文件：

```text
bench/policy/backends/openpi_mem.py
scripts/serve/serve_openpi_mem.py
tests/serving/test_openpi_mem_backend.py
configs/eval/suites/wa01_smoke_pi05_mem_v1.yaml
configs/eval/suites/wr03_smoke_pi05_mem_v1.yaml
scripts/eval/eval_pi05_mem.sh
```

完成标准：

```text
MEM backend 在 0/20/40/60/80/100 step 正确形成 T=6 history
非 20-step 请求间隔 fail fast
backend reset 无跨 episode 泄漏
独立 MEM server 可加载 PI05_MEM checkpoint
MEM contract/server id 与 vanilla OpenPI 可明确区分
通用 WS 协议、rollout、client 和 vanilla backend/server 零修改
闭环 smoke
```

预计 4–6 个工作日。

### PR 5：正式训练和评测配置

新增训练脚本、WA-01 smoke suite、WR-03/TM04 suite、日志和结果汇总。

训练和评测预计 1–2 周，取决于 GPU 数量和显存。

## 23. 主要风险与处理

### 风险 1：JAX 参数树不兼容

处理：

- 保留 `nn.scan`；
- 保留所有 Flax 子模块名字；
- 用自定义 `attention_fn` 而不是新建 Q/K/V projections；
- 参数树 parity 作为 PR 1 阻塞测试。

### 风险 2：T=1 数值不一致

处理：

- 当前帧 temporal PE 严格为零；
- T=1 直接走 vanilla spatial attention；
- FP32 要求严格一致；
- BF16 至少要求无可观测漂移，优先争取 bit-exact。

### 风险 3：padding 历史泄漏

处理：

- temporal key mask；
- state projection 前再次清零；
- padded pixels/state 随机替换不影响输出的测试；
- current frame 强制有效。

### 风险 4：视觉训练显存过高

每个样本需要：

```text
2 cameras × 6 frames = 12 images
```

处理顺序：

1. batch size 从 2 开始；
2. BF16；
3. 保留现有 remat；
4. FSDP；
5. 只编码两个真实相机；
6. 必要时再增加 gradient accumulation；
7. 最后才考虑只更新 ST layer 的 element-level gradient mask。

### 风险 5：200-seed 数据不足

处理：

- 先验证小数据过拟合；
- 优先 WR-03/TM04；
- 正式结果说明这是 task-specific post-training；
- 如短期记忆能力不足，再增加 video-language 或混合机器人视频预训练。

## 24. 最终交付定义

第一版完成不能只以“可以 forward”为标准，必须同时满足：

```text
1. 原 pi05_base 视觉权重严格加载
2. T=1 视觉编码与 vanilla 一致
3. T=6 causal ST attention 正确
4. padding history 不泄漏
5. 6 个 continuous proprio tokens 正确进入 prefix
6. 视觉塔在训练中获得非零梯度
7. dataset 与 MEM backend 使用相同的 T=6、20-step stride 时序语义
8. 给定相同的当前帧 request sequence，reset 后 backend 可确定性地重建同一 history
9. reset 不跨 episode
10. 非 20-step policy 请求间隔会被 backend 明确拒绝，而不是静默改变时间尺度
11. vanilla OpenPI backend、server、rollout、client 和协议均无代码改动
12. 能在 WR-03 或 TM04 完成长时闭环评测
```

总体原则：

```text
核心 JAX 模型保持无隐藏状态并接收显式 [T,...] tensor
episode history 仅由独立 MEM backend 管理
MEM 使用独立 backend、server、contract family 和评测 suite
vanilla backend/server/rollout/client/协议完全不变
视觉参数树与原 π0.5 兼容
完整短期记忆一次实现
vanilla policy 路径零侵入且不回归
```
