# Depth–UVD Geometric CoT for QwenGR00T: V1 Design

状态：V1 方法设计已确认；实现细节和 loss 配置单独展开。

## 1. 目标与范围

V1 在现有 QwenGR00T + Qwen3.5-VL + GR00T flow-matching action expert 上加入训练期几何中间表示：

1. 当前观测的主视角 dense metric depth；
2. 执行完整 action chunk 后的 future 主视角 dense metric depth；
3. 当前到 future 之间 EEF 的主视角 UVD 轨迹。

这些 depth/UVD decoder 只在训练时使用。推理时不显式生成 depth map 或 UVD 数值，而是使用最终层 UVD query embeddings 作为 action expert 的条件。

V1 不处理：EEF 遮挡建模、显式 action-conditioned future decoder、多模态 future 分布、速度/加速度监督和完整实验 ablation 设计。

## 2. 总体数据流

```text
RGB multi-view + language instruction
              ↓
        Qwen3.5-VL backbone
              ↓
 [V, L] → D_current → D_future → UVD
                                  ↓
                 [V, L, UVD_final_tokens]
                                  ↓
                         GR00T DiT Action Expert
                                  ↓
                           action chunk
```

query group 的顺序表达一种单向几何推理：先编码当前深度，再编码执行 chunk 后的未来深度，最后编码当前到未来的 EEF 轨迹。

## 3. Query groups 和 attention

### 3.1 Query groups

V1 使用三组独立初始化的 learnable query tokens：

```text
D_current：暂定 8 个 depth task tokens
D_future ：暂定 8 个 depth task tokens
UVD      ：一个共享 trajectory query 参数，运行时展开为 K 个时间 token
```

current/future depth query 不表示图像 patch 位置，而是 task-level geometric condition。二维空间位置由主视角 image patch tokens 保留。

### 3.2 Attention mask

采用 group-causal mask：

```text
[V, L] → D_current → D_future → UVD
```

具体含义：

- V/L 保留 Qwen 原生的 mask 行为；
- 每个 query group 内部使用 full attention；
- D_current 可以读取 V/L，但不能读取 future depth 或 UVD；
- D_future 可以读取 V/L 和 D_current；
- UVD 可以读取 V/L、D_current 和 D_future；
- 后面的 group 不反向影响前面的 group。

这对应参考仓库 GeoPredict 中的 block-causal/group-causal attention 思路。

## 4. Depth branch

### 4.1 Image feature

V1 不要求直接提取 Qwen3.5 vision encoder 的多层 DeepStack feature。使用 Qwen 多模态 backbone 最后一层 hidden states 中主视角的 image-token 部分：

```text
H_last = Qwen([V, L, D_current, D_future, UVD])
F_main = H_last 中主视角 image patch tokens
```

`F_main` 仍然保持主视角 patch token 的二维顺序，并 reshape 为：

\[
F_{main}\in\mathbb{R}^{B\times C\times H_p\times W_p}.
\]

主视角 depth decoder 使用 `F_main`；query 可以读取全部多视角和语言信息。

### 4.2 Decoder

V1 采用真实 image patch grid + ConvStack：

```text
F_main
  → 1×1 projection
  → residual convolution
  → upsampling
  → residual convolution
  → ...
  → dense depth map
```

ConvStack 在 decoder 内部逐级产生更高分辨率 feature，不要求 Qwen3.5 直接提供多层视觉 encoder feature。

DPT 作为后续方案保留。DepthVLM 的 DPT 直接使用 vision encoder 中间层和 LLM 最后一层 image tokens 构成多尺度 feature；但当前 Qwen3.5 wrapper 尚未暴露相同接口，因此不作为 V1 的主 decoder。

### 4.3 Query conditioning

current/future 使用相同的 ConvStack 和相同的 FiLM 参数生成器，差异来自不同 query embeddings。

对每组 depth query 做 mean pooling：

\[
q_c=\operatorname{mean}(Q_c),\qquad
q_f=\operatorname{mean}(Q_f).
\]

由小型 MLP 生成 FiLM 参数：

\[
(\gamma,\beta)=\operatorname{MLP}(q).
\]

在 ConvStack 的主要 feature stage 中调制：

\[
F'=(1+\gamma)\odot F+\beta.
\]

current/future depth 分别为：

\[
D_c=\operatorname{ConvStack}(F_{main},q_c),
\]

\[
D_f=\operatorname{ConvStack}(F_{main},q_f).
\]

FiLM 的最后一层采用接近零的初始化，使训练开始时 decoder 近似不改变 `F_main`，降低辅助任务对 action 主任务的干扰。

V1 暂不使用 query-to-spatial cross-attention。add 是轻量 ablation，cross-attention 是后续更强条件注入方案。

### 4.4 Depth output

current/future depth 都是主视角 dense metric depth map。输出和 supervision 的具体 loss（L1、MSE、SiLog、log-depth 等）在 loss 设计阶段确定。

V1 不加入 visibility mask，也不强制 EEF reference-point depth 与像素处的 visible-surface depth 相等。

## 5. UVD branch

### 5.1 Target

UVD 是主视角 EEF 的时间轨迹：

\[
p_k=(u_k,v_k,d_k).
\]

- (u,v) 是图像像素坐标；
- (d) 是相机坐标系下的 metric depth；
- 原始数据保留像素坐标；
- 经过图像 resize/crop 后同步变换坐标；
- 进入网络时再归一化到统一连续坐标。

### 5.2 Temporal sampling

future frame 定义为执行完整 action chunk 后的 observation。标签构造时必须保证：

```text
D_current 和 UVD_start 来自 current timestamp
D_future  和 UVD_end   来自 future timestamp
```

对于 action chunk 长度 $H$，V1 暂定：

\[
M=\lfloor0.3H\rfloor,
\qquad K=M+2.
\]

其中两个额外点是起点和终点，中间 $M$ 个点从 dense EEF trajectory 中均匀选择真实帧，不使用线性插值。

当前 baseline 的 action horizon 为 (H=8)，因此：

\[
M=2,\qquad K=4.
\]

### 5.3 Query construction

模型只定义一个共享 learnable trajectory query 参数：

\[
q_{traj}\in\mathbb{R}^{1\times1\times D}.
\]

运行时根据已知的 K 展开为 K 个 token，并加入连续时间 embedding：

\[
q_k^{UVD}=q_{traj}+E_{time}(\tau_k),
\qquad
\tau_k=\frac{k}{K-1}.
\]

因此：

- 参数层面只有一个共享 trajectory query；
- attention 层面有 K 个 UVD tokens；
- K 在推理时由已知 action horizon 决定；
- UVD tokens 并行预测，不使用自回归 decoder。

### 5.4 UVD prediction

UVD group 内部 full attention。每个最终 token 通过共享 MLP 回归一个 UVD 点：

\[
\hat p_k=\operatorname{MLP}(h_k^{UVD})\in\mathbb{R}^3.
\]

UVD 的 pointwise loss 是主 UVD supervision。速度和加速度不作为 V1 的额外目标。

## 6. Action expert condition

V1 将 Qwen 最终层的 V/L hidden states 与 UVD 最终层 tokens 拼接：

```text
[V, L, UVD_final_tokens]
        ↓
GR00T DiT encoder_hidden_states
```

UVD 放在 V/L 后面，保持与 Qwen 内部生成顺序一致。

当前 QwenGR00T action head 的 DiT 配置是交替结构：

```text
cross-attention → full self-attention → cross-attention → full self-attention → ...
```

其中 action/state/future tokens 在 full self-attention 中交互，并在 cross-attention 层读取 `[V,L,UVD]` condition。

Depth query 不直接送入 Action Expert；它们通过 `D_future → UVD` 的路径间接影响 action condition。

## 7. Training objective

V1 的总目标形式为：

\[
\mathcal L=
\mathcal L_{action}
+\lambda_c\mathcal L_{depth-current}
+\lambda_f\mathcal L_{depth-future}
+\lambda_u\mathcal L_{UVD}
\]

明确约束：

- $\mathcal L_{action}$ 是主 loss；
- depth/UVD 是辅助 supervision；
- endpoint MSE 暂不作为独立必选 loss，因为它与逐点 UVD MSE 重复；
- 不实现 depth-map/UVD 跨模态 endpoint loss、helper 或 diagnostic metric；
- 当前 active objective 使用 action、current depth、future depth 和 UVD 四项，具体权重以 summary 和实际 config 为准。

## 8. V1 理论假设

V1 采用以下方法假设：

1. current RGB + language 足以在当前数据分布中推断 policy-conditioned future geometry；
2. depth decoder 可以直接读取 image feature，depth query 是条件和辅助监督载体，不是严格 information bottleneck；
3. Action Expert 保留 V/L 直连，UVD 是额外 geometric condition，而不是强制 action 必须经过的 bottleneck；
4. 当前和 future depth 使用共享 decoder，query embedding 表达 temporal/task 差异；
5. EEF 在主视角中的遮挡暂时不是主要误差来源；
6. EEF UVD 位置本身比显式速度/加速度更适合作为第一版 geometric trace。

## 9. Deliberately deferred questions

以下问题不阻塞 V1 方法定义，但属于 implementation/loss/evaluation 阶段：

- Qwen3.5 image-token span 和主视角 patch grid 的具体提取；
- Qwen3.5 新增 query tokens 的 position IDs 和自定义 attention mask 实现；
- resize/crop 后 pixel UVD 的数据管线变换；
- depth/UVD loss 的具体形式和权重；
- 遮挡与 visibility mask；
- 不同 action horizon 下 K 的 batch padding；
- depth query 数量 4/8/16 的经验比较；
- FiLM 是否被 decoder bypass，以及 UVD 是否被 Action Expert 忽略。

## 10. Reference mapping

- `reference/DepthVLM/model/dpt_depth_head.py`：vision multi-scale feature + DPT dense depth；
- `reference/DepthVLM/model/modeling_qwen3_vl.py`：DeepStack 中间视觉 feature 与 LLM image tokens 的组合；
- `reference/lingbot-vla-v2/lingbotvla/models/vla/vision_models/align_heads/depth_head.py`：task query + Perceiver resampler；
- `reference/lingbot-vla-v2/.../modeling_lingbot_vla_v2.py`：8 task tokens 与 depth latent tokens 的分工、current/future depth query；
- `reference/lingbot-depth/mdm/model/modules_decoder.py`：多尺度 ConvStack、residual block 和上采样 decoder；
- `reference/GeoPredict/models/keypoints.py`：少量 query 汇总带时间位置的轨迹输入；
- `reference/GeoPredict/models/geopredict.py`：group-causal attention 和空间位置编码。


## 11. Implementation boundary

V1 的实现遵循“新增相邻功能文件，最小化修改 baseline”的约束：

- depth/UVD query、group-causal mask 和 query-to-backbone adapter 放在现有 `starVLA/model/modules/` 的相邻功能目录中；
- depth ConvStack/FiLM decoder 放在 depth 或 projector 类似功能目录中，不把 decoder 逻辑直接塞进 QwenGR00T 主类；
- LIBERO depth/camera/EEF-UVD 读取和标签配对放在 `starVLA/dataloader/` 或其相邻的新 dataset adapter 文件中；
- action head 只做必要的 condition 接口注册；不重写现有 QwenGR00T 或 DiT baseline；
- 每个新增模块先提供独立的 shape/label 单测，再接入训练脚本。

现有 baseline 文件只有在注册新 model/config 或传递新 condition 时做最小更新。参考脚本
`examples/modelExtensions/CoT/scripts/build_libero_replay_rgbd_lerobot.py` 作为数据约定来源，
不直接把它改造成训练时的隐式标签生成器。

## 12. LIBERO RGB-D / EEF-UVD data contract

### 12.1 Dataset scope

数据根目录为：

```text
/root/data/yxz/datasets/libero_rerender
```

包含四个 LeRobot 数据集：`libero_10`、`libero_goal`、`libero_object` 和
`libero_spatial`。当前核对到的共同设置是：20 FPS、RGB 256×256、主视角为
`agentview`，腕部视角为 `robot0_eye_in_hand`。

Parquet 中的核心字段为：

```text
observation.images.image
observation.images.wrist_image
observation.state       # float32[8]
action                  # float32[7]
timestamp
frame_index
episode_index
task_index
observation.depth.image_m_path
observation.depth.wrist_m_path
observation.camera.params_path
```

`observation.state` 的布局已经在 `meta/info.json` 中明确为：

```text
[robot0_eef_pos(3), quat2axisangle(robot0_eef_quat)(3), robot0_gripper_qpos(2)]
```

因此 EEF 世界位置是 `state[..., 0:3]`；姿态和 gripper 不参与 V1 的 UVD 回归。

### 12.2 Depth and camera files

以 `libero_goal` 的首个 episode 为例：

```text
depth/.../observation.depth.image_m/episode_000000.npz
  └── depth_m.npy                  # float16[112, 256, 256], meters

depth/.../observation.depth.wrist_m/episode_000000.npz
  └── depth_m.npy                  # float16[112, 256, 256], meters

camera/.../episode_000000.npz
  ├── agentview_K.npy               # float32[112, 3, 3]
  ├── agentview_T_world_camera.npy  # float32[112, 4, 4]
  ├── wrist_K.npy                   # float32[112, 3, 3]
  └── wrist_T_world_camera.npy      # float32[112, 4, 4]
```

V1 只使用 `agentview` 的 depth 和 camera 参数。depth 的单位是米；网络输入和
depth supervision 可以保留米制，但 loss 需要在实现阶段显式记录尺度策略。

### 12.3 EEF UVD is derived, not currently stored

当前 rerender 输出目录没有独立的 `observation.eef.*`、`uvd` 或 `trace` 文件，也没有在
`meta/modality.json` 中注册这些字段。UVD 标签应在 dataset adapter 中按帧派生：

```text
p_world = observation.state[0:3]
T_camera_world = inverse(agentview_T_world_camera)
p_camera = T_camera_world @ [p_world, 1]
[u', v', w'] = agentview_K @ p_camera[0:3]
u = u' / w'
v = v' / w'
d = p_camera[2]
```

得到的 `(u, v, d)` 就是主视角 EEF UVD。这里的 `d` 是相机坐标系 z 深度，
与 depth map 在该坐标约定下使用同一米制单位；它不是 EEF 到相机的欧氏距离。

注意 `agentview_K` 已经匹配默认 rerender 的 `rot180` 存储图像约定。后续若改变
resize/crop，必须在 dataset adapter 中同步变换 `u,v` 和 depth target，不能只变换 RGB。
投影后应检查 `d > 0` 以及 `0 <= u < W, 0 <= v < H`，越界点用 valid mask 标记。

现有生成脚本曾识别过以下旧字段，但在最终输出时主动删除：

```text
observation.eef.agentview_uv
observation.eef.wrist_uv
observation.eef.agentview_depth_m
observation.eef.wrist_depth_m
```

所以训练代码不能依赖这些 key；第一版应统一在线派生，或由新的离线预处理 adapter
显式缓存一个命名清楚的 `eef_agentview_uvd`。

## 13. Training sample pairing from LIBERO

对当前帧索引 `t` 和 action chunk 长度 `H`，数据 adapter 需要返回：

```text
RGB multi-view: frames at t (or the configured observation window)
D_current:      agentview depth at t
actions:        action[t : t+H]
D_future:       agentview depth at the observation timestamp after the complete chunk
UVD_dense:      projected EEF UVD on the real frames from current to future
UVD_K:          endpoints plus uniformly selected real intermediate frames
```

future frame 不能简单假定为 `t + H`，除非已经确认 action、camera 和 parquet frame
使用同一频率、同一时间索引。当前 LIBERO 元数据虽然都是 20 FPS，但 adapter 仍应基于
`timestamp/frame_index` 和 episode 边界检查配对；不足以形成完整 chunk 的样本直接过滤。

V1 的 UVD 采样仍使用：

\[
M=\lfloor0.3H\rfloor, \qquad K=M+2,
\]

保留 current/future 两个端点，内部点从 `UVD_dense` 的真实帧均匀抽取，不做线性插值。
若某个投影点越界或深度无效，adapter 输出对应的 `uvd_valid_mask`；V1 默认 loss 暂时
只使用有效点，但是否启用更严格的 visibility 规则仍属于后续实现决策。

## 14. Known / Assumptions / Unknowns / Evidence update

### Known

- 四个 rerender LIBERO 子集的 fps、RGB 分辨率、depth 单位和核心 key 已核对；
- depth 是逐 episode 的 `depth_m.npy`，主视角数组为 `[T,256,256]`；
- camera NPZ 提供逐帧 `agentview_K` 和 `agentview_T_world_camera`；
- EEF 世界位置可从 `observation.state[0:3]` 读取，UVD 可由相机投影派生；
- 当前 QwenGR00T action head 接收 VLM condition，V1 继续使用 `[V,L,UVD]`。

### Assumptions

- 默认使用 `agentview` 作为主视角，并使用 rerender 已匹配 rot180 图像的 intrinsics；
- V1 通过有效坐标检查处理越界/不可投影 EEF 点，不引入复杂遮挡模型；
- action chunk 的 future observation 配对在 dataset adapter 离线完成。

### Unknowns

- 实际训练 dataloader 的 observation window 与 `action[t:t+H]` 的时间边界；
- resize/crop 具体发生在哪一层，以及 depth resize 的插值和 invalid value 规则；
- UVD 派生标签是每次读取计算，还是先离线缓存为新字段；
- valid/visibility mask 在 depth loss、UVD loss 和 consistency loss 中的具体传播方式；
- query/mask/decoder 接入现有 Qwen3.5 wrapper 的代码接口。

### Evidence

- `/root/data/yxz/datasets/libero_rerender/*/meta/info.json`：字段、fps、分辨率、单位、路径模板；
- `/root/data/yxz/datasets/libero_rerender/*/meta/modality.json`：LeRobot state/action/video 映射；
- 首个 `libero_goal` episode 的 NPZ headers：depth `[112,256,256]`，camera `K=[112,3,3]`、`T=[112,4,4]`；
- `examples/modelExtensions/CoT/scripts/build_libero_replay_rgbd_lerobot.py`：agentview 映射、rot180 intrinsics 和 world-to-pixel 投影约定。

## 15. Implementation evidence update (2026-07-16)

### Implemented

- `CoTLeRobotSingleDataset` now reads the rerendered LIBERO depth/camera NPZ files,
  derives main-camera EEF UVD from `observation.state[...,0:3]`, checks projection
  validity, resizes metric depth, and returns real-frame UVD samples. For `H=8`, the
  default `K=floor(0.3H)+2=4`, with frame indices including both endpoints.
- `GeometricQueryReasoner` implements independent current/future depth queries, one
  shared trajectory seed expanded to runtime `K` tokens, time embeddings, and the
  group-causal mask `[V,L] -> D_current -> D_future -> UVD`.
- `SharedFiLMConvStack` is shared by current/future depth and predicts positive
  metric depth. The action condition is `[V,L,UVD_final_tokens]`; `predict_action`
  does not run numeric depth/UVD heads.
- The implementation is connected through the new `QwenGR00TCoT` framework and a
  dedicated trainer/config/launcher. Existing Qwen V/L attention behavior and the
  baseline DiT action implementation are reused.

### Evidence

- Real adapter inspection across all four datasets succeeded. Dataset lengths were
  `66984, 52042, 52970, 101469`; a real sample returned action `(8,7)`, depth
  `(1,224,224)`, UVD `(4,3)`, and endpoint-preserving indices `[0,3,5,8]`.
- A real Qwen3.5 forward smoke produced final hidden `[1,178,2560]`, action condition
  `[1,182,2560]`, UVD query tokens `[1,4,2560]`, and depth `[1,1,224,224]`.
- A real forward/backward smoke produced finite losses and nonzero gradients in the
  query reasoner, depth decoder, and action branch.
- Focused unit tests pass: `13/13` in `tests/test_*cot*.py`; a fresh real inference
  check also produced an action array `(1,8,7)` while hooks verified that neither
  numeric depth nor numeric UVD decoder was called.
- Initialization-scale calibration on 32 real samples (8 from each of the four
  LIBERO rerender subsets) measured mean raw losses: action `1.33017`, current
  depth `0.37218`, future depth `0.36581`, and UVD `0.02131`. The explicit action
  coefficient is `lambda_action=1.0`; auxiliary coefficients are `0.14, 0.15,
  0.62`, targeting roughly 4%, 4%, and 1% of action loss; action remains dominant. Full values and
  per-dataset measurements are in
  `/root/data/yxz/outpus/qwen3_gr00t_libero_CoT_v1/loss_calibration.json`.

### Pilot status and remaining unknowns

- The requested 10k-step launcher was attempted. Dataset/model/DeepSpeed setup
  completed, but the run did not complete a training step because external processes
  occupied the selected GPUs; the exact logs are preserved under
  `/root/data/yxz/outpus/qwen3_gr00t_libero_CoT_v1/logs/`.
- A one-GPU batch-1 trainer smoke reached the optimizer step, then hit the same
  external-memory condition while DeepSpeed materialized a 16.91 GiB FP32 partition
  with only 13.65 GiB free. Therefore the auxiliary weights are calibrated
  initialization-scale weights, not yet weights validated by a learning-curve pilot.
- V1 intentionally does not solve true EEF visibility/occlusion; invalid projection
  points are masked, while a visible foreground surface can still disagree with the
  EEF's camera-z depth. This remains a known data/model limitation.
- The adapter currently uses `t+H` after episode-boundary checks for the future frame;
  if a dataset changes action execution latency or timestamp semantics, this pairing
  must be revisited before making a cross-dataset claim.

### Final Known / Assumptions / Unknowns / Evidence state

**Known:** the local rerender data contract, projection convention, metric-depth units,
actual Qwen hidden/condition shapes, group-causal query implementation, decoder/loss
interfaces, and finite real forward/backward behavior are verified.

**Assumptions:** `agentview` is the main view; `state[0:3]` is the EEF world position;
`H=8` and `t+H` represent the completed action chunk for this LIBERO export; invalid
projection points are handled by masks; auxiliary weights start from the recorded
calibration.

**Unknowns:** whether action/camera timestamps remain aligned for other exports;
learning-curve stability and final loss weights under full training; how much the
DiT uses UVD versus V/L; and whether occlusion-aware visibility labels are needed.

**Evidence:** focused tests, real LIBERO adapter smoke, real Qwen forward/backward smoke,
32-sample four-subset calibration JSON, resolved config and launcher, plus the
preserved pilot logs listed above.
