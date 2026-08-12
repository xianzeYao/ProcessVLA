# QwenGR00T CoT V2 / V3：设计与模型架构问答

这份文档用“你问—我答”的方式说明仓库中已经实现的 QwenGR00T CoT V2 与 V3。讨论范围是当前代码和已经生成的数据，不把尚未实现的设想写成现状。

## 先给结论

**你：V2 和 V3 分别解决什么问题？**

我：两者都让策略在预测动作之外，显式学习场景几何：当前深度、未来深度和末端执行器轨迹。

- V2 把末端轨迹表示成一个或两个 UVD 点序列。LIBERO 是单个 EEF 中心轨迹；RoboCasa 是左右手各一个 pinch 点轨迹。
- V3 是 LIBERO 专用实验，把每个轨迹时刻扩成 `left_finger_tip`、`right_finger_tip`、`wrist_hand_base` 三个点，在保留 EEF 运动信息的同时显式编码夹爪开合。
- 两者都只在主视角下重建 depth 和 UVD。LIBERO 的 wrist RGB 会输入模型，但 wrist depth 目前没有进入 reconstruction loss。

**你：模型最终还是动作模型吗？**

我：是。Depth 和 UVD 是辅助几何任务，主任务仍是 action diffusion。几何 token 与原始视觉语言 token 一起经过 Qwen，随后作为 action expert 的条件，因此辅助任务不是独立旁路，而是在训练 Qwen 内部可被动作预测使用的几何表示。

## 数据到底进入了什么

**你：LIBERO 的 RGB 视角有哪些？**

我：当前 dataloader 输入两路 RGB：

- `primary_image`：第三视角 / agentview；
- `wrist_image`：腕部 eye-in-hand 视角。

当前和未来 depth 都来自 `observation.depth.image_m_path`，也就是主视角。UVD 也始终使用主视角的相机内外参或已经投影好的 agentview sidecar。这样 UVD 的坐标系始终唯一，不会把 wrist camera 和 agentview 的内外参混在一起。

数据里已经存在 `observation.depth.wrist_m_path`，但 V2/V3 当前都没有把它作为训练 target。后续可以增加独立 wrist-depth decoder/loss，而 UVD 仍保持 agentview 单视角。

**你：RoboCasa 的视角呢？**

我：当前 RoboCasa/GR1 数据输入 `ego_view` RGB，depth 和 UVD 也都在这一个主视角下构造。当前没有 wrist-camera RGB reconstruction 分支。

**你：UVD 的三个量是什么？**

我：`u`、`v` 是图像平面坐标，进入模型前归一化到 `[0,1]`；`d` 是相机坐标系中的正向深度，单位是米，再除以配置中的 `uvd_depth_scale`。遮挡不会使几何点失效，因为这些点来自仿真状态和相机投影，不要求它们对应当前图像中可见的表面像素。训练 validity 仍要求投影有限、深度为正，并按数据版本处理是否在图像范围内。

## V2：单点或双点轨迹

**你：V2 的监督点具体是什么？**

我：

- LIBERO：从 `observation.state[:3]` 取得 EEF 世界位置，再投影到 agentview，得到一条 UVD 轨迹。
- RoboCasa：左右手分别使用 thumb-index pinch 点，投影到 ego/agentview，得到两条 UVD 轨迹。

因此 V2 的统一抽象是 `[time, hand, coordinate]`。LIBERO 的 `hand_count=1`；RoboCasa 的 `hand_count=2`。

**你：V2 的 token 顺序是什么？**

我：是 time-major，而不是先放完一只手的整条轨迹。设每只手采样 `K` 个时刻：

```text
LIBERO:   [P0, P1, ..., P(K-1)]
RoboCasa: [L0, R0, L1, R1, ..., L(K-1), R(K-1)]
```

每个 UVD query embedding 由以下部分相加：

```text
trajectory_seed + time_embedding(t) + hand_embedding(hand_id)
```

LIBERO 只有一个 hand，因而不需要额外 hand embedding；RoboCasa 用 hand embedding 区分左右手。

**你：V2 能恢复什么？**

我：单个 EEF 中心点直接表达 xyz 轨迹，但不能仅靠这个点恢复完整旋转。RoboCasa 的左右 pinch 点分别表达两只手的代表性运动轨迹，也不能被描述成单个刚体的完整 SE(3) 恢复。V2 的目标是让模型学习“末端往哪里走”，不是严格的刚体姿态参数化。

## V3：LIBERO 三点轨迹

**你：V3 的三个点是哪三个？**

我：固定顺序是：

```text
0: left_finger_tip
1: right_finger_tip
2: wrist_hand_base
```

仿真中对应：

```text
gripper0_finger_joint1_tip
gripper0_finger_joint2_tip
gripper0_right_gripper
```

每个 episode 的 sidecar 保存世界坐标、agentview UVD、正深度投影 validity 和 in-frame validity。它不修改原有 Parquet，只在数据集根目录增加确定性路径的 NPZ 与可选 metadata。

**你：V3 的 token 顺序是什么？**

我：仍然是 time-major：

```text
[L0, R0, W0, L1, R1, W1, ..., L(K-1), R(K-1), W(K-1)]
```

每个 query embedding 是：

```text
trajectory_seed + time_embedding(t) + landmark_embedding(landmark_id)
```

当前 LIBERO 配置 `K=4`，所以 V2 有 4 个 UVD token，V3 有 `4 × 3 = 12` 个 UVD token。

**你：为什么需要 landmark embedding？**

我：同一时刻三个点共享相同 time embedding。如果没有 landmark embedding，模型无法从 query 身份上区分左指、右指和 wrist，只能依赖位置顺序隐式猜测。显式 landmark embedding 把语义写进模型参数，同时不需要 dataset embedding 或 effector embedding，因为当前 V3 只训练 LIBERO 单一末端类型，不做跨数据集混训。

**你：这三个点真的编码了 6D 和 gripper opening 吗？**

我：在当前 LIBERO 平行夹爪运动学下，实验证据非常强。全量 1,693 episodes、273,465 帧中没有退化三角形：

- finger-tip distance 对 gripper aperture 的 held-out 线性恢复 `R² = 0.9999999999999374`；
- aperture MAE 约 `4.98e-9 m`；
- 用三角形构造局部 frame，再拟合一个固定 triangle-to-EEF frame offset，held-out rotation geodesic error 的 median 为 `1.54e-5°`，p95 为 `1.01e-4°`。

所以针对这套已知机械结构，可以说三点几乎无损地编码了 EEF 6D pose 和 gripper aperture。

但严格的通用刚体表述需要一句限定：左右 finger tip 会随夹爪开合，不是三个固定在同一刚体上的点。这里之所以能恢复，是因为平行夹爪运动学已知且关系稳定。若以后要做不依赖夹爪运动学的纯 SE(3) 表示，应在 wrist/palm 同一刚体上定义三个固定且不共线的局部点。

## Qwen 内部的 geometry token 架构

**你：geometry token 插在序列哪里？**

我：Qwen 先按原流程构造视觉语言 native tokens，然后在序列尾部追加固定数量的 geometry slots：

```text
[native Qwen tokens]
[current-depth queries]
[future-depth queries]
[UVD trajectory queries]
```

Depth current/future 各有 `Q` 个 query；UVD 数量由版本、采样时刻和 hand/landmark 数决定。这些 slot 在训练和推理时都存在，不会只在有标签时插入。

**你：模型有没有把 GT UVD 或 GT 时间直接喂给 Qwen，造成 label leakage？**

我：没有。Backbone 使用由配置确定的固定 query 和默认归一化时间。Ground-truth UVD、valid mask、实际 sampled time 只在 backbone 输出后用于 target packing、loss 和诊断，不进入 Qwen hidden-state 计算。

**你：attention 怎么设计？**

我：在使用 SDPA full-attention 的 Qwen 层中：

- native token 保持原有因果读取关系；
- current-depth query group 内部可以互相读取；
- future-depth query group 内部可以互相读取；
- 同一时间的多 hand / 多 landmark UVD token 可以双向互读；
- 后续时间可以读取此前时间，较早时间不能读取未来时间。

Qwen 的 linear-attention 层保持自身原生路径，full-attention 层应用上述 geometry mask。这样同一时刻的 `L/R/W` 能共同形成一个三角形状态，同时保持跨时间的因果顺序。

## Depth 与 UVD 怎么解码

**你：depth query 最后怎么变成一张深度图？**

我：current 和 future 两组 depth hidden tokens 分别经过同一个 learned attention-pooling 模块，得到两个 summary。主视角图像 token 与 summary 一起进入共享的 FiLM convolution decoder，输出：

```text
depth_current
depth_future
```

两张图共享 decoder 参数，但 condition summary 不同。当前 decoder 只针对主视角图像 token；wrist RGB 不对应独立 wrist-depth 输出。

**你：UVD 怎么解码？**

我：每个 UVD hidden token 经过共享 MLP head，输出三个值：

- `u,v` 通过 sigmoid 限制到归一化图像范围；
- `d` 通过 softplus 保证为正。

共享 head 的含义是：点或 landmark 的身份来自其 hidden token 中的 time/hand/landmark embedding，而不是为每个点维护独立回归头。

## Loss 设计

**你：V2 的 UVD loss 是什么？**

我：V2 包含两项：

```text
L_uvd_v2 = L_absolute + λ_relative * L_adjacent_delta
```

- `L_absolute`：对每个有效 UVD 坐标做 masked Smooth-L1；
- `L_adjacent_delta`：分别沿每只手自己的时间轨迹，监督相邻时间位移。

第二项比较的是同一条轨迹的 `pred[t+1]-pred[t]` 与 GT delta，而不是比较 flatten 后相邻的左右手 token。

**你：V3 增加了什么？**

我：V3 把 temporal loss 独立应用到三个 landmark，并提供一个同时间三角形 shape loss：

```text
grasp_axis = right - left
wrist_axis = wrist - 0.5 * (left + right)

L_uvd_v3 = L_absolute
         + λ_temporal * L_adjacent_delta
         + λ_shape * L_triangle_shape
```

Shape loss 比较 `grasp_axis` 和 `wrist_axis`，对整体平移不敏感，并且只有同一时刻三个点都有效时才计算。首个 V3 配置中 `lambda_uvd_shape=0.0`：模型会记录 shape metric，但它暂时不参与优化，便于后续做受控 ablation。

**你：总 loss 怎么组成？**

我：V2/V3 外层一致：

```text
L_total = λ_action * L_action
        + λ_depth_current * L_depth_current
        + λ_depth_future * L_depth_future
        + λ_uvd * L_uvd
```

Depth 使用有效像素上的 masked Smooth-L1；action loss 来自 DiT action expert。

## Action expert 到底看哪些 token

**你：动作预测默认能看到 depth token 吗？**

我：代码支持两种明确模式：

```text
include_depth_in_action_condition = false
condition = [native, UVD]

include_depth_in_action_condition = true
condition = [native, depth_current, depth_future, UVD]
```

这个选项必须是布尔值，字符串、整数或 `null` 会直接报错，避免 YAML 写错后静默改变实验含义。

历史 V2 配置可以保持 `false`，行为不变；新增的 LIBERO/RoboCasa V2 depth-condition 实验和当前 LIBERO V3 配置设为 `true`。因此这些实验会让 action expert 直接使用两组 depth query hidden states，而不仅仅依赖 depth loss 间接塑造 Qwen。

**你：action expert 本身是什么？**

我：当前配置使用 DiT-B diffusion action model。它接收 Qwen/native/geometry condition、可选 state 和训练 action target；训练时重复采样 diffusion steps，推理时按配置的 inference timesteps 生成归一化动作，再由部署层反归一化。

LIBERO 当前是 `action_dim=7`、`action_horizon=8`；RoboCasa Fourier 当前是 `action_dim=29`、`action_horizon=16`。

## 训练与推理数据流

**你：一次训练 forward 的完整顺序是什么？**

我：

```text
RGB + instruction
  -> Qwen native inputs
  -> append current-depth / future-depth / UVD query slots
  -> Qwen geometry-aware forward
  -> split native / depth-current / depth-future / UVD hidden states
  -> decode two depth maps and UVD trajectory
  -> build action condition
  -> DiT action loss
  -> aggregate action + depth + UVD losses
```

Dataset 同时提供 depth/UVD target 和 masks，但 target 不参与 query 构造。

**你：推理时必须有 depth GT 或 sidecar 吗？**

我：模型 forward 不需要 GT geometry 才能生成 geometry token。`predict_geometry` 可以从 RGB/语言输出 depth 与 UVD；`predict_action` 也从相同的 learned query hidden states构造动作条件。Sidecar 是训练监督和离线评估资产，不是部署时输入。

## V2 与 V3 的兼容边界

**你：V3 会影响现有 V2 数据读取或训练吗？**

我：不会。V3 使用独立的 dataloader dispatch、framework registry key、YAML、trainer 和 launcher。三点数据以 sidecar 方式加入，没有改写 Parquet。真实 V2 sample 在 sidecar/metadata 加入前后的 keys、shape、dtype 和 array bytes 的 SHA256 完全一致。

**你：V2 checkpoint 能直接 strict-load 到 V3 吗？**

我：不能把 V2 geometry checkpoint 宣称为 V3 strict-compatible。V3 新增 landmark embedding，UVD token 数也从 `K` 变成 `3K`。V3 checkpoint validation 会拒绝缺少 `geometry_tokens.landmark_embedding` 的 V2 geometry state。基础 VLM/action 初始化仍可用于新建 V3 run。

V2 自身也会拒绝早期缺少 shared depth-attention pooling 参数的旧 mean-pooling geometry checkpoint，避免表面上加载成功、实际架构不一致。

## 当前实验配置

**你：LIBERO V3 第一版具体怎么配？**

我：核心参数是：

```yaml
framework:
  name: QwenGR00TCoTV3
  geometry:
    depth_query_count: 8
    include_depth_in_action_condition: true
    uvd_num_points: 4
    landmark_count: 3
    lambda_uvd_temporal: 0.1
    lambda_uvd_shape: 0.0

datasets:
  vla_data:
    dataset_py: cot_v3_lerobot_datasets
```

对应 8 个 current-depth queries、8 个 future-depth queries 和 12 个 UVD queries。

**你：现阶段哪些事情明确没做？**

我：

- wrist depth 已有数据，但尚未加入 decoder 与 loss；
- RoboCasa V3 尚未实现；
- V3 首轮不启用 triangle-shape loss；
- 尚未把三个点改成同一刚体上的固定非共线点；
- 不把当前 moving-finger triangle 的恢复能力无条件推广到未知夹爪结构。

## 代码索引

**你：想从代码里核对，先看哪里？**

我：

- V2 framework：`starVLA/model/framework/VLM4A/QwenGR00TCoTV2.py`
- V3 framework：`starVLA/model/framework/VLM4A/QwenGR00TCoTV3.py`
- V2 geometry layout/mask/packing：`starVLA/model/modules/geometric_cot_v2.py`
- V3 landmark layout/mask/packing：`starVLA/model/modules/geometric_cot_v3.py`
- geometry losses：`starVLA/model/modules/cot_losses.py`
- LIBERO V2 dataloader：`starVLA/dataloader/cot_lerobot_datasets.py`
- LIBERO V3 dataloader：`starVLA/dataloader/cot_v3_lerobot_datasets.py`
- RoboCasa V2 dataloader：`starVLA/dataloader/robocasa_lerobot_datasets.py`
- V3 trainer：`starVLA/training/train_starvla_cot_v3.py`
- V3 YAML：`examples/modelExtensions/CoT/configs/qwen35_gr00t_libero_CoT_v3_q0_depthcond.yaml`
- V3 launcher：`examples/modelExtensions/CoT/scripts/run_qwen35_gr00t_libero_CoT_v3.sh`
- 全量 sidecar 验证：`artifacts/libero_gripper_triangle_v3/sidecar_validation_report.json`
- 恢复能力统计：`artifacts/libero_gripper_triangle_v3/recovery_validation_report.json`
- V2 before/after 兼容报告：`artifacts/libero_gripper_triangle_v3/v2_compatibility_report.json`
