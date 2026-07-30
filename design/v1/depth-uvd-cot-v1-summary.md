# Depth–UVD Geometric CoT V1：具体设计与验证清单

> 本文是 V1 设计的执行摘要。完整设计、数据契约和实现证据见
> design/depth-uvd-cot-v1-design.md。

## Current implementation update (2026-07-29)

当前 active objective 只有 Action、current depth、future depth 和 UVD。UVD d 与 visible-surface depth 不保证同一物理点，因此代码不实现二者的跨模态 endpoint loss、helper 或 diagnostic metric。

精度口径也已更新：Qwen3.5 VLM 按现有 wrapper 使用 BF16；CoT Action Expert 不额外强制 FP32，而是对齐原始 QwenGR00T baseline 的 action call/autocast 语义。评测精度取决于具体入口：共享 policy server 可能整体 cast 为 BF16，standalone evaluator 可能采用选择性 cast，运行结果必须记录入口和 `USE_BF16`。

## 1. V1 要解决的问题

V1 的核心假设是：当前深度、执行 action chunk 后的未来深度，以及 current→future 的 EEF 3D 轨迹，可以作为一种几何 CoT，帮助 action expert 建立更稳定的空间和时间条件。

候选研究命题是：

> 训练期的 current/future depth 与 UVD trajectory auxiliary supervision，能够让 VLA backbone 形成对 action-relevant geometry 的中间表示；推理时只保留 UVD query embedding，也能改善 action prediction。

这仍然是待实验验证的 hypothesis，不应把 auxiliary decoder 能够下降当作 action improvement 的证据。

## 2. 总体数据流

~~~text
RGB multi-view images + language instruction
                    ↓
              Qwen3.5-VL
                    ↓
             final hidden states
                    ↓
       query reasoner with group-causal mask
                    ↓
   D_current → D_future → UVD trajectory tokens
       ↓            ↓              ↓
  depth decoder  depth decoder   UVD head
       ↓            ↓              ↓
 D_current_hat  D_future_hat     UVD_hat
                                  │
             [V, L, UVD_final_tokens]
                                  ↓
                         DiT Action Expert
                                  ↓
                            action chunk
~~~

训练时计算三个 active auxiliary loss（current depth、future depth、UVD），另保留零值 geometry compatibility metric；推理时不生成数值 depth map 或 UVD，只使用最终 UVD token embedding。

## 3. Query reasoner

### 3.1 Query 数量

- current depth：8 个独立 learnable query tokens；
- future depth：8 个独立 learnable query tokens；
- UVD：1 个共享 trajectory seed，运行时扩展为 K 个 token；
- 当前 action horizon H=8，采用 K=floor(0.3H)+2=4；
- UVD 每个 token 加入 tau=k/(K-1) 的连续时间 embedding。

因此“一个 trajectory query”指参数只有一个 seed；计算图中仍然有 K 个时间 token，分别回归 K 个 UVD 点。

### 3.2 信息流 mask

~~~text
[V, L] → D_current → D_future → UVD
~~~

- Qwen 原生 V/L mask 保留不改；
- query group 内部是 full attention；
- current depth 只能读取 V/L；
- future depth 可读取 V/L 和 current depth；
- UVD 可读取 V/L、current depth 和 future depth；
- 后面的 group 不反向影响前面的 group。

当前实现是在 Qwen 最后一层 V/L hidden 后增加 query reasoner，而不是重写 Qwen3.5 的 hybrid attention 内部。

## 4. Depth branch

- 使用 Qwen 最后一层 hidden states 中主视角的 image patch token；
- 取第一段连续 image-token run，保持 raster order；
- 从实际 token 数推断 patch grid；
- current/future query group 分别 mean-pool 成一个 query condition；
- 两个 depth 分支共享同一个 ConvStack 和 FiLM 参数；
- query 通过 FiLM 调制 spatial feature：

$$
F'=(1+\gamma(q))\odot F+\beta(q)
$$

- decoder 配置为 256 feature channels、3 个 residual/upsampling stages，输出 224×224；
- 输出使用 Softplus，保证 metric depth 为正；
- current 和 future 使用同一主视角 image feature，但由不同 query condition 区分。

V1 不使用 DPT 多尺度 encoder feature；DPT/DeepStack 是后续增强方案。

## 5. UVD branch

每个点为：

$$
p_k=(u_k,v_k,d_k)
$$

- u,v：主视角像素坐标；
- d：相机坐标系 z-depth，单位为米；
- 原始像素坐标由 state 和 camera pose 投影得到；
- resize 后进入网络/loss 前归一化到 [0,1]；
- depth 不做 [0,999] 离散化，保持 metric depth；
- UVD token 通过共享 MLP 回归 (u,v,d)，uv 使用 sigmoid，depth 使用 Softplus；
- 不使用自回归 decoder，也不对中间 UVD 点做线性插值。

## 6. LIBERO 数据契约

数据根目录：

~~~text
/root/data/yxz/datasets/libero_rerender
~~~

使用四个 rerender LeRobot 子集：

~~~text
libero_object
libero_goal
libero_spatial
libero_10
~~~

主要字段：

~~~text
observation.images.image
observation.images.wrist_image
observation.state              # EEF xyz, axis-angle, gripper
action                         # 7D
observation.depth.image_m_path
observation.camera.params_path
~~~

主视角只使用 agentview。每个 episode 的 NPZ 提供：

~~~text
depth_m
agentview_K
agentview_T_world_camera
~~~

EEF UVD 由以下步骤派生：

~~~text
p_world = observation.state[0:3]
T_camera_world = inverse(agentview_T_world_camera)
p_camera = T_camera_world @ [p_world, 1]
[u', v', w'] = K @ p_camera[:3]
u = u' / w'
v = v' / w'
d = p_camera[2]
~~~

无效点包括：相机后方、NaN、越界点；这些点通过 valid mask 排除。V1 暂时不建模 EEF 被遮挡时 depth map 看到前景物体的情况。

future frame 当前定义为 t+H，并做 episode boundary 检查。UVD 的 K 个点从真实 dense EEF trajectory frame 均匀采样，保留起点和终点。

## 7. Training objective

当前正式配置为：

~~~text
lambda_action         = 1.00
lambda_depth_current  = 0.14
lambda_depth_future   = 0.15
lambda_uvd            = 0.62
lambda_geometry       = 0.00  # disabled; endpoint visibility is not reliable
~~~

总 loss：

$$
L = 1.0L_{action}
+0.14L_{D_c}
+0.15L_{D_f}
+0.62L_{UVD}
$$

其中：

- L_action：baseline DiT action loss；
- L_Dc / L_Df：valid-pixel masked Smooth-L1 depth loss；
- L_UVD：valid-point masked Smooth-L1 UVD regression；
- geometry_loss：仅作为零值兼容字段，不进入 total loss。

前三个权重来自四个子集各 8 个样本的初始化尺度统计；geometry 的旧校准值不再使用。新的 active auxiliary/action 比例需要在修订后的短 pilot 中重新记录。

## 8. 已经验证的内容

- 四个 LIBERO rerender dataset 可以正常读取；
- 真实样本得到 action (8,7)、depth (1,224,224)、UVD (4,3)；
- projection、resize、mask、sampling 单测通过；
- group-causal mask 和 variable-K reasoner 单测通过；
- decoder 输出 shape/正深度单测通过；
- focused tests：13/13 通过；
- 真实 Qwen3.5 forward/backward 中，query、depth decoder、action branch 都有非零梯度；
- 真实 inference 输出 (1,8,7)，且 hook 验证没有调用 numeric depth/UVD decoder；
- config、launcher、baseline 参数对齐审计通过。

这些证据证明代码路径和机制接口可运行，但还不证明方法提升了 LIBERO action success。

## 9. 必须验证的内容

### P0：训练和数据正确性

1. 在没有外部 GPU 显存占用的环境跑正式短 pilot，至少记录前 1k–10k steps 的：
   - action loss；
   - 两个 depth loss；
   - UVD loss；
   - geometry compatibility metric 是否保持为零；
   - total loss；
   - gradient norm 和显存。
2. 确认 t:t+H action chunk 与 future observation t+H 在 timestamp/frame semantics 上确实对应“执行完整 chunk 后”。
3. 检查 episode 边界、有效 UVD 比例和四个子集的 valid-mask 分布。
4. 用训练后的 checkpoint 检查 action loss 是否下降、auxiliary loss 是否下降，以及是否出现某个 auxiliary 分支主导梯度。

### P1：方法机制

必须至少做以下 ablation：

| Ablation | 要回答的问题 |
|---|---|
| baseline [V,L] | CoT 是否带来真实增益 |
| [V,L] + auxiliary decoder，但 action condition 不加 UVD | auxiliary supervision 本身是否有效 |
| 完整 V1 [V,L,UVD] | UVD embedding 是否被 action expert 使用 |
| 去掉 future depth | future geometry 是否有贡献 |
| group-causal 改 full query attention | causal ordering 是否重要 |
| UVD token mask/condition ablation | action gain 是否来自 UVD，而非额外 token/参数 |

### P1：评价指标

- action：LIBERO success rate、按 task/group 汇总的 success；
- depth：valid-pixel MAE/RMSE，current 与 future 分开；
- UVD：normalized coordinate error、pixel error、camera-z depth error；
- 训练效率：显存、step time、参数量和推理开销。

### P2：鲁棒性和泛化

- 不同 H 与不同 K；
- 不同 FPS 或 action chunk duration；
- 不同 resize/crop；
- 多视角数量变化；
- EEF 遮挡和 visibility-aware supervision；
- 是否需要引入 Qwen vision encoder 的多尺度 feature / DPT。

## 10. 当前 Known / Assumptions / Unknowns

### Known

- 当前代码、数据读取、query mask、decoder、loss 和 train/inference 分支已经接通；
- UVD 是主视角 EEF 的 camera-coordinate (u,v,z)，不是欧氏距离；
- action 系数为显式 1.0，辅助总权重初始化约为 action 的 10%；
- 当前 V1 使用最终 Qwen hidden，不使用 vision encoder 多尺度 feature。

### Assumptions

- agentview 是主视角；
- observation.state[0:3] 是 EEF world position；
- 当前 LIBERO export 中 future frame 可以用 t+H 表示；
- 暂不处理真实 EEF visibility/occlusion。

### Unknowns

- full training 后 UVD embedding 是否真正改善 action；
- 最终 loss 权重是否需要根据 learning curve 调整；
- group-causal ordering 是否优于 full attention；
- future depth 是否带来独立增益；
- 不同 dataset/FPS/chunk 下的时间配对和泛化；
- 遮挡处理是否成为主要误差来源。

## 11. 推荐下一步实验顺序

1. 先跑完整 V1 与 baseline 的同预算短 pilot，确认 loss/梯度/显存稳定；
2. 做 baseline、aux-only、full V1 三个最小对照；
3. 再做去 future depth、去 geometry、full-attention 三个机制 ablation；
4. 最后才扩展 DPT、多尺度 feature、visibility mask 和跨 FPS 泛化。

## 12. 与 LIBERO baseline 的参数对齐

当前配置与
`examples/modelExtensions/CoT/configs/qwen35_gr00t_libero_baseline.yaml`
逐字段比较后的结论如下。

### 保持一致的训练参数

- Qwen3.5-4B backbone 路径；
- DiT-B action expert、action hidden size、cross-attention dimension；
- action horizon `H=8`；
- per-device batch size `16`；
- base/Qwen/action learning rates：`3e-5 / 1e-5 / 1e-4`；
- warmup `5000`、max steps 默认 `60000`；
- cosine scheduler、minimum LR、AdamW betas/epsilon/weight decay；
- gradient clipping、gradient checkpointing、gradient accumulation；
- baseline 中的 `vlm_data` 配置和 `sequential_step_sampling: false`。

### 有意改变的参数

| 参数 | baseline | V1 | 原因 |
|---|---|---|---|
| `run_id` | `qwen35_gr00t_libero_baseline` | `qwen3_gr00t_libero_CoT_v1` | 区分实验输出 |
| `run_root_dir` | `/root/data/yxz/outputs` | `/root/data/yxz/outpus` | 遵循本实验指定路径 |
| `framework.name` | `QwenGR00T` | `QwenGR00TCoT` | 注册新增 CoT framework |
| `dataset_py` | `lerobot_datasets` | `cot_lerobot_datasets` | 返回 depth/UVD targets |
| `data_root_dir` | `libero_replay` | `libero_rerender` | 使用 RGB-D + camera rerender 数据 |
| `data_mix` | `libero_all` | `libero_cot_all` | 使用四个 CoT adapter 数据集 |
| `video_backend` | `torchvision_av` | `torchvision_av` | 已切换到可用的 `CoT_linearATT` 环境 |

其余新增字段均属于 V1 geometry 配置：query 数量、group-causal reasoner、UVD 点数、depth decoder 和 loss weights，不改变 baseline action expert 的结构或基础优化超参。

launcher 默认仍使用 baseline 的 `60000` steps 和 `8` processes；需要做 10k pilot 时通过环境变量覆盖：

~~~bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \\
NUM_PROCESSES=8 \\
MAX_TRAIN_STEPS=10000 \\
bash examples/modelExtensions/CoT/scripts/run_qwen35_gr00t_libero_CoT_v1.sh
~~~
