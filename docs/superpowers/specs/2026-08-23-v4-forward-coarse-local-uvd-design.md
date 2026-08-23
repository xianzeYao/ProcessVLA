# QwenGR00TCoT V4 正向 Coarse→Local UVD 设计

## 状态与结论

本设计取代当前 V2 上的 reverse full-UVD 实验。V4 基于已经验证过的
`QwenGR00TCoTV2` 的 q0 action-query ablation + depth-conditioned 架构，但把轨迹推理明确拆成两层：

```text
VLM / 大脑：coarse UVD，与 action/local 点数相同，stride 2
        ↓
局部规划 / 小脑：local UVD，与 action horizon 对齐，stride 1
        ↓
动作头：沿用各 benchmark 的 action horizon
```

两条 UVD 都从当前时刻向未来正向预测。LIBERO 使用 action/local/coarse points 8；RoboCasa
沿用现有 action horizon 16，并使用 local/coarse points 16。Coarse 的 stride 为 2，因此
物理规划窗口恒为 action horizon 的两倍：LIBERO 16 steps，RoboCasa 32 steps。这里的
coarse “完整”是指覆盖固定的两倍 action window，而不是从当前帧覆盖到整个 episode 结束。
旧的 128 点、stride-4、
终点到当前帧的反向 full-UVD 方案只保留在 Git 历史中，不再作为活动配置或 V2 功能。

## 目标

- 让 coarse UVD 表达比当前 action chunk 更长的任务级移动意图。
- 让 local UVD 在 coarse 计划之后逐帧细化各 benchmark 的完整 action horizon。
- 保持训练和推理都是同一条单次 causal forward，不引入 teacher forcing 的暴露偏差。
- 把新架构、数据字段、配置、训练入口和测试放在 V4，恢复 V2 的原始行为和 checkpoint
  contract。
- 用固定、全监督、同点数的 coarse/local slots 避免 episode 尾部出现仅少量有效点、
  其余 slot 无监督。

## 非目标

- V4 第一版不使用 flow matching 生成 coarse UVD。
- 不反向预测轨迹。
- 不预测直到 episode terminal 的变长全轨迹。
- 不在第一版加入 coarse/local consistency loss、object anchor 或额外 object decoder。
- 不从旧 reverse-full run 续训，也不把旧 V2 reverse-full checkpoint 当作 V4 checkpoint。

## 精确的数据定义

设样本当前帧为 `t`，episode 最后一帧为 `T`，各 benchmark 参数为：

| Benchmark | Hands | Action horizon | Local points/stride | Coarse points/stride |
|---|---:|---:|---:|---:|
| LIBERO | 1 | 8 | 8 / 1 | 8 / 2 |
| RoboCasa Fourier | 2 | 16 | 16 / 1 | 16 / 2 |

动作监督保持现有 benchmark 定义。future depth 的目标帧为
`min(t + action_horizon, T)`。

### Local UVD

Local UVD 固定为 `action_horizon` 个逐帧 future-state waypoint。令
`H = action_horizon`：

```text
local_offset[j] = j + 1                         j = 0..H-1
local_index[j]  = min(t + local_offset[j], T)
```

LIBERO episode 中部为 `[t+1, t+2, ..., t+8]`，RoboCasa 为
`[t+1, t+2, ..., t+16]`。它不包含当前帧 `t`，从而与各自未来动作位置一一对应。

数据字段沿用现有 local 名字：

- `uvd`
- `uvd_valid_mask`
- `uvd_out_of_frame_mask`
- `uvd_boundary_clamp_mask`
- `uvd_frame_indices`
- `uvd_time`
- `uvd_endpoint_indices`

`uvd_time[j] = (j + 1) / H`，表示固定 local 规划槽位，而不是被 episode terminal
截断后的实际时间。

### Coarse UVD

Coarse UVD 固定为 `H` 个 stride-2 future-state waypoint：

```text
coarse_offset[j] = 2 * (j + 1)                  j = 0..H-1
coarse_index[j]  = min(t + coarse_offset[j], T)
```

LIBERO episode 中部为 `[t+2, t+4, ..., t+16]`，RoboCasa 为
`[t+2, t+4, ..., t+32]`，均覆盖 local/action 窗口及同等长度的后续意图。
数据字段使用独立、无歧义的名字：

- `uvd_coarse`
- `uvd_coarse_valid_mask`
- `uvd_coarse_out_of_frame_mask`
- `uvd_coarse_boundary_clamp_mask`
- `uvd_coarse_frame_indices`
- `uvd_coarse_time`
- `uvd_coarse_endpoint_indices`

`uvd_coarse_time[j] = (j + 1) / H`。

### Episode 尾部 terminal-repeat

超过 `T` 的请求索引全部 clamp 到 `T`，因此固定的 local/coarse 位置始终有真实帧目标。
例如 LIBERO 的 `T=147, t=146` 时：

```text
local indices  = [147, 147, 147, 147, 147, 147, 147, 147]
coarse indices = [147, 147, 147, 147, 147, 147, 147, 147]
```

这些重复位置继续参与监督；不能因为 horizon 越界而把它们标成 padding/invalid。这样模型
显式学习“到达终点后保持在终点”。但是 terminal 帧本身若因为投影到相机后无效、深度非正、
NaN 或出画面而无效，其重复位置仍继承该几何有效性，不能伪造有效标签。

`uvd_frame_indices` 和 `uvd_coarse_frame_indices` 保存 clamp 后的真实帧索引；固定的 time
字段仍保存名义槽位时间。二者分别用于检查数据来源和构造 query time embedding。

## V4 模型结构

### Token 顺序与 causal 依赖

LIBERO 单手配置的完整序列为：

```text
[native image/language]
[depth_current × 8]
[depth_future × 8]
[uvd_coarse × 8]
[uvd_local × 8]
```

RoboCasa 双手采用 time-major 顺序，完整序列为：

```text
[native image/language]
[depth_current × 8]
[depth_future × 8]
[uvd_coarse × (16 times × 2 hands)]
[uvd_local × (16 times × 2 hands)]
```

几何 token 是逐 slot 可独立学习的 query tensor，并加入对应尺度的固定时间 embedding。
Coarse 和 local 使用两套独立的 per-slot queries、time-embedding MLP 和回归 head，避免两个
时间尺度在同一个 query 族或 decoder 中互相抢占表示能力。这里的 q0 仅表示 action query
token 数为 0，不表示 coarse/local 没有 learnable query。

attention 保持 V2 的 group-causal 规则：

- native token 不读取任何几何 token；
- depth token 只读取 native 和自身允许的 depth group；
- 第 `j` 个 coarse token 读取 native、depth 以及不晚于 `j` 的 coarse token；
- 第 `j` 个 local token 读取全部 coarse token，以及不晚于 `j` 的 local token；
- local 对 coarse 的依赖由序列顺序和 mask 保证，不依赖额外的两次 forward。

单手时每个时刻一个 token；多手扩展仍采用 time-major，同一时刻的不同 hand token 可互读。

### UVD 解码

V4 使用两个直接回归 head：

```text
coarse hidden -> coarse UVD head -> sigmoid(u,v), softplus(d)
local hidden  -> local UVD head  -> sigmoid(u,v), softplus(d)
```

第一版不采用 ACoT EAR 的 flow head。固定 8/8 或 16/16 query 已经给出确定长度，query
的数量就是预测点数；无需 EOS、长度分类器或变长生成。

### Action condition

V4 保持 `num_target_vision_tokens = 0` 和 depth-conditioned 路径。Action model 的 condition
按以下顺序拼接：

```text
[native, depth_current, depth_future, uvd_coarse, uvd_local]
```

因此 action hidden condition 能读取 coarse 计划和 local 细化结果。训练 action head 时使用
模型自己在同一次 causal forward 中得到的 geometry hidden states，不把 ground-truth UVD
坐标重新编码后喂给 action head。

## 损失与稳定性

每组 UVD 都采用 masked absolute regression 加相邻位移 regression：

```text
L_local  = L_local_abs  + 0.1 * L_local_relative
L_coarse = L_coarse_abs + 0.1 * L_coarse_relative

L_total = 1.0  * L_action
        + 0.14 * L_depth_current
        + 0.15 * L_depth_future
        + 0.62 * L_local
        + 0.20 * L_coarse
```

每项损失只按自己的有效坐标/segment 做 mean normalization，避免手数、horizon 或有效点数
改变时隐式改变 loss 权重。terminal-repeat slots 是有意监督，因此会进入 mean；投影无效的
slots 仍由 validity mask 排除。

第一版不加显式 cross-scale consistency loss。LIBERO 的 `+2,+4,+6,+8` 以及 RoboCasa 的
`+2,+4,...,+16` 重合物理帧本来就分别有 ground-truth 监督，额外强绑两个 head 会让问题
归因更困难。先在诊断中记录这些重合点的 coarse/local prediction gap；只有观察到明显
跨尺度冲突时再决定是否加入 consistency 项。

稳定性措施为：固定长度、terminal-repeat 全监督、coarse/local 独立参数、各 loss 独立
归一化、gradient clipping 1.0，并保留现有 module-gradient 和 action-intervention 诊断。
不引入 ACoT 式 ground-truth coarse teacher forcing，因为那需要不同的训练/推理路径并造成
exposure gap；若直接回归版本学不稳，再把 flow 或 scheduled sampling 作为独立后续实验。

## 输出、指标与干预诊断

V4 的 local 输出继续使用 `uvd`，以复用现有可视化和 local 指标；新增 coarse 输出
`uvd_coarse`。训练 loss key 为：

- local：`uvd_loss`、`uvd_absolute_loss`、`uvd_relative_loss`
- coarse：`uvd_coarse_loss`、`uvd_coarse_absolute_loss`、
  `uvd_coarse_relative_loss`

时间指标要分别记录 local/coarse 的 XY MAE、depth MAE、首点、末点和逐槽位误差。另记录：

- coarse/local 在各 benchmark action window 内重合偶数帧上的 prediction gap；
- terminal-repeat 样本比例，避免末点总体指标被重复 terminal 样本误读；
- coarse、local token 各自的 cosine/effective-rank；
- action intervention 的 `coarse_only`、`local_only`、`coarse+local`、depth 和全部 geometry
  shuffle/zero 变体。

诊断只用于度量依赖关系，不改变训练图或主 loss。

## LIBERO 与 RoboCasa 数据边界

V4 的 horizon sampler、terminal-repeat 和 target packing 逻辑共用，但两套 episode geometry
loader 保持现有定义：

- LIBERO 继续从单臂 EEF world XYZ 投影到 agentview UVD；
- RoboCasa 继续通过 `select_robocasa_uvd_world_columns` 读取左右手
  thumb-index pinch world position，投影后应用 `dial_content_region_mask`；
- RoboCasa 保留现有 `action_dim=29`、`state_dim=58`、absolute action mode、16-step action
  chunk、Fourier mixture 和双手 time-major 顺序；
- terminal-repeat 在投影前对 frame index clamp，因此左右手各自继承 terminal 帧的坐标与
  validity，不能把一只无效手的 mask 复制成另一只手。

## 版本隔离和文件边界

V2 恢复到 reverse-full 四个实现 commit 之前的行为：layout 中没有 full/coarse slots，framework
不解析 full-UVD 配置、不输出 full-UVD loss，原 q0-depth YAML 和 checkpoint key 保持不变。
V3 继续继承干净的 V2 行为，不接入 V4 coarse 路径。

V4 使用独立入口：

- `starVLA/dataloader/gr00t_lerobot/cot_geometry_v4.py`
- `starVLA/dataloader/cot_v4_lerobot_datasets.py`
- `starVLA/dataloader/robocasa_v4_lerobot_datasets.py`
- `starVLA/model/modules/geometric_cot_v4.py`
- `starVLA/model/framework/VLM4A/QwenGR00TCoTV4.py`
- `starVLA/training/train_starvla_cot_v4.py`
- `examples/modelExtensions/CoT/configs/qwen35_gr00t_libero_CoT_v4_q0_depthcond_coarse8_local8.yaml`
- `examples/modelExtensions/CoT/configs/qwen35_gr00t_robocasa_fourier_CoT_v4_q0_depthcond_coarse16_local16.yaml`
- `examples/modelExtensions/CoT/scripts/run_qwen35_gr00t_CoT_v4_common.sh`
- `examples/modelExtensions/CoT/scripts/run_qwen35_gr00t_libero_CoT_v4.sh`
- `examples/modelExtensions/CoT/scripts/run_qwen35_gr00t_robocasa_fourier_CoT_v4.sh`

V4 可以复用与版本无关的 projection、depth decoder、loss 和 Qwen geometry-forward utility，
但不通过在 V2 layout 上增加 optional coarse 字段实现。V4 framework 注册名为
`QwenGR00TCoTV4`。LIBERO dataset config 使用 `cot_v4_lerobot_datasets`，RoboCasa 使用
`robocasa_v4_lerobot_datasets`，避免修改现有 V2 dataset factory 的选类行为。

旧的 `qwen35_gr00t_libero_CoT_v2_q0_depthcond_reverse_full_uvd_s4.yaml` 从活动配置中删除；它及
旧实现都可通过 commit `868aad3`、`d2c9ead`、`1cf29f3`、`3711c93` 恢复。

V4 首次实验从同一 Qwen3.5 base 和随机初始化的新 geometry/action modules 开始，不隐式加载
V2 或 reverse-full checkpoint。V4 state dict 使用 `coarse_*`/`local_*` 独立 key，错误地按
V2 严格加载时必须失败，而不是静默丢弃新参数。

## 配置基线

LIBERO V4 配置继承现有 V2 q0-depth 的训练超参，只改版本和几何字段：

```yaml
framework:
  name: QwenGR00TCoTV4
  action_model:
    action_horizon: 8
    num_target_vision_tokens: 0
  geometry:
    depth_query_count: 8
    include_depth_in_action_condition: true
    local_uvd_num_points: 8
    coarse_uvd_num_points: 8
    coarse_uvd_stride: 2
    lambda_uvd: 0.62
    lambda_uvd_relative: 0.1
    lambda_uvd_coarse: 0.2
    lambda_uvd_coarse_relative: 0.1

datasets:
  vla_data:
    dataset_py: cot_v4_lerobot_datasets
    cot_geometry:
      action_horizon: 8
      local_uvd_num_points: 8
      coarse_uvd_num_points: 8
      coarse_uvd_stride: 2
      terminal_repeat: true
```

V4 对这些结构字段做 fail-fast 校验：首个配置必须是 action horizon 8、local 8、coarse 8、
stride 2 和 `terminal_repeat: true`，model/data 两侧点数不一致时启动即报错。

RoboCasa V4 配置继承现有
`qwen35_gr00t_robocasa_fourier_CoT_v2_q0_depthcond.yaml`，保留 100k steps、学习率、数据
mixture 和 action/state 维度，仅改为 `QwenGR00TCoTV4`、V4 dataset factory、local 16、
coarse 16、stride 2 和 terminal-repeat。RoboCasa config 对 action/local horizon 16、双手 2、
coarse 16 和 stride 2 做同样的 model/data 一致性校验。

## 测试与启动门槛

实现采用独立 V4 测试，至少覆盖：

1. LIBERO 中部精确得到 local `[t+1..t+8]`、coarse `[t+2,t+4,..,t+16]`；RoboCasa
   得到 local `[t+1..t+16]`、coarse `[t+2,t+4,..,t+32]`；
2. 两个 benchmark 的 episode 尾部所有越界 slot 重复 terminal 帧并保留监督，几何无效性
   正确传播；
3. local/coarse 的 target shape、time、frame index 和 time-major packing；
4. token 数量和顺序：LIBERO 为 depth 8+8、coarse 8、local 8；RoboCasa 为
   depth 8+8、coarse 16×2 hands、local 16×2 hands；
5. attention 方向为 coarse→local，反向读取被禁止；
6. separate per-slot query、time embedding、head 和 loss 梯度都非零；
7. action condition 在 q0-depth 下包含 depth、coarse 和 local；
8. total-loss 权重及每组 mean normalization 正确；
9. RoboCasa 左右手的 time-major packing、pinch 字段选择和 dial validity mask 保持正确；
10. V2/V3 regression tests 证明旧配置、layout、outputs 和 checkpoint contract 已恢复；
11. 两套 V4 YAML dry-run 都能解析到独立 trainer、dataset 和 framework。

每个 benchmark 启动正式训练前都运行短 smoke/timing gate：相同 GPU 数、batch 和诊断开关
下比较对应 V2 q0-depth 与 V4 的稳定段 `model_ms` 和峰值显存。LIBERO 相比 V2 增加 12 个
geometry token；RoboCasa 从 depth 16 + local 12 变为 depth 16 + coarse 32 + local 32，
因此必须分别测量，不能套用 LIBERO 比例。若任一 benchmark 的 median `model_ms` 超过对应
V2 的 1.5 倍、出现周期性 OOM/NaN 或显存持续增长，则停止该正式训练并先定位数据等待、
attention backend、诊断 cadence 或 allocator 行为。两套实现和 smoke 都在本范围内；当前明确
授权的正式长训是 LIBERO，因此通过 gate 后先在 tmux 启动 LIBERO。RoboCasa 保持可直接启动
状态，除非用户进一步指定，否则不擅自占用另一组 GPU 启动 100k 长训。

## 验收标准

- V2 和 V3 的现有非 reverse-full 测试通过，V2 源码中不存在活动 full/coarse 分支。
- 所有 V4 单元、集成和 YAML dry-run 测试通过。
- LIBERO 和 RoboCasa 各一个真实 batch 验证 local/coarse 索引、validity、loss 和预测 shape。
- smoke run 无 NaN/OOM，coarse/local head 与 token 参数都有非零梯度。
- 两套 timing gate 有记录且不触发 1.5× 停止条件。
- 代码与配置提交后，才启动 tmux LIBERO 正式训练并回报 session、run directory、log 和初始
  指标；RoboCasa 回报已通过的 smoke/timing 结果及正式启动命令。
