# CoT V2 条件输入、解码器与诊断设计

## 目标与研究优先级

CoT V2 优先追求结构清晰且可验证的几何推理链，而不是把所有几何特征直接堆给策略。UVD 轨迹 token 是唯一直接提供给 Action Expert 的显式几何 CoT 表征。当前/未来稠密深度只作为辅助监督，用于塑造 Qwen 内部的 token 推理，不能成为动作策略的直接旁路。

## 推理路径与梯度路径

Qwen 中的序列顺序为：

```text
[V, L] -> D_current -> D_future -> UVD
```

Action Expert 的条件固定为：

```text
[V, L, UVD_final]
```

`D_current_final` 和 `D_future_final` 不直接拼接到 Action Expert。UVD token 可以在 Qwen 内读取前面的深度 token，因此深度信息只有先被压缩进 UVD 推理状态，才能通过显式几何 CoT 路径影响动作。V/L 仍保留直接语义与视觉路径，所以这是“软几何瓶颈”，不是硬信息瓶颈。

第一轮工程验证不要求 UVD inference intervention 或训练消融。UVD zero/shuffle、latent-only 等作为后续科学证据实验保留。

## Token 布局

Current depth 和 future depth 各使用 8 个独立初始化的 learnable tokens：

```text
D_current: [B, 8, H]
D_future:  [B, 8, H]
```

V2 主配置固定为 8。这样在从 V1 独立 query module 改成直接进入 Qwen 时，不同时改变 token 数量，便于归因。第一轮实现不自动进行 1/4/8 token sweep。

UVD token 数量继续由 benchmark/action horizon 决定，并采用 time-major 顺序。每个 UVD token 包含归一化时间 embedding；双臂数据还包含 hand embedding。同一时刻的左右手 token 可以互相 attention，不同时间保持时间因果顺序。

## Depth Readout

旧实现把 8 个最终 depth-token hidden states 直接求平均。新实现改成一套 current/future 共享的 attention pooling：

\[
s_i=w^T\operatorname{LN}(h_i),\qquad
\alpha_i=\operatorname{softmax}(s_i),\qquad
q=\sum_i\alpha_i h_i.
\]

Current 和 future 使用相同的 LayerNorm 与 scoring projection。Pooling 返回 summary `[B,H]` 和权重 `[B,8]`。该模块不包含 current/future 专属参数，也不新增 token 间 reasoning。

Pooling 后的 summary 用于 FiLM 调制现有共享 ConvStack：

```text
Qwen 最后一层主视角 image patch features F
                 +
共享 attention pooling 得到的 depth summary q
                 |
         Shared FiLM ConvStack
                 |
          224 x 224 metric depth
```

Current/future 完整共享 pooling 和 depth decoder 权重，两者唯一差异来自 Qwen 产生的 depth-token states。正常动作推理路径不执行 depth decoder。

## UVD 数值读出

保留现有共享 pointwise 两层 MLP：

```text
一个 final UVD token -> 共享 MLP -> 一个绝对 (u,v,d) 点
```

MLP 不允许不同 UVD token 互相通信，时间推理必须在 Qwen 内完成。U/V 通过 sigmoid 约束到 `[0,1]`，depth 通过 softplus 保持为正。监督由所有点的 absolute Smooth L1 和同手相邻点增量 Smooth L1 构成。不增加自回归、时序卷积、残差积分或 Transformer decoder。

## Loss 时序

从 step 0 开始使用固定权重：

\[
L=L_a+0.14L_{dc}+0.15L_{df}
  +0.62L_{u,abs}+0.062L_{u,rel}.
\]

原有 learning-rate warmup 已经会缩放训练早期的实际参数更新，因此不额外加入 auxiliary ramp-up 或 decay。只有 checkpoint gradient probe 显示可重复的阶段性梯度冲突时，才重新考虑独立 loss schedule。

## 轻量在线 Diagnostics

### Attention pooling 利用率

在现有 diagnostics interval 上，分别记录 current/future：

- 自然对数原始 attention entropy，以及 `[0,1]` 范围的归一化熵 `H/log(8)`；
- 最大 attention weight；
- 使用原始熵计算的 effective token count `exp(H)`，范围 `[1,8]`；
- depth tokens 的平均非对角 pairwise cosine；
- depth-token covariance effective rank。

Effective rank 定义为 `exp(H(p))`，其中 `p` 是中心化 token covariance 的 FP32 归一化特征值分布。所有方差均为零时记录 rank 0，而不是 NaN。

### Decoder reliance

在现有固定 diagnostic samples 上，不更新优化器，解码四种输入：

1. 正常 current/future pooled summaries；
2. zero summary；
3. current/future summary 互换；
4. batch size 大于 1 时，跨样本循环打乱 summary。

分别记录相对正常预测的变化，以及每种 intervention 的 target error。如果 future depth 在 zero/swap/shuffle 后几乎不变，说明 decoder 可能主要依赖 image patches，而没有真正使用 depth tokens。

Decoder-reliance 只在配置的 geometry diagnostics interval 执行，不在普通训练 step 中运行，也不增加 backward。

### UVD 表征利用率

在同一 diagnostics interval 记录：

- 每个时间位置的 U/V pixel MAE 和 depth MAE；
- 每个时间位置的 valid ratio；
- UVD tokens 的平均非对角 pairwise cosine；
- UVD-token covariance effective rank；
- 已有的起点、终点、轨迹长度和相邻运动指标。

双臂数据保持 time-major 顺序，并只比较同一只手的轨迹。

### Gradient clipping

复用 `clip_grad_norm_` 已经返回的梯度模长，记录：

- `train/grad_norm_pre_clip`；
- `train/grad_clip_threshold`；
- `train/grad_clip_triggered`；
- `train/grad_clip_scale`。

这四个字段不安装额外 hook，也不增加 backward。

## 离线 Gradient Probe

离线 probe 以独立设计文档为准：

`design/v2/2026-07-30-offline-gradient-probe-design_zh.md`。

它在固定样本和选定共享参数上测量各 loss 的 gradient norm 以及与 action gradient 的 cosine。在线 representation/decoder diagnostics 和离线 gradient probe 是互补但独立的代码路径。

## 配置与 Checkpoint 兼容

Attention pool 是 V2 新增的可训练模块，新 V2 checkpoint 必须保存其参数。旧 mean-pooling V2 checkpoint 严格加载到新结构时必须明确报 incompatible checkpoint，禁止静默使用随机初始化的 pool 参数。V1 和 baseline checkpoint 行为不变。

所有新 diagnostics 继续受现有 `trainer.test_diagnostics` 总开关控制。子功能使用显式布尔开关，配置缺失时默认关闭，因此普通训练不承担 intervention 计算开销。常规 loss 标量日志保持不变。

## 验证要求

测试必须证明：

1. 共享 attention pooling 返回正确 shape，且权重归一化；
2. current/future 调用使用同一套 pool 参数；
3. 非均匀 token states 可以获得非均匀权重和有效梯度；
4. zero/swap/shuffle intervention 使用预期 summary，且不修改模型参数；
5. entropy、effective token count、cosine 和 effective rank 在正常与退化 toy tensor 上都为有限值；
6. per-time UVD metrics 保持单臂和双臂 time-major 顺序；
7. diagnostics 关闭时不会执行额外 decoder intervention；
8. clipping metrics 与手算的 clipped/unclipped 情况一致；
9. V1 和 baseline 的聚焦回归继续通过。

真实 V2 GPU smoke 需要验证 forward/backward shape 和一次 diagnostic pass。它只能证明实现行为，不能证明策略提升或科学假设成立。

## 暂缓的科学证据

第一轮实现不包含：

- UVD condition zero/shuffle benchmark evaluation；
- 关闭 UVD supervision 的 latent-only V2 训练；
- depth-supervision ablation；
- depth-token count sweep；
- dynamic auxiliary-loss schedule；
- DPT、多尺度、时序或自回归 decoder。
