# Geometric CoT Adapter：后续插件化方向

> 状态：Idea Note，暂不进入当前主线实现
> 日期：2026-07-27

## 1. 核心想法

从一个已经训练好的 VLM + Action Expert baseline 出发，额外安装一个
Geometric CoT Adapter。Adapter 从 VLM hidden states 中提取并组织几何推理
latent，同时承担两种功能：

1. 训练或诊断时显式解码当前深度、未来深度和 EEF UVD 轨迹；
2. 动作推理时不要求显式解码，只把几何 latent 作为额外条件送入 Action Expert。

候选方法主张：

> A lightweight geometric reasoning adapter augments pretrained,
> token-conditioned VLA policies with current-state perception,
> future-state prediction, and temporally structured 3D trajectory reasoning,
> while retaining optional explicit geometry decoding without mandatory decoding
> overhead during action inference.

## 2. 概念架构

```text
Pretrained VLM
  │
  ├── H_VL [B, N, C_vlm]
  └── spatial image features / metadata
             │
             ▼
    Geometric CoT Adapter
    ├── input projection: C_vlm → C_adapter
    ├── current-depth queries
    ├── future-depth queries
    ├── time-aware UVD queries
    └── group-causal query reasoner
             │
       ┌─────┴─────────────┐
       ▼                   ▼
 Optional Decoders     Action Bridge
 ├── current depth         │
 ├── future depth          ▼
 └── UVD trajectory   Action Expert
```

默认推理路径只运行 Adapter latent 和 Action Bridge；显式 depth/UVD decoding
按需开启，用于可视化、诊断或提供可解释输出。

## 3. 插件接口

为了避免绑定某个特定 VLM hidden size，Adapter 使用固定内部维度：

```text
VLMFeatureAdapter:     C_vlm → C_adapter
GeometricReasoner:     H_VL → Z_current, Z_future, Z_uvd
ActionConditionBridge: C_adapter → C_action
```

更换 VLM 时主要替换输入投影和空间 token 提取接口；更换 Action Expert 时主要
替换 ActionConditionBridge。

可支持的模型范围应谨慎描述为：

> 支持能够输出 token features 的 VLM，以及能够接收 token condition 或提供
> cross-attention hook 的 Action Expert。

在没有跨架构实验证据前，不宣称适用于任意 VLM + Action Expert。

## 4. Action Bridge 候选

### 4.1 Token concatenation

```text
condition = [H_VL, P(Z_uvd)]
```

优点是实现简单、接近当前 CoT V1；风险是 Action Expert 可能忽略 UVD，而且新增
token 会改变原有 cross-attention 的归一化。

### 4.2 Zero-initialized gated cross-attention（推荐）

```text
delta_h = tanh(g) * CrossAttn(h_action, P(Z_uvd))
h_action' = h_action + delta_h
```

令 `g = 0` 初始化，使安装 Adapter 后的初始模型严格退化为原 baseline，再逐步
学习使用几何条件。该方案更适合“插件增强且不破坏预训练策略”的方法主张，但需要
Action Expert 暴露插入位置。

## 5. 建议训练方式

```text
Stage 0: 加载训练完成的 baseline checkpoint

Stage 1:
  冻结 VLM 和原 Action Expert
  训练 Geometric CoT Adapter、Decoders 和 Action Bridge

Stage 2:
  保持 VLM 冻结
  低学习率解冻 Action Expert，或仅训练 Action Expert LoRA

Stage 3（可选）:
  极低学习率联合微调
```

如果安装 Adapter 后直接全量微调 VLM 和 Action Expert，可以称为模块化扩展，
但“parameter-efficient plug-in”的主张会明显变弱。

## 6. 必要实验对照

从同一个 baseline checkpoint 开始，后续训练步数、数据顺序和 scheduler 保持一致：

| 实验 | 后续训练 | 作用 |
|---|---|---|
| A | 原 baseline 继续训练 | 排除额外训练步数收益 |
| B | Adapter + action loss | 排除单纯参数量或结构收益 |
| C | Adapter + action/depth/UVD losses | 检验 Geometric CoT 监督 |
| D（可选） | Geometry decoding，但不接入 Action | 检验辅助表征学习 |

只有当 `C > A/B/D` 时，才能较有力地说明可被 Action Expert 使用的几何 CoT
latent 带来了额外收益。

还需要通过正常、置零和 batch 间打乱 `Z_uvd` 的配对推理，验证 Action Expert
确实依赖 Adapter，而不是绕过几何条件。

## 7. 与当前 CoT V1 的关系

当前实现已经具备部分插件结构：

- 外置 `GeometricQueryReasoner`；
- 共享 depth decoder；
- UVD numeric head；
- UVD latent 作为 Action Expert condition。

要成为真正通用的 Adapter，仍需解决：

- query hidden dimension 与 Qwen hidden size 强耦合；
- depth spatial token 提取依赖 Qwen 实现；
- Action condition 依赖 GR00T 的接口；
- 当前 full-width 两层 query reasoner 参数量较大；
- 缺少零初始化、baseline-preserving 的 Action Bridge。

## 8. Research State

### Known

- 外置 query reasoner 可以独立于 VLM 主干工作，并支持显式 geometry decoding。
- 当前 latent UVD 可以作为 Action Expert condition。
- 当前直接拼接条件存在被 Action Expert 绕过的可能。

### Assumptions

- 从强 baseline 初始化可以降低重新学习基础视觉动作能力的成本。
- group-causal geometry latent 能提供普通 VLM hidden states 中缺少的预测性空间结构。
- gated Action Bridge 能在保留 baseline 初始行为的同时逐步学习几何条件。

### Unknowns

- 冻结 VLM 和原 Action Expert 时，仅训练 Adapter 是否足以提升闭环成功率。
- 性能收益来自几何监督、额外参数还是额外训练步数。
- Adapter 在不同 VLM hidden spaces 和不同 Action Expert 接口间的迁移程度。
- 显式 depth/UVD 预测质量和动作性能是否具有稳定相关性。

### Next decisive experiment

当前主线完成后，从同一个 baseline checkpoint 运行 A/B/C 三组严格配对实验，
优先回答“几何监督产生的 Adapter latent 是否被 Action Expert 使用并带来额外收益”。
