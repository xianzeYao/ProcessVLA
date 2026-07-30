# CoT V2 离线 Gradient Probe 设计

## 目标

测量每个 CoT V2 objective 如何作用于共享表征参数，使 loss 权重依据梯度模长和方向确定，而不是依据 raw loss 数值。Probe 不得改变正常训练的 backward 路径，也不能拖慢每个训练 step。

## 方案决策

采用混合 diagnostics：

1. 在线保留低成本的 raw/weighted loss、合成模块 gradient norm 和预测指标；
2. 新增独立离线 probe，加载 checkpoint，在固定 dataloader samples 上测量但不更新 optimizer；
3. 在线 trainer 只增加低成本 clipping 可观测性：复用 `clip_grad_norm_` 返回的 pre-clipping norm，额外计算 clipping-trigger flag 和 scale。

不采用：

- 每个训练 step 计算 per-loss gradients：需要多次遍历 Qwen 和 Action Expert 的 backward graph，会显著增加时间、显存和分布式通信；
- 只根据 raw loss 校准：loss 数值不能决定参数更新强度，也不能显示方向冲突；
- 只有离线 diagnostics：checkpoint probe 可以做梯度归因，但看不到连续 loss、clipping 和预测趋势。

## Probe 输入与可复现性

命令参数：

- `--config_yaml`：用于构建模型和 benchmark dataloader 的完整 V2 YAML；
- `--checkpoint`：精确的 `.pt` 或 `.safetensors` checkpoint；
- `--output`：JSON 输出路径；
- `--num_batches`：默认 `8`；
- `--batch_size`：默认 `1`；
- `--seed`：默认 `42`；
- `--qwen_tail_layers`：默认 `2`；
- `--sample_indices`：可选的逗号分隔样本 index，覆盖 seeded sampling。

Probe 把 dataset workers 设为 0。默认根据 `seed` 从完整 dataset 均匀无放回采样，而不是只取最前面的 episodes。输出中保存 sample indices，后续 checkpoint 可以通过 `--sample_indices` 重用完全相同的样本集合。

这些样本不是 validation set：它们只用于诊断 training-distribution objective，并且永远不更新参数。

Provenance 保存：checkpoint 绝对路径、文件大小和修改时间、config SHA-256、样本数、seed、参数名/参数量、dtype、device 和 Git revision。不会对数 GB checkpoint 做完整 hash。LIBERO、CALVIN 和 RoboCasa 共用同一命令以及各自现有 V2 YAML。

模型保持 train mode，以复现训练计算路径。每个 batch 只进行一次 forward，因此所有 per-loss gradient measurement 使用同一组 action diffusion noise 和 dropout realization。每个 batch forward 前按 `seed + batch ordinal` 重置随机数生成器，使不同 checkpoint probe 使用相同随机实现。

不构建 optimizer 或 scheduler，也不执行任何参数更新。

## Objectives 与有效权重

Probe 从 `QwenGR00TCoTV2.forward` 读取：

| 名称 | Raw objective | 有效权重 |
|---|---|---:|
| `action` | `action_loss` | `lambda_action` |
| `depth_current` | `depth_current_loss` | `lambda_depth_current` |
| `depth_future` | `depth_future_loss` | `lambda_depth_future` |
| `uvd_absolute` | `uvd_absolute_loss` | `lambda_uvd` |
| `uvd_relative` | `uvd_relative_loss` | `lambda_uvd * lambda_uvd_relative` |

`uvd_loss` 不作为额外 objective，因为它已经等于 `uvd_absolute + lambda_uvd_relative * uvd_relative`。同时包含两者会重复计算 UVD supervision。

## 共享参数范围

默认只测量 action 和 auxiliary objectives 可能发生交互的参数：

- `geometry_tokens` 中所有 trainable parameters；
- Qwen language model 最后两个 Transformer layers 中的 trainable parameters。

Depth decoder、UVD numeric head 和 Action Expert 私有参数不参与 cosine 计算。它们的梯度主要反映各自 head fitting，而不是共享表征上的任务竞争。

Qwen tail layer 数可配置；`0` 表示只测 geometry tokens。第一版暂不支持 full-Qwen scope，因为同时保留多组完整模型梯度成本过高，并非首轮诊断所必需。

以下情况明确报错：请求的 Qwen layer path 不存在、选定 scope 为空，或者所有测量 batch 上 action gradient 都为零。

## 梯度计算

对同一次 forward 的结果，在选定共享参数上调用 `torch.autograd.grad`。它不会写入参数 `.grad`，因此不会意外执行优化步骤。

每个参数组和 objective 都以 FP32 累积：

\[
\lVert g_i\rVert_2,
\qquad
\lVert \lambda_i g_i\rVert_2,
\qquad
\cos(g_i,g_a)=
\frac{g_i^\top g_a}{\lVert g_i\rVert_2\lVert g_a\rVert_2}.
\]

未使用参数的 gradient 在同一选定参数向量中按零处理。实现只保留 action gradient 和正在累加的 weighted auxiliary gradient；各 auxiliary objective 顺序处理，不同时保留五份完整梯度副本。

同时报告：

\[
g_{aux}=\sum_i\lambda_i g_i,
\quad
\frac{\lVert g_{aux}\rVert_2}{\lVert\lambda_a g_a\rVert_2},
\quad
\cos(g_{aux},g_a).
\]

统计分别针对：`geometry_tokens`、`qwen_tail` 和二者并集 `shared_total`。

## 输出

JSON 包含：

- 每个 batch 的 raw/weighted losses；
- raw/weighted gradient norms；
- auxiliary-to-action norm ratios；
- 每个 auxiliary gradient 与 action gradient 的 cosine；
- combined auxiliary norm、ratio 和 cosine；
- 跨 batch 的 mean、standard deviation、median、minimum 和 maximum；
- finite/nonzero flags 和有效测量数量；
- provenance 与选定参数 metadata。

终端打印紧凑表格，但跨 checkpoint 比较时以 JSON 为准。

## 在线补充指标

正常训练继续保留当前 raw/weighted loss 和周期性 combined-module gradient logs。启用 gradient clipping 时增加：

- `train/grad_norm_pre_clip`；
- `train/grad_clip_threshold`；
- `train/grad_clip_triggered`；
- `train/grad_clip_scale = min(1, threshold / max(pre_clip_norm, eps))`。

这些字段直接复用已有 gradient norm，不安装额外 hook，不增加 backward。

## 错误处理

以下情况命令返回非零状态：checkpoint 不存在、缺少 V2 objective keys、参数 scope 为空、loss/gradient 非有限，或者所有 batch 上 action gradient 都为零。

单个 batch 上某个 auxiliary gradient 为零只记录，不直接判定失败，因为 validity mask 可能让该 objective 在该 batch 上没有有效监督。

脚本不修改 model、optimizer、scheduler、dataset 或 checkpoint。只写入用户指定的 probe JSON。

## 验证要求

使用小型可微 toy model 验证：

1. aligned、orthogonal、conflicting gradients 的 L2 norm 和 cosine sign 精确正确；
2. UVD relative 的嵌套有效权重正确；
3. unused gradients 按零表示；
4. combined auxiliary statistics 等于手算向量和；
5. 在线 clipping metrics 能区分 clipped/unclipped step；
6. CLI 能拒绝缺失 objective keys 和空参数 scope。

另用一个 synthetic V2-shaped module tree 验证参数选择。真实 checkpoint 运行属于实验 smoke test，不作为单元测试要求。

## 科学解释边界

测试通过只能说明 probe 计算了预期量，不能说明当前权重最优，也不能说明 auxiliary geometry 提高了 policy success。权重决策需要在多个 checkpoint 上，对有代表性的固定 batches 重复测量，再进行短程受控训练比较。
