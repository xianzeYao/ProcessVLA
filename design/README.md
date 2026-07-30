# CoT_vla Design Index

本目录保存 Depth–UVD Geometric CoT 的方法设计、实现事实和后续候选方向。
文档按版本组织；训练日志、评测结果和临时排障记录不放在这里。

## 当前主线

| 状态 | 文档 | 用途 |
|---|---|---|
| Proposed / design approved | [V2 Design Report](v2/depth-uvd-geometric-cot-v2-design-report.md) | 下一版直接将几何 query 放入 Qwen 的方法定义、mask、数据流和验证标准 |
| Implemented / historical baseline | [V1 Complete](v1/depth-uvd-cot-v1-complete.md) | 当前 V1 代码、训练、诊断和评测的完整事实记录 |

V2 尚未实现。V1 仍是当前可运行实现；V2 文档中的接口和行为不能当作当前代码事实。

## V1 文档

| 文件 | 角色 |
|---|---|
| [depth-uvd-cot-v1-complete.md](v1/depth-uvd-cot-v1-complete.md) | V1 规范、实现和实验状态的 canonical 文档 |
| [depth-uvd-cot-v1-design.md](v1/depth-uvd-cot-v1-design.md) | 最初的方法设计和数据契约 |
| [depth-uvd-cot-v1-summary.md](v1/depth-uvd-cot-v1-summary.md) | 较短的设计与验证清单 |
| [depth_uvd_cot_v1_architecture.png](v1/depth_uvd_cot_v1_architecture.png) | V1 架构图 |
| [depth_uvd_cot_v1_timeline.png](v1/depth_uvd_cot_v1_timeline.png) | V1 时间对齐图 |

阅读 V1 时优先使用 `complete.md`；另外两份文档保留设计演化过程，不再作为唯一事实来源。

## V2 文档

| 文件 | 角色 |
|---|---|
| [depth-uvd-geometric-cot-v2-design-report.md](v2/depth-uvd-geometric-cot-v2-design-report.md) | V2 canonical Design Report |
| `depth_uvd_geometric_cot_v2_architecture.png` | V2 架构图；由 imagegen 生成后与报告同目录保存 |

## 后续候选方向

[geometric_cot_adapter_idea.md](geometric_cot_adapter_idea.md) 是从强 baseline
checkpoint 安装轻量几何 adapter 的独立支线。它不是 V2：

- V2：query 直接进入 Qwen，强调深层多模态几何推理；
- Adapter：保留外置模块，强调 baseline-preserving 和跨架构插件化。

二者应分别做实验，不在同一版本中同时引入。

## 维护约定

1. 每个版本只指定一份 canonical 设计文档。
2. 文档必须标注 `Proposed`、`Implemented` 或 `Evaluated`，不能把设计假设写成代码事实。
3. 架构图放在对应版本目录，Markdown 使用相对路径。
4. 训练参数的最终事实以实际运行 YAML 和 checkpoint 元数据为准；设计报告中的数值是起始约定。
5. 设计变更先更新对应版本报告，再进入实现计划。
