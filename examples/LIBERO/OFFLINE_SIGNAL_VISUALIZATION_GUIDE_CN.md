# StarVLA 纯离线 Signal 可视化 LIBERO 实用手册

这份手册只讲一件事：

如何对同一个 LIBERO rollout 纯离线计算多个 signal 模型的曲线，并合成一个可视化视频。

这里的“多个模型”指 signal benchmark 模型，例如 `liv`、`robometer`、`vlac`、`robodopamine`。它不是“多个 policy checkpoint 同屏对比脚本”。如果你要比较多个 policy checkpoint，当前做法是对每个 checkpoint 的 rollout 目录分别跑一遍流程。

## 1. 先认清三个入口脚本

- `examples/LIBERO/eval_files/compute_external_signal_curve.py`
  负责离线计算单个模型的曲线，并保存为 sidecar 文件：
  `<video_stem>_<model>_curve.npz`
- `examples/LIBERO/eval_files/visualize_signal_benchmark_with_video.py`
  负责把多个 signal 曲线和 rollout 视频合成为：
  `<video_stem>_signals_overlay.mp4`

- `examples/LIBERO/eval_files/run_offline_signal_and_visualization.sh`
  是一份“能跑通的模板”，但里面的路径、Python 环境、视频样例都是硬编码的。建议把它当模板抄，不要直接无脑运行。

## 2. 前置条件

至少要满足下面这些条件：

- 你已经有一批 LIBERO rollout 视频，例如：
  `rollout_open_the_middle_drawer_of_the_cabinet_episode1_success.mp4`
- 你能为每个 rollout 提供对应的任务指令文本 `instruction`
- 如果你要跑多模型 signal benchmark，需要相应模型的运行环境可用。

当前仓库里这几类 signal 的现实依赖关系是：

- `liv`：直接依赖当前仓库里的 `starVLA.model.framework.liv_utils`
- `robometer`：依赖本仓库同级的 `robometer`
- `vlac`：依赖本仓库同级的 `VLAC`
- `robodopamine`：依赖本仓库同级的 `Robo-Dopamine`

实践上，最稳妥的方式是像 `run_offline_signal_and_visualization.sh` 那样，为不同模型准备不同的 Python 解释器。

如果你只关心“纯离线重算曲线”，而不关心 StarVLA 评测阶段在线保存的内容，那么要点很简单：

- 不需要 `hidden_states`
- 只要你能提供 `instruction` 文本，就能离线重算 `liv / robometer / vlac / robodopamine`

其中只有 `VLAC` 额外要求 `reference video`，`Robo-Dopamine` 的 wrist 视频是可选增强项，不是硬要求。

## 3. 推荐的目录组织

以一个纯离线重算目录为例：

```text
offline_signal_bundle/
├── videos/
│   ├── rollout_open_the_middle_drawer_of_the_cabinet_episode1_success.mp4
│   ├── rollout_open_the_middle_drawer_of_the_cabinet_wrist_episode1_success.mp4
│   └── rollout_open_the_middle_drawer_of_the_cabinet_reference_success.mp4
├── metadata/
│   └── instructions.json
├── repos/
│   ├── ProcessVLA/
│   ├── robometer/
│   ├── VLAC/
│   ├── Robo-Dopamine/
│   └── LIV/
└── models/
```

## 4. 纯离线重算前要准备什么

对每个 rollout，纯离线模式只关心下面四类输入：

- 主视角 rollout 视频
- `instruction` 文本
- `VLAC` 的 reference video
- `Robo-Dopamine` 可选 wrist 视频

如果你打算把这套流程发给别人用，推荐额外准备一个文件名到指令的映射表，例如：

```json
{
  "rollout_open_the_middle_drawer_of_the_cabinet_episode1_success.mp4": "open the middle drawer of the cabinet"
}
```

## 5. 多个 signal 模型可视化同一个 LIBERO rollout

下面是最常用的链路。

### Step 1. 先确定一个 rollout 视频

假设：

```bash
export PROCESSVLA_ROOT=/path/to/Critic4VLA/ProcessVLA
export SIGNAL_SCRIPT=$PROCESSVLA_ROOT/examples/LIBERO/eval_files/compute_external_signal_curve.py
export VIS_SCRIPT=$PROCESSVLA_ROOT/examples/LIBERO/eval_files/visualize_signal_benchmark_with_video.py
export STARVLA_PY=python
export ROBOMETER_PY=/path/to/robometer/python
export VLAC_PY=/path/to/vlac/python
export ROBODOPAMINE_PY=/path/to/robodopamine/python
export RUN_ROOT=/path/to/starvla4libero/my_model_steps_30000_pytorch_model.pt
export TASK_SUITE=libero_goal
export VIDEO=$RUN_ROOT/results/$TASK_SUITE/rollout_open_the_middle_drawer_of_the_cabinet_episode1_success.mp4
export INSTRUCTION="open the middle drawer of the cabinet"
export REF_VIDEO=$RUN_ROOT/results/$TASK_SUITE/rollout_open_the_middle_drawer_of_the_cabinet_episode2_success.mp4
export WRIST_VIDEO=$RUN_ROOT/results/$TASK_SUITE/rollout_open_the_middle_drawer_of_the_cabinet_wrist_episode1_success.mp4
```

### Step 2. 计算各个模型的离线曲线

#### 2.1 LIV

```bash
"$STARVLA_PY" "$SIGNAL_SCRIPT" \
  --model liv \
  --video-path "$VIDEO" \
  --instruction "$INSTRUCTION"
```

输出：

```text
<video_stem>_liv_curve.npz
```

#### 2.2 Robometer

```bash
"$ROBOMETER_PY" "$SIGNAL_SCRIPT" \
  --model robometer \
  --video-path "$VIDEO" \
  --instruction "$INSTRUCTION"
```

输出：

```text
<video_stem>_robometer_curve.npz
```

#### 2.3 VLAC

VLAC 必须额外提供一个 reference/demo video。实操里最好选“同任务的成功轨迹”。

```bash
"$VLAC_PY" "$SIGNAL_SCRIPT" \
  --model vlac \
  --video-path "$VIDEO" \
  --instruction "$INSTRUCTION" \
  --reference-video-path "$REF_VIDEO"
```

输出：

```text
<video_stem>_vlac_curve.npz
```

#### 2.4 Robo-Dopamine

如果有 wrist 视频，建议传进去；没有也能跑。

```bash
"$ROBODOPAMINE_PY" "$SIGNAL_SCRIPT" \
  --model robodopamine \
  --video-path "$VIDEO" \
  --instruction "$INSTRUCTION" \
  --wrist-video-path "$WRIST_VIDEO"
```

输出：

```text
<video_stem>_robodopamine_curve.npz
```

### Step 3. 合成多模型 overlay 视频

当上面这些 `.npz` 都在视频旁边以后，执行：

```bash
"$STARVLA_PY" "$VIS_SCRIPT" \
  --video-path "$VIDEO" \
  --instruction "$INSTRUCTION" \
  --models liv robometer vlac robodopamine \
  --plot-layout subplot
```

输出两个文件：

- `<video_stem>_signals_overlay.mp4`
- `<video_stem>_signals_overlay.npz`

其中：

- `subplot` 会给每个 signal 一个独立子图，更适合 4 个模型一起看
- `sameplot` 会把所有曲线画到一个坐标轴上，更适合只看 2 到 3 条曲线

### Step 4. 你可以省掉的步骤

`visualize_signal_benchmark_with_video.py` 对 `liv` 做了一个特殊处理：

- 如果它发现 `<video_stem>_liv_curve.npz` 不存在，会现场重算 `liv`

但对下面这些模型不会自动重算：

- `robometer`
- `vlac`
- `robodopamine`

也就是说，多模型可视化时，通常还是建议先把所有外部 `.npz` 都算好。

## 6. 批量处理多个 rollout 视频

如果你想对同一个 checkpoint 下的很多视频批量做多模型可视化，推荐这种写法：

注意：

- 如果这些视频来自不同任务，不要给所有视频共用同一个 `INSTRUCTION`
- 纯离线分发时，建议你自己准备一个“视频文件名 -> instruction”的映射表，例如第 9.4 节里的 `instructions.json`

```bash
export PROCESSVLA_ROOT=/path/to/Critic4VLA/ProcessVLA
export SIGNAL_SCRIPT=$PROCESSVLA_ROOT/examples/LIBERO/eval_files/compute_external_signal_curve.py
export VIS_SCRIPT=$PROCESSVLA_ROOT/examples/LIBERO/eval_files/visualize_signal_benchmark_with_video.py
export STARVLA_PY=python
export ROBOMETER_PY=/path/to/robometer/python
export VLAC_PY=/path/to/vlac/python
export ROBODOPAMINE_PY=/path/to/robodopamine/python
export RUN_ROOT=/path/to/starvla4libero/my_model_steps_30000_pytorch_model.pt
export TASK_SUITE=libero_goal
export RESULT_DIR=$RUN_ROOT/results/$TASK_SUITE
export REF_VIDEO=$RESULT_DIR/rollout_open_the_middle_drawer_of_the_cabinet_episode2_success.mp4

for VIDEO in "$RESULT_DIR"/rollout_*_episode*_success.mp4; do
  BASENAME="$(basename "$VIDEO")"
  INSTRUCTION="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))[sys.argv[2]])' metadata/instructions.json "$BASENAME")"

  "$STARVLA_PY" "$SIGNAL_SCRIPT" \
    --model liv \
    --video-path "$VIDEO" \
    --instruction "$INSTRUCTION"

  "$ROBOMETER_PY" "$SIGNAL_SCRIPT" \
    --model robometer \
    --video-path "$VIDEO" \
    --instruction "$INSTRUCTION"

  "$VLAC_PY" "$SIGNAL_SCRIPT" \
    --model vlac \
    --video-path "$VIDEO" \
    --instruction "$INSTRUCTION" \
    --reference-video-path "$REF_VIDEO"

  "$ROBODOPAMINE_PY" "$SIGNAL_SCRIPT" \
    --model robodopamine \
    --video-path "$VIDEO" \
    --instruction "$INSTRUCTION"

  "$STARVLA_PY" "$VIS_SCRIPT" \
    --video-path "$VIDEO" \
    --instruction "$INSTRUCTION" \
    --models liv robometer vlac robodopamine \
    --plot-layout subplot
done
```

## 7. 批量处理多个 policy checkpoint

如果你说的“多个模型”其实是“多个 StarVLA policy checkpoint”，那当前脚本的正确使用方式是：

1. 每个 checkpoint 先各自跑出自己的 `<RUN_ROOT>`
2. 在每个 `<RUN_ROOT>` 下面分别执行第 5 节或第 6 节
3. 最终对比不同 checkpoint 生成的多个 overlay 视频

当前仓库没有“把多个 checkpoint 的曲线直接画进同一个 overlay 视频”的现成脚本。

## 8. 常见坑

### 8.1 忘了传 `--instruction`

纯离线模式下，最稳妥的做法就是显式传：

```bash
--instruction "put the bowl on the plate"
```

### 8.2 VLAC 必须给 reference video

这个是硬要求：

```text
--reference-video-path is required for vlac
```

通常选同任务的成功轨迹最稳。

### 8.3 没有 PyAV 时的读写问题

脚本会优先用 `PyAV`，其次尝试 `imageio`，再其次尝试 `OpenCV`。如果你遇到：

- 读 mp4 失败
- 写 overlay 失败

优先检查这几个库在对应环境里是否安装完整。

### 8.4 给别人跑“纯离线版”时，不要依赖默认模型路径

当前代码支持下面这些环境变量，发给别人时最好一起写进启动脚本：

```bash
export CRITIC4VLA_ROBOMETER_MODEL_PATH=/abs/path/to/Robo_Meter
export CRITIC4VLA_ROBOMETER_QWEN_SNAPSHOT_PATH=/abs/path/to/Qwen3-VL-4B-Instruct-snapshot
export CRITIC4VLA_ROBOMETER_UNSLOTH_SNAPSHOT_PATH=/abs/path/to/unsloth-Qwen3-VL-4B-Instruct-snapshot
export CRITIC4VLA_ROBODOPAMINE_MODEL_PATH=/abs/path/to/Robo_Dopamine
export CRITIC4VLA_VLAC_MODEL_PATH=/abs/path/to/VLAC
export CRITIC4VLA_VLAC_MODEL_TYPE=internvl2
```

如果这些变量不设，脚本会回退到你本机上的默认路径。把代码发给别人时，这通常是跑不通的。

## 9. 纯离线分发给别人时，你必须提供什么

如果你的目标是让别人“从零离线重算四种曲线”，最少要给下面这些东西。

### 9.1 必须给

- `ProcessVLA` 代码
- 同级依赖仓库：
  `robometer`、`VLAC`、`Robo-Dopamine`、`LIV`
- 主视角 rollout 视频
- 每个 rollout 对应的 `instruction` 文本
- 至少一个同任务的 `reference video`
  这是给 `VLAC` 用的
- 下面这些模型权重或本地快照：
  - `LIV` 权重
  - `Robometer` checkpoint
  - `Robometer` 依赖的本地 Qwen snapshot
  - `VLAC` checkpoint
  - `Robo-Dopamine` checkpoint
- 各环境的 Python 解释器或环境安装说明
- 一个可直接运行的 shell 脚本

### 9.2 建议给

- wrist 视频
  这样 `Robo-Dopamine` 的结果通常更完整
- 一份成功轨迹视频清单
  方便别人给 `VLAC` 选 reference
- 预先写好的环境变量脚本
  避免对方自己改路径

### 9.3 不需要给

如果你走的是“纯离线重算”路线，这些都不是硬要求：

- `hidden_states`

### 9.4 最推荐的交付目录

```text
offline_signal_bundle/
├── repos/
│   ├── ProcessVLA/
│   ├── robometer/
│   ├── VLAC/
│   ├── Robo-Dopamine/
│   └── LIV/
├── models/
│   ├── Robo_Meter/
│   ├── VLAC/
│   ├── Robo_Dopamine/
│   └── qwen_snapshots/
├── videos/
│   ├── rollout_xxx_episode1_success.mp4
│   ├── rollout_xxx_wrist_episode1_success.mp4
│   └── rollout_xxx_reference_success.mp4
├── metadata/
│   └── instructions.json
└── run_offline_signals.sh
```

其中 `instructions.json` 最简单可以写成：

```json
{
  "rollout_open_the_middle_drawer_of_the_cabinet_episode1_success.mp4": "open the middle drawer of the cabinet"
}
```

### 9.5 最小可运行脚本模板

```bash
#!/usr/bin/env bash
set -euo pipefail

export PROCESSVLA_ROOT=/abs/path/to/Critic4VLA/ProcessVLA
export SIGNAL_SCRIPT=$PROCESSVLA_ROOT/examples/LIBERO/eval_files/compute_external_signal_curve.py
export VIS_SCRIPT=$PROCESSVLA_ROOT/examples/LIBERO/eval_files/visualize_signal_benchmark_with_video.py
export STARVLA_PY=python
export ROBOMETER_PY=/path/to/robometer/python
export VLAC_PY=/path/to/vlac/python
export ROBODOPAMINE_PY=/path/to/robodopamine/python
export CRITIC4VLA_ROBOMETER_MODEL_PATH=/abs/path/models/Robo_Meter
export CRITIC4VLA_ROBOMETER_QWEN_SNAPSHOT_PATH=/abs/path/models/qwen_snapshots/Qwen3-VL-4B-Instruct
export CRITIC4VLA_ROBOMETER_UNSLOTH_SNAPSHOT_PATH=/abs/path/models/qwen_snapshots/unsloth-Qwen3-VL-4B-Instruct
export CRITIC4VLA_ROBODOPAMINE_MODEL_PATH=/abs/path/models/Robo_Dopamine
export CRITIC4VLA_VLAC_MODEL_PATH=/abs/path/models/VLAC

VIDEO=/abs/path/videos/rollout_open_the_middle_drawer_of_the_cabinet_episode1_success.mp4
WRIST_VIDEO=/abs/path/videos/rollout_open_the_middle_drawer_of_the_cabinet_wrist_episode1_success.mp4
REF_VIDEO=/abs/path/videos/rollout_open_the_middle_drawer_of_the_cabinet_reference_success.mp4
INSTRUCTION="open the middle drawer of the cabinet"

"$STARVLA_PY" "$SIGNAL_SCRIPT" \
  --model liv \
  --video-path "$VIDEO" \
  --instruction "$INSTRUCTION"

"$ROBOMETER_PY" "$SIGNAL_SCRIPT" \
  --model robometer \
  --video-path "$VIDEO" \
  --instruction "$INSTRUCTION"

"$VLAC_PY" "$SIGNAL_SCRIPT" \
  --model vlac \
  --video-path "$VIDEO" \
  --instruction "$INSTRUCTION" \
  --reference-video-path "$REF_VIDEO"

"$ROBODOPAMINE_PY" "$SIGNAL_SCRIPT" \
  --model robodopamine \
  --video-path "$VIDEO" \
  --instruction "$INSTRUCTION" \
  --wrist-video-path "$WRIST_VIDEO"

"$STARVLA_PY" "$VIS_SCRIPT" \
  --video-path "$VIDEO" \
  --instruction "$INSTRUCTION" \
  --models liv robometer vlac robodopamine \
  --plot-layout subplot
```

## 10. 最实用的建议

如果你只是想尽快跑通，按这个顺序来最省事：

1. 先确认已有 `rollout_*.mp4`
2. 为每个视频准备好 `instruction`
3. 先单独跑 `liv`
4. 再补 `robometer / vlac / robodopamine`
5. 最后再跑 `visualize_signal_benchmark_with_video.py`

如果你想直接照模板改，优先看：

- `examples/LIBERO/eval_files/run_offline_signal_and_visualization.sh`

但请先把里面这些硬编码路径改掉：

- `STARVLA_PY`
- `LIV_PY`
- `VLAC_PY`
- `ROBODOPAMINE_PY`
- `ROBOMETER_PY`
- `ROOT`
- `DEFAULT_REF_VIDEO`
