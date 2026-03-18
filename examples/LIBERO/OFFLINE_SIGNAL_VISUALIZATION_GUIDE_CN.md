# ProcessVLA Signal 可视化说明

这套工具现在明确分成两条：

- online：
  推理时保存到 hidden states 里的 `signal`，后处理再可视化
- offline：
  不依赖在线保存的 `signal`，而是直接对 rollout 视频单独算 signal 曲线

## Online

入口脚本：

- `examples/LIBERO/eval_files/visualize_online_signal_from_hiddenstates.py`

它做两件事：

1. 从 `step_*_last_hidden_states_meta.pt` 里把推理时保存的 `signal` 拼成一条 online 曲线
2. 把这条曲线叠到 rollout 视频上

最小用法：

```bash
python examples/LIBERO/eval_files/visualize_online_signal_from_hiddenstates.py \
  --video-path /path/to/rollout.mp4 \
  --hidden-dir /path/to/signal_dumps/task_episode_0
```

输出：

- `/path/to/rollout_online_signal_curve.npz`
- `/path/to/rollout_online_signal_overlay.mp4`

## Offline

入口脚本：

- `examples/LIBERO/eval_files/offline_signal_benchmark.py`
- `examples/LIBERO/eval_files/run_offline_signal_and_visualization.sh`

它支持两种方式：

1. 直接根据 rollout 视频离线计算 signal，并顺手渲染
2. 直接读取你已经离线算好的多条 `.npz` 曲线，再渲染

最小用法：

```bash
python examples/LIBERO/eval_files/offline_signal_benchmark.py \
  --video-path /path/to/rollout.mp4 \
  --instruction "open the middle drawer"
```

输出：

- `/path/to/rollout_signal_curve.npz`
- `/path/to/rollout_offline_signals_overlay.mp4`

如果你已经离线算好了多条曲线，也可以直接只做渲染：

```bash
python examples/LIBERO/eval_files/offline_signal_benchmark.py \
  --video-path /path/to/rollout.mp4 \
  --curve-paths /path/to/a_signal_curve.npz /path/to/b_signal_curve.npz
```

也支持用 rollout + hidden-dir 自动拿 instruction，再离线算：

```bash
python examples/LIBERO/eval_files/offline_signal_benchmark.py \
  --video-path /path/to/rollout.mp4 \
  --hidden-dir /path/to/signal_dumps/task_episode_0
```

## 当前状态

- `use_signal` / `inject_signal` 已经接到评测链路和模型接口里了
- hidden states / metadata 也可以按 episode + step 落盘
- online 的 `signal` 可视化读的是推理时保存下来的 `signal`
- offline 的 `signal` 计算读的是 rollout 视频
- `signal` 本体还没有接具体算法，目前 online/offline 都还是占位结果
