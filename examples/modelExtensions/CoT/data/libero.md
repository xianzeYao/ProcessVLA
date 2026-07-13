# LIBERO Replay RGBD 数据集

## 数据位置

四类 suite 的正式 replay 数据集统一放在：

```text
/root/data/yxz/datasets/libero_replpay/
├── libero_spatial_no_noops_replay
├── libero_object_no_noops_replay
├── libero_goal_no_noops_replay
└── libero_10_no_noops_replay
```

| suite | episodes | frames |
|---|---:|---:|
| `libero_spatial` | 432 | 52,970 |
| `libero_object` | 454 | 66,984 |
| `libero_goal` | 428 | 52,042 |
| `libero_10` | 379 | 101,469 |

对应的 origin HDF5 位于：

```text
/root/data/yxz/datasets/libero_original_hdf5/
├── libero_spatial/
├── libero_object/
├── libero_goal/
└── libero_10/
```

LeRobot↔HDF5 的逐帧 mapping 位于：

```text
/root/data/yxz/outputs/libero_hdf5_mapping/{spatial,object,goal,libero_10}/
```

## 每个 episode 的文件组织

以某个 suite 根目录为例：

```text
<dataset_root>/
├── data/chunk-000/episode_000000.parquet
├── videos/chunk-000/observation.images.image/episode_000000.mp4
├── videos/chunk-000/observation.images.wrist_image/episode_000000.mp4
├── depth/chunk-000/observation.depth.image_m/episode_000000.npz
├── depth/chunk-000/observation.depth.wrist_m/episode_000000.npz
└── camera/chunk-000/episode_000000.npz
```

RGB 视频是 replay 后重新渲染的 agentview/wrist RGB，分辨率 `256x256`、`20 fps`、H.264/AVC、`yuv420p`。存储图像采用与 LeRobot 对齐的 `rot180` 数组约定，gripper debug sites 已隐藏。

## Parquet 键值

每行对应一个 LeRobot 保留帧：

| key | 含义 |
|---|---|
| `observation.state` | replay-derived 8 维状态：`eef_xyz(3) + axis_angle(3) + gripper_qpos(2)`，float32 |
| `action` | 原始 LeRobot action，7 维 float32；未替换、未重新计算 |
| `timestamp` | LeRobot 时间戳 |
| `frame_index` | LeRobot 帧序号 |
| `episode_index` | LeRobot episode 序号 |
| `index` | 数据集全局索引 |
| `task_index` | task 索引 |
| `observation.depth.image_m_path` | agentview metric depth NPZ 的相对路径 |
| `observation.depth.wrist_m_path` | wrist metric depth NPZ 的相对路径 |
| `observation.camera.params_path` | camera 参数 NPZ 的相对路径 |
| `source.hdf5_path` | 对应 origin HDF5 文件路径 |
| `source.hdf5_demo_id` | 对应 HDF5 demo，例如 `demo_40` |
| `source.hdf5_index` | 该 LeRobot 帧对应的 HDF5 state index |

没有写入以下冗余字段：

```text
observation.eef.gripper_site_pos
observation.eef.agentview_uv
observation.eef.wrist_uv
observation.eef.agentview_depth_m
observation.eef.wrist_depth_m
```

## Depth NPZ

每个 depth NPZ 只有一个键：

```text
depth_m: float16[T, 256, 256]
```

单位是米（metric depth）。`T` 与对应 parquet episode 的帧数完全一致。

读取示例：

```python
import numpy as np

depth = np.load(dataset_root / row["observation.depth.image_m_path"])["depth_m"]
```

## Camera NPZ

每个 camera NPZ 有四个键：

```text
agentview_K              float32[T, 3, 3]
agentview_T_world_camera float32[T, 4, 4]
wrist_K                  float32[T, 3, 3]
wrist_T_world_camera     float32[T, 4, 4]
```

`K` 已经适配最终存储的 RGB/depth 图像方向，不再带 `_rot180` 后缀。

世界坐标点投影到相机坐标时使用：

```python
p_camera = np.linalg.inv(T_world_camera) @ [x, y, z, 1]
uv_h = K @ p_camera[:3]
uv = uv_h[:2] / uv_h[2]
```

## LeRobot 对齐规则

- episode 选择、去 noop、帧顺序以 LeRobot 数据为准。
- 每个保留帧通过 mapping 找回对应 HDF5 MuJoCo state。
- RGB、metric depth、camera 参数都从该 HDF5 state replay 得到。
- `source.hdf5_index` 显式保存帧对应关系，支持非连续删帧。
- `action` 保持原始 LeRobot action，保证动作监督与原 baseline 对齐。
- `observation.state` 使用 replay state，使 `state[:3]` 与 replay 中的 `gripper0_grip_site` 同源。

正式 replay 数据的构建日志位于：

```text
/root/data/yxz/outputs/libero_spatial_replay_rgbd_state_build.log
/root/data/yxz/outputs/libero_object_replay_rgbd_state_build.log
/root/data/yxz/outputs/libero_goal_replay_rgbd_state_build.log
/root/data/yxz/outputs/libero_10_replay_rgbd_state_build.log
```
