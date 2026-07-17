# GPU 训练占用脚本设计

## 目标

在 `/home/yxz/CoT/gpu_script` 下提供一个独立的 `train.sh` 工具：等待任意四张当前可用的 A800 GPU，使用有上限的 PyTorch 保活进程占用每卡约 70,000 MiB，并把利用率目标设为至少 85%；运行训练命令时释放占用，训练退出后自动重新申请四张卡。

## 范围与安全边界

- 只选择没有计算进程、剩余显存满足阈值且在启动前再次通过检查的 GPU。
- 不使用 `killall`，不杀任意 PID，也不修改其他进程的 CUDA 状态。
- 只清理由本工具记录并启动的 worker PID。
- 每卡目标为 70,000 MiB，选择时额外保留 1,500 MiB 余量；利用率目标为 85%，不追求 100%。
- 找不到四张同时满足条件的 GPU 时等待，不部分占用。
- “低输出”仅表示终端不刷屏；不隐藏进程、不伪装进程、不规避 `nvidia-smi`、管理员或集群监控。

## 使用接口

```bash
cd /home/yxz/CoT/gpu_script
./train.sh hold
./train.sh status
./train.sh release
./train.sh run -- accelerate launch ...
```

`hold` 会等待四张卡并持续占用，直到手动中断；`release` 只释放本工具自己的 worker；`run -- COMMAND ...` 会先释放占用，执行命令，命令退出或被 Ctrl-C 中断后自动重新等待四张卡。默认使用已激活的 `CoT` 环境，也允许用 `PYTHON_BIN` 显式指定解释器。

## 架构

`train.sh` 负责 GPU 发现、筛选、锁、状态、进程生命周期和 `hold/release/run/status` 接口。它通过 `nvidia-smi` 获取所有 GPU 的显存与计算进程，选择候选卡，在启动前二次检查，然后每卡启动一个 worker。锁目录和状态文件用于避免同一目录下多个实例抢占同一批卡。

`train_worker.py` 是单卡 PyTorch 进程。它在安全上限内分块申请显存，然后持续执行 FP16 矩阵乘法。收到 SIGTERM 或 SIGINT 时退出并释放自己的张量。worker 只由 `train.sh` 管理，不触碰其他进程。

## 生命周期

```text
hold/run -> 轮询候选卡 -> 二次检查 -> 启动 4 个 worker
       ^                                      |
       |                                      v
       +----------- 训练命令结束 <------ 释放自己的 worker
```

对于 `run`，训练命令启动前会先释放保活 worker；命令结束后由 shell trap 重新进入等待/占用流程，并保留训练命令原始退出码。期间如果其他任务占用了 GPU，本工具不会强行抢夺，而是继续等待。

## 异常处理

- `nvidia-smi`、Python 或 PyTorch 不可用时，提前给出明确错误。
- 任意 worker 启动失败时，停止本次已经启动的全部自有 worker，再重新等待。
- 遇到过期状态文件时，确认对应 PID 不存在后清理状态。
- 训练命令退出码在重新占用动作后仍然保留。
- 显存分配使用目标值减安全余量，不按总显存盲目申请，避免 OOM。

## 验证

- Bash 语法检查与 Python 字节码编译。
- 在当前 8 张 A800 上只读检查 GPU 发现逻辑，不分配显存。
- 用降低后的测试参数执行短暂 hold/release 冒烟测试。
- 用 `run -- true` 验证释放、执行命令和退出后的重新等待。
- 验证状态只显示本工具 PID，第二个实例会拒绝启动。
