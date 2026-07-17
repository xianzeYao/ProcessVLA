# GPU 训练占用脚本实现计划

> **给执行代理：** 按任务逐项执行，使用复选框跟踪进度。实现必须遵守全局约束。

**目标：** 构建一个安全的本地 GPU 占用工具，使用任意四张可用 A800，每卡目标 70,000 MiB、利用率目标至少 85%，并在包装的训练命令退出后重新占用。

**架构：** Bash 控制器 `train.sh` 负责 GPU 筛选、锁、状态、进程生命周期和 `hold/release/run/status` 接口；每张 GPU 启动一个独立的 Python/PyTorch worker。控制器在启动前二次检查候选卡，并且只清理状态文件中记录的自有 PID。

**技术栈：** Bash、`nvidia-smi`、Python 3、`CoT` 环境中的 PyTorch。

## 全局约束

- 必须正好占用 4 张 GPU，索引可以是可见 GPU 中的任意 4 张。
- 每卡目标为 `70,000 MiB`，选择时至少保留 `1,500 MiB` 空闲余量。
- 利用率目标为 `85%`，不强行追求 100%。
- 绝不杀任意进程，只能终止本工具记录且验证仍属于本工具的 PID。
- 不能部分占用；启动失败时清理本轮所有 worker 后重试。
- 默认使用调用者已激活的 `CoT` 环境；也可以通过 `PYTHON_BIN` 指定 Python。
- 低输出不等于隐藏：不规避 `nvidia-smi`、管理员或集群监控，不伪装进程名。

---

### 任务 1：添加单卡 worker 与单元测试

**文件：**
- 新建：`/home/yxz/CoT/gpu_script/train_worker.py`
- 新建：`/home/yxz/CoT/gpu_script/tests/test_train_worker.py`

**接口：**
- 命令行：`train_worker.py --gpu-index INDEX --target-memory-mib N --safety-mib N --heartbeat PATH --matrix-size N`。
- 收到 SIGTERM/SIGINT 时正常退出；CUDA 显存分配失败或参数非法时返回非零。
- 不执行全局进程终止，只写入传入的 heartbeat 路径。

- [ ] 写纯函数测试：分配预算等于 `max(0, target - safety)`；非正尺寸被拒绝；heartbeat 的父目录会创建。
- [ ] 运行 `PYTHONPATH=/home/yxz/CoT/gpu_script python -m pytest /home/yxz/CoT/gpu_script/tests/test_train_worker.py -q`，预期初始失败，因为模块尚不存在。
- [ ] 实现参数解析、`compute_allocation_bytes`、分块 `torch.empty(..., dtype=torch.uint8)` 显存申请，以及使用两个 FP16 方阵循环执行 `torch.matmul`。设置 CUDA 设备并在矩阵乘后同步；信号处理器设置停止事件并释放张量。
- [ ] 在 `CoT` 环境执行单元测试与 `python -m py_compile /home/yxz/CoT/gpu_script/train_worker.py`，预期测试通过、编译返回 0。

### 任务 2：添加 `train.sh` 控制器

**文件：**
- 新建：`/home/yxz/CoT/gpu_script/train.sh`
- 新建：`/home/yxz/CoT/gpu_script/README.md`

**接口：**
- `./train.sh hold`：等待四张安全候选卡，启动 worker 并阻塞。
- `./train.sh release`：只终止状态文件中属于本工具的 worker。
- `./train.sh status`：显示状态、GPU 索引、自有 PID 与存活情况。
- `./train.sh run -- COMMAND [ARGS...]`：释放占用、执行命令、命令退出后重新进入占用流程，并返回命令原始退出码。

- [ ] 在 README 中说明 `conda activate CoT`、`PYTHON_BIN`、保守默认值、`GPU_COUNT=4`、`TARGET_MEMORY_MIB=70000`、`MIN_FREE_MEMORY_MIB=71500`、`MIN_UTILIZATION_PERCENT=85`，以及只选空闲 GPU 的规则。
- [ ] 使用 `mkdir "$STATE_DIR/lock"` 原子加锁；状态文件权限为 600；通过 `kill -0` 和 `/proc/$pid/cmdline` 检查 PID；过期状态视为已释放。
- [ ] 查询 `nvidia-smi --query-gpu=index,memory.free,utilization.gpu --format=csv,noheader,nounits`，逐卡查询计算 PID；有计算 PID、空闲显存不足或利用率超过阈值的卡都拒绝。选出四张后立即重新查询四张并再次检查。
- [ ] 每张卡启动一个 worker，日志写入 `logs/gpu-INDEX.log`，记录 PID；短暂等待后检查所有 worker 仍存活。失败时清理整批 worker 并回到轮询。
- [ ] 清理时只向记录且匹配 `train_worker.py` 的 PID 发送 TERM，最多等待 10 秒；仍存活时才向同一批匹配 PID 发送 KILL。
- [ ] 实现 `hold` 的 5 秒轮询、幂等 `release`、`status` 和带 trap 的 `run`。Ctrl-C 时仍执行清理。
- [ ] 执行 `bash -n /home/yxz/CoT/gpu_script/train.sh` 与 `./train.sh status`，预期语法通过且不改动其他 GPU 进程。

### 任务 3：在当前 8 卡机器上验证

- [ ] 运行只读 GPU 发现，确认不启动 worker、不分配显存、不终止其他进程。
- [ ] 用 `TARGET_MEMORY_MIB=1000 MIN_FREE_MEMORY_MIB=3000 ./train.sh hold` 做短暂冒烟测试，随后 Ctrl-C；预期只启动满足空闲条件的四卡，并只删除自己的 worker。
- [ ] 运行 `TARGET_MEMORY_MIB=1000 MIN_FREE_MEMORY_MIB=3000 ./train.sh run -- bash -c 'sleep 2'`；预期命令前释放占用，命令返回 0，随后重新进入占用等待。
- [ ] 运行 `./train.sh print-config`，确认输出为 4 张卡、70,000 MiB、1,500 MiB 余量和 85% 利用率目标。
- [ ] 先执行 `conda activate CoT`，再使用 `./train.sh run -- <训练命令>`；四卡不可同时安全使用时只等待，不过载、不部分占用。
