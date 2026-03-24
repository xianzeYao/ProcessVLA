# SLURM Skills for UniVLA (n84416302)

这份文档用于给 agent 直接执行：自动选资源、提交作业、监控状态、收集日志、汇总结果。

## 1. 账户与集群快照（以当前配置为准）

- 用户名：`n84416302`
- 用户组：`slurm-lrc-users`, `slurm-camera-users`, `c84391361`, `d84417809`
- 可用 QOS：`low`, `normal`, `high`, `highest`
- 常见 GPU：`h200`（主用）、`h100`
- 节点画像：约 `208 CPU / 1.5TB RAM / 8 GPU`（以节点实际为准）

## 2. 分区与时长策略（实用版）

优先按任务时长选分区，再按抢占需求选 QOS：

1. 快速调试（<= 1h）：`camera-short` + `normal`
2. 开发验证（<= 4h）：`lrc-dev` + `normal`
3. 常规训练（<= 48h）：`lrc-xlong` + `normal`
4. 高优先级交互/紧急任务（<= 10h）：`lrc-xlong` + `highest`
5. 更长训练（1.5d~5d）：`camera-long` / `camera-xlong` / `lrc-xlong`

### 2.1 分区限制速查表

| Partition | 典型时长上限 | 典型场景 |
| --- | --- | --- |
| `lrc-dev` | 4h | 快速开发/验证 |
| `lrc-xlong` | 5d | 长训练主力 |
| `camera-short` | 1h | 极短调试 |
| `camera-long` | 1d12h | 中等训练 |
| `camera-xlong` | 5d | 长训练 |
| `camera-inf` | 100d | 超长推理/训练 |
| `cpu` | 5d | CPU任务 |

## 3. 已验证可用命令组合（可直接复用）

```bash
srun --qos highest --partition=lrc-xlong --ntasks=1 --gres=gpu:h200:1 --time=09:59:59 --pty bash -i
srun --qos highest --partition=lrc-xlong --ntasks=1 --gres=gpu:h200:2 --time=09:59:59 --pty bash -i
srun --qos highest --partition=lrc-xlong --ntasks=1 --gres=gpu:h200:6 --time=26:00:00 --pty bash -i

srun --qos normal  --partition=lrc-xlong --ntasks=1 --gres=gpu:h200:2 --time=09:59:59 --pty bash -i
srun --qos normal  --partition=lrc-xlong --ntasks=1 --gres=gpu:h200:4 --time=48:00:00 --pty bash -i


srun --qos highest --partition=lrc-dev   --ntasks=1 --gres=gpu:h200:2 --time=1:59:59 --pty bash -i
srun --qos highest --partition=lrc-dev   --ntasks=1 --gres=gpu:h200:4 --time=1:59:59 --pty bash -i

srun --qos normal  --partition=camera-short --ntasks=1 --gres=gpu:h200:1 --time=0:59:00 --pty bash -i
srun --qos normal  --partition=camera-short --ntasks=1 --gres=gpu:h200:2 --time=0:59:00 --pty bash -i

srun --qos highest --partition=camera-long  --ntasks=1 --gres=gpu:h200:1 --time=09:59:59 --pty bash -i
srun --qos highest --partition=camera-xlong --ntasks=1 --gres=gpu:h200:1 --time=09:59:59 --pty bash -i
```

## 4. 常用 alias（建议放到 `~/.bashrc`）

```bash
alias count='ls -l | grep "^-" | wc -l'
alias gpustat='sinfo -p lrc-xlong,lrc-dev,camera-short,camera-long,camera-xlong -o"%P %.16F"'
alias igpu='srun --qos highest --partition=lrc-xlong --ntasks=1 --gres=gpu:h200:1 --time=09:59:59 --pty bash -i'
alias gpu='srun --partition=camera-short --ntasks=1 --gres=gpu:h200:1 --time=03:59:59 --pty bash -i'
alias job='squeue -u $USER'
alias check='scontrol show job'
alias e='vim'
alias gm='printutil'
```

## 5. Conda 环境激活（统一写法）

```bash
eval "$(conda shell.bash hook)" && conda activate dream_vla_temp
# 或
eval "$(conda shell.bash hook)" && conda activate dreamvla
```

## 6. Agent 自动提交流程（强约束）

当用户说“挂实验/跑训练/收日志”时，按以下流程执行：

1. 创建实验目录：`experiments/<exp_name>_<YYYYMMDD_HHMMSS>/`
2. 写入 `submit.sh`、`metadata.yaml`（记录参数、分区、qos、git commit）
3. 预检查：
   - `squeue -u $USER`
   - `sinfo -p <partition>`
   - 检查数据路径、checkpoint、conda 环境
4. `sbatch submit.sh` 并解析 `job_id`
5. 轮询状态并回传：
   - `squeue -j <job_id>`
   - `sacct -j <job_id> --format=JobID,State,Elapsed,ExitCode`
6. 收集日志：
   - `slurm_<job_id>.out`
   - `slurm_<job_id>.err`
   - `training.log`（若有 `tee`）
7. 任务结束后生成 `summary.md`：
   - 最终状态、耗时、关键 loss/metric、错误摘要、产物路径

## 7. 推荐 `sbatch` 模板（可参数化）

```bash
#!/bin/bash
#SBATCH --job-name=${JOB_NAME}
#SBATCH --partition=${PARTITION}
#SBATCH --qos=${QOS}
#SBATCH --gres=gpu:${GPU_TYPE}:${NUM_GPUS}
#SBATCH --time=${TIME_LIMIT}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=${CPUS_PER_TASK}
#SBATCH --mem=${MEMORY}
#SBATCH --output=${EXP_DIR}/slurm_%j.out
#SBATCH --error=${EXP_DIR}/slurm_%j.err

set -euo pipefail

echo "==== Job Meta ===="
echo "JobID: ${SLURM_JOB_ID}"
echo "Node:  $(hostname)"
echo "Start: $(date)"
echo "PWD:   $(pwd)"

eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENV}"

cd "${PROJECT_DIR}"
mkdir -p "${EXP_DIR}"

nvidia-smi || true

echo "==== Run Command ===="
echo "${TRAIN_CMD}"

bash -lc "${TRAIN_CMD}" 2>&1 | tee "${EXP_DIR}/training.log"
RC=${PIPESTATUS[0]}

echo "==== Job End ===="
echo "End:  $(date)"
echo "Code: ${RC}"
exit ${RC}
```

## 8. 监控与日志收集命令

```bash
# 任务状态
squeue -u $USER
squeue -j <job_id>
scontrol show job <job_id>
sacct -j <job_id> --format=JobID,JobName%30,Partition,State,Elapsed,ExitCode

# 实时日志
tail -f experiments/<exp>/slurm_<job_id>.out
tail -f experiments/<exp>/slurm_<job_id>.err
tail -f experiments/<exp>/training.log

# 快速抓错误
grep -nEi "error|exception|traceback|oom|cuda" experiments/<exp>/slurm_<job_id>.err
grep -nEi "loss|eval|acc|success|reward" experiments/<exp>/training.log | tail -n 50
```

## 9. 常见失败处理策略

1. `PD (Resources)`：减少 GPU 数，或换 `lrc-dev/camera-short` 短时跑验证
2. `QOS/Time limit`：降时长，或切到允许更长时长的分区
3. `OUT_OF_MEMORY`：降 `batch_size`、增 `grad_accumulation_steps`
4. 环境错误：先打印 `which python`、`python --version`、`conda info --envs`
5. 训练脚本退出非 0：保留 `PIPESTATUS[0]` 作为真实退出码，避免被 `tee` 吞掉

## 10. UniVLA/AReaL 里应继承的实践

1. 每个实验独立目录，集中存放 `submit.sh + metadata + slurm logs + training.log`
2. 脚本开头打印完整配置（模型、数据、超参、路径）
3. 开跑前做文件存在性检查（checkpoint / dataset / main script）
4. 训练命令统一 `2>&1 | tee ...`，并显式处理退出码
5. 结束后输出“关键产物路径”（ckpt、adapter、评估结果）

### 10.1 可直接参考的现有脚本

- UniVLA:
  - `start_slurm_scripts/run_vla_finetune_experiments.py`
  - `start_slurm_scripts/run_vlm_pretrain_experiments.py`
  - `start_slurm_scripts/submit_slurm_finetune_libero_stage1_lam.sh`
  - `start_slurm_scripts/submit_slurm_trace_lam_v2_bristol.sh`
- AReaL:
  - `/home/n84416302/AReaL/SLURM_TESTING_GUIDE.md`
  - `/home/n84416302/AReaL/run_sensenova_test_slurm_simple.sh`

## 11. 给未来 agent 的执行约定（可直接复制到系统提示）

1. 优先使用本文件中的分区/QOS白名单组合，不自行猜测未验证组合。
2. 所有实验都创建独立目录并保存 `submit.sh` 与日志。
3. 提交后必须回报 `job_id`、分区、qos、gpu、time、日志路径。
4. 运行中每次状态变化都更新：`PENDING -> RUNNING -> COMPLETED/FAILED`。
5. 失败时先给“可执行修复方案”，再自动生成下一版提交脚本（如用户允许）。
6. 优先 H200；若资源紧张才回退到 H100。

## 12. 已升级为 Codex Skill

- Skill 目录：`/home/n84416302/UniVLA/skills/slurm-experiment-ops`
- 技能入口：`/home/n84416302/UniVLA/skills/slurm-experiment-ops/SKILL.md`
- 自动脚本：
  - `skills/slurm-experiment-ops/scripts/submit_and_watch.sh`（主实现）
  - `scripts/submit_and_watch.sh`（项目内快捷入口）
