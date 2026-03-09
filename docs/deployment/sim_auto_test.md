# 仿真自动测试

本节描述在 **Kuavo 仿真环境** 中自动评估已训练策略的流程，包括入口脚本、配置、与仿真器的 ROS 通信、策略加载与 PI0/PI05 的部署适配。

---

## 1. 概述

仿真自动测试用于在仿真器中批量运行评估 episodes，统计任务成功率并保存 rollout 视频与日志。核心流程为：

- 加载已训练的策略 checkpoint 及 preprocessor；
- 与仿真器通过 ROS 进行 reset / init / success 通信；
- 每步：获取观测 → 预处理 → 策略推理 → 后处理 → 执行动作；
- 按 episode 统计成功/失败，输出汇总结果。

**入口脚本**：`kuavo_deploy/src/scripts/script_auto_test.py`  
**核心逻辑**：`kuavo_deploy/src/eval/sim_auto_test.py` 中的 `kuavo_eval_autotest()`

---

## 2. 前置条件

1. **仿真器已启动**：仿真环境（如比赛仿真器）需先运行，并与本脚本使用同一 `ROS_MASTER_URI`。
2. **ROS 服务**：
   - 脚本提供 `/simulator/init`，仿真器就绪时调用以通知脚本开始；
   - 仿真器需提供 `/simulator/reset`，用于每 episode 开始前重置场景；
   - 仿真器需提供 `/simulator/start`，每 episode 开始后由脚本调用以启动任务计时等。
3. **ROS 话题**：
   - `/simulator/success`（Bool）：仿真器在任务成功时发布 `True`；
   - `/kuavo/pause_state`、`/kuavo/stop_state`：可选，用于外部暂停/停止机械臂运动。
4. **Checkpoint 与 preprocessor**：训练输出目录（run_dir）需包含 `policy_preprocessor.json` 以及 epoch 子目录（如 `epoch3/` 或 `epochbest/`）内的 `model.safetensors`。

---

## 3. 配置文件

部署配置位于 `configs/deploy/kuavo_env.yaml`，与仿真评估相关的关键字段如下。

### 3.1 推理与任务配置（`inference`）

| 字段 | 说明 | 示例 |
|------|------|------|
| `policy_type` | 策略类型 | `diffusion`、`act`、`pi0`、`pi05` |
| `eval_episodes` | 评估 episode 数 | `100` |
| `seed` / `start_seed` | 随机种子 | `42` |
| `device` | 推理设备 | `cuda` 或 `cpu` |
| `max_episode_steps` | 单 episode 最大步数 | `200` |
| `task` | 任务名（路径用） | `sim_task2` |
| `method` | 方法名（路径用） | `pi05_rgb` |
| `timestamp` | run 时间戳（路径用） | `run_20260127_011739` |
| `epoch` | 检查点 epoch | `3` 或 `best` |
| `checkpoint_run_dir` | 可选，自定义 run 目录绝对路径 | 见下 |

### 3.2 检查点路径的两种方式

**方式一**：使用 `checkpoint_run_dir` + `epoch`（当 checkpoint 不在默认 `outputs/train` 下时）

```yaml
inference:
  checkpoint_run_dir: "/path/to/run_20260127_011739"
  epoch: 3
```

- `pretrained_path` = `{checkpoint_run_dir}/epoch3`
- 输出目录 = `{checkpoint_run_dir}/eval_output/epoch3`

**方式二**：使用 `task` + `method` + `timestamp`（默认）

```yaml
inference:
  task: sim_task2
  method: pi05_rgb
  timestamp: run_20260127_011739
  epoch: 3
```

- `pretrained_path` = `outputs/train/sim_task2/pi05_rgb/run_20260127_011739/epoch3`
- 输出目录 = `outputs/eval/sim_task2/pi05_rgb/run_20260127_011739/epoch3`

### 3.3 环境与 ROS 配置（`env`）

- `env_name`：`Kuavo-Sim`（仿真）或 `Kuavo-Real`（真机）；
- `obs_key_map`：ROS 话题到观测键的映射（RGB、深度、关节、夹爪等）；
- `image_size`、`depth_range`：需与训练时一致。

---

## 4. 运行方式

```bash
# 在项目根目录（如 /root/kuavo_ws/kuavo_data_challenge）下
conda activate kdc  # 或你的 conda 环境

python kuavo_deploy/src/scripts/script_auto_test.py \
  --task auto_test \
  --config configs/deploy/kuavo_env.yaml
```

可选参数：

- `--verbose` / `-v`：输出更详细的日志；
- `--dry_run`：仅打印将要执行的操作，不实际运行。

### 4.1 指定 GPU

```bash
CUDA_VISIBLE_DEVICES=2 python kuavo_deploy/src/scripts/script_auto_test.py \
  --task auto_test \
  --config configs/deploy/kuavo_env.yaml
```

### 4.2 PaliGemma Tokenizer 路径（PI0/PI05）

若 tokenizer 路径与训练时不同（例如换机器部署），可设置环境变量：

```bash
export PALIGEMMA_TOKENIZER_PATH=/root/kuavo_ws/models/paligemma-3b-pt-224
python kuavo_deploy/src/scripts/script_auto_test.py ...
```

脚本会覆盖 checkpoint 中保存的 tokenizer 路径，使用该本地路径加载。

---

## 5. 执行流程概览

1. **初始化**：解析配置 → 解析 `pretrained_path`、`run_dir`、`output_directory` → 校验路径存在。
2. **加载策略**：`setup_policy(pretrained_path, policy_type, device)` 加载对应策略；`make_pre_post_processors(..., run_dir, preprocessor_overrides)` 从 run_dir 加载 preprocessor，PI0/PI05 会覆盖 tokenizer 路径。
3. **仿真器握手**：脚本等待仿真器调用 `/simulator/init`；若 8 秒内未收到，则假定仿真器不支持 init 协议，自动继续。
4. **Episode 循环**：
   - 每 episode 前：等待 init（或短暂 sleep）→ 调用 `/simulator/reset`；
   - 创建 `KuavoSimEnv`，`env.reset()`；
   - 调用 `/simulator/start`；
   - 步循环：取观测 → PI0/PI05 时删除 depth 键、注入空 `task` → preprocessor → policy.select_action → postprocessor → env.step；
   - 成功判定：`/simulator/success` 收到 `True`，或 `terminated`/`truncated`；
   - 保存 rollout 视频与日志。
5. **汇总**：打印成功率 `success_count / eval_episodes`，结果写入 `evaluation_autotest.log`。

---

## 6. PI0 / PI05 部署适配要点

为与 **RGB-only、无语言** 训练保持一致，部署时对 PI0/PI05 做了如下处理：

1. **裁剪 input_features**：在 `setup_policy()` 中，从 `config.input_features` 中去掉所有含 `"depth"` 的键，使视觉 backbone 仅接收 RGB 图像。
2. **删除 depth 观测**：在每步 `preprocessor()` 前，从 `observation` 中移除 `observation.depth_h`、`observation.depth_l`、`observation.depth_r`。
3. **注入空 task**：tokenizer 需要 `task` 字段。为保持“无语言条件”，在观测中注入 `observation["task"] = [""]`，满足 preprocessor 契约而不引入语义 prompt。
4. **推理设置**：强制 `config.dtype = "float32"`、`config.compile_model = False`，避免精度与编译相关问题；`config.device` 设为部署时的 device。

---

## 7. 控制信号

- **暂停/恢复**：`kill -USR1 <pid>`，脚本会阻塞在 `check_control_signals()` 直到恢复；
- **停止**：`kill -USR2 <pid>`，当前 episode 结束后退出。

脚本启动时会打印当前进程 pid 及上述命令。

---

## 8. 输出内容

- **目录**：`outputs/eval/<task>/<method>/<timestamp>/epoch<epoch>/`（或 `checkpoint_run_dir/eval_output/epoch<epoch>/`）
- **文件**：
  - `evaluation_autotest.log`：评估时间戳、episode 数、逐 episode 成功数累计；
  - `rollout_<episode>_<cam_key>.mp4`：各相机视角的 rollout 视频。

---

## 9. 常见问题

**Q1：仿真器已启动，脚本一直等待 init？**  
**A**：检查 (1) 仿真器与脚本是否使用同一 `ROS_MASTER_URI`；(2) 仿真器是否在就绪时调用 `rosservice call /simulator/init`；(3) 运行 `rosservice list | grep simulator` 确认服务存在。

**Q2：FileNotFoundError: policy_preprocessor.json / epoch 目录不存在？**  
**A**：确认 `run_dir`（或 `checkpoint_run_dir`）路径正确，且内含 `policy_preprocessor.json` 与 `epoch3`（或 `epochbest`）子目录；训练输出结构见 [训练流水线](../training/pipeline.md)。

**Q3：PI0/PI05 推理报错 tokenizer 或 shape 不匹配？**  
**A**：设置 `PALIGEMMA_TOKENIZER_PATH` 指向本地 tokenizer 目录；若使用 `HF_HUB_OFFLINE=1`，需确保该路径存在且包含所需文件。

**Q4：训练为 RGB-only、无语言，部署也要一致吗？**  
**A**：是。训练与部署的输入模态（RGB vs RGB+Depth）和语言条件（有/无）必须一致，否则会产生分布偏移，影响成功率。详见 [策略对比](../training/comparison.md)。

---

## 10. 相关文档

- [快速开始](../getting_started/quick_start.md)：从数据到训练与仿真评估的完整路径
- [训练流水线](../training/pipeline.md)：训练输出目录结构、preprocessor 保存
- [策略概览](../concepts/policy_overview.md)：各策略的输入输出与部署支持
- [策略对比](../training/comparison.md)：训练/部署一致性、成功率评估
