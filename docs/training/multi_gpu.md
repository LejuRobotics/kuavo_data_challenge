# 多卡训练

本文说明使用 **Accelerate** 进行多 GPU 并行训练的配置与运行方式。单卡训练与配置参数说明见 [训练流水线](pipeline.md) 与 [配置说明](configs.md)。

!!! abstract "前置条件"
    需已安装 **accelerate** 库（安装 LeRobot 时通常已附带）。若未安装，执行：`pip install accelerate`。多卡训练与单卡使用同一套策略配置（如 `diffusion_config.yaml`），仅启动方式改为 `accelerate launch`。

---

## 1. 配置 Accelerate YAML

根据本机 GPU 数量与编号，编辑：

```bash
vim configs/accelerate/accelerate_config.yaml
```

（或使用其他编辑器打开 `configs/accelerate/accelerate_config.yaml`。）

**主要字段说明**（按需修改）：

| 字段 | 说明 | 示例 |
|------|------|------|
| `distributed_type` | 分布式类型 | `MULTI_GPU`（单机多卡） |
| `num_processes` | 使用的进程数（一般等于使用的 GPU 数） | `2`、`4` |
| `gpu_ids` | 使用的 GPU 编号，逗号分隔 | `"0,1"`、`"6,7"` |
| `fp16` | 是否使用 FP16 混合精度 | `true` / `false` |

示例（使用 2 张 GPU，编号 6 和 7）：

```yaml
compute_environment: LOCAL_MACHINE
distributed_type: MULTI_GPU
fp16: true
machine_rank: 0
main_process_ip: null
main_process_port: null
main_training_function: main
num_machines: 1
num_processes: 2
gpu_ids: "6,7"
```

---

## 2. 安装 accelerate（如未安装）

一般安装 LeRobot 时已安装；若未安装，执行：

```bash
pip install accelerate
```

---

## 3. 运行多卡训练示例

配置好 `configs/accelerate/accelerate_config.yaml` 后，在**项目根目录**下执行：

```bash
accelerate launch --config_file configs/accelerate/accelerate_config.yaml \
  kuavo_train/train_policy_with_accelerate.py \
  --config-path=../configs/policy \
  --config-name=diffusion_config.yaml \
  task=your_task_name \
  method=your_method_name \
  root=/path/to/lerobot_data/lerobot \
  training.batch_size=32 \
  policy_name=diffusion
```

将 `your_task_name`、`your_method_name`、`/path/to/lerobot_data/lerobot` 等替换为你的实际值；`--config-name` 可改为 `act_config.yaml`、`pi0_rgb_config.yaml`、`pi05_config.yaml` 等以训练不同策略。

!!! note "策略配置参数"
    `diffusion_config.yaml`（及其他 `configs/policy/*.yaml`）中的**详细参数**（如 task、method、root、batch_size、学习率、数据增广等）说明见 [训练流水线](pipeline.md) 的「入口脚本与配置」与 [配置说明](configs.md) 的「主要参数说明」。推荐直接修改 YAML 文件，再在命令行中只覆盖少量参数（如 task、method、root）。

---

## 4. 单卡与多卡对比

| 方式 | 命令 | 说明 |
|------|------|------|
| **单卡** | `python kuavo_train/train_policy.py --config-path=... --config-name=... task=... method=... root=...` | 通过 `CUDA_VISIBLE_DEVICES` 指定卡号；见 [配置说明](configs.md) |
| **多卡** | `accelerate launch --config_file configs/accelerate/accelerate_config.yaml kuavo_train/train_policy_with_accelerate.py ...` | 由 `accelerate_config.yaml` 指定 GPU 与进程数；策略配置与单卡相同 |

---

## 5. 相关文档

- [训练流水线](pipeline.md)：统一入口、数据流与各策略示例
- [配置说明](configs.md)：task / method / root / batch_size / policy_name 等参数说明与配置文件位置
