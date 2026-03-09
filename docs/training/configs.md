# 训练配置说明

本文说明使用**转换好的 LeRobot 数据**进行模仿学习训练时的**配置方式**与**常用参数**。策略相关 YAML 位于项目 `configs/policy/` 下，可按需修改或通过命令行覆盖。

!!! abstract "说明"
    训练入口为 `kuavo_train/train_policy.py`，通过 `--config-path`、`--config-name` 指定策略配置；常用参数（task、method、root、batch_size、policy_name）可在命令行传入，**推荐直接修改对应 YAML 文件**，避免命令行输入错误。多卡训练见 [多卡训练](multi_gpu.md)。

---

## 1. 单卡训练命令示例

使用转换好的数据进行模仿学习训练：

```bash
python kuavo_train/train_policy.py \
  --config-path=../configs/policy/ \
  --config-name=diffusion_config.yaml \
  task=your_task_name \
  method=your_method_name \
  root=/path/to/lerobot_data/lerobot \
  training.batch_size=128 \
  policy_name=diffusion
```

将 `your_task_name`、`your_method_name`、`/path/to/lerobot_data/lerobot` 等替换为你的实际任务名、方法名与数据路径。

---

## 2. 主要参数说明

| 参数 | 说明 | 示例 |
|------|------|------|
| **task** | 自定义任务名称，建议与数据转换时的 task 定义对应 | `pick_and_place`、`sim_task2` |
| **method** | 自定义方法名，用于区分不同训练实验 | `diffusion_bs128`、`diffusion_bs128_usedepth_nofuse`、`act_rgb` |
| **root** | 训练数据的本地路径，**需包含末尾的 `lerobot`**，与数据转换保存路径对应 | `/path/to/lerobot_data/lerobot` |
| **training.batch_size** | 批大小，可根据 GPU 显存调整 | `32`、`64`、`128` |
| **policy_name** | 使用的策略，用于策略实例化；目前支持 | `diffusion`、`act`、`pi0`、`pi05` |

- **其他参数**（如学习率、epoch、数据增广等）详见各策略 YAML 文件；**推荐直接修改 YAML**，避免命令行过长或输错。
- 若使用 **PI0 / PI05**，需在配置或命令行中设置 `policy.pretrained_path`（如 `lerobot/pi0_base`、`lerobot/pi05_base` 或本地路径），详见 [训练流水线](pipeline.md) 与各策略文档。

---

## 3. 项目中的配置文件位置

| 目录/文件 | 说明 |
|-----------|------|
| **configs/policy/** | 各策略训练配置 |
| `configs/policy/diffusion_config.yaml` | Diffusion 策略配置 |
| `configs/policy/act_config.yaml` | ACT 策略配置 |
| `configs/policy/pi0_rgb_config.yaml`、`pi0_config.yaml` 等 | PI0 系列配置 |
| `configs/policy/pi05_config.yaml`、`pi05_rgb_depth_config.yaml` 等 | PI05 系列配置 |
| **configs/accelerate/** | 多卡训练时使用的 Accelerate 配置 |
| `configs/accelerate/accelerate_config.yaml` | 多 GPU 进程与设备配置 |
| **configs/data/** | 数据转换配置 |
| `configs/data/KuavoRosbag2Lerobot.yaml` | rosbag → LeRobot 转换参数 |
| **configs/deploy/** | 部署与推理配置 |
| `configs/deploy/kuavo_env.yaml` | 仿真/真机部署主配置 |

修改训练相关参数时，以 `configs/policy/*.yaml` 为准；单卡训练用 `train_policy.py`，多卡训练用 `train_policy_with_accelerate.py` 并配合 `configs/accelerate/accelerate_config.yaml`，详见 [多卡训练](multi_gpu.md)。

---

## 4. 相关文档

- [训练流水线](pipeline.md)：入口脚本、数据流与各策略命令行示例（即「模仿学习训练」详细参数说明）
- [多卡训练](multi_gpu.md)：多卡训练与 Accelerate 配置
- [训练策略对比](comparison.md)：各策略对比与选型
