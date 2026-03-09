# 快速开始

本节提供从**环境安装**到**训练与部署**的完整流程概览。详细步骤请参考 [Kuavo Data Challenge GitHub 仓库（dev 分支）](https://github.com/LejuRobotics/kuavo_data_challenge/tree/dev) 的 README。

!!! abstract "本节步骤概览"
    | 步骤 | 内容 |
    |------|------|
    | 1 | 环境要求与安装（系统、Python、ROS、NVIDIA） |
    | 2 | 克隆代码与依赖安装 |
    | 3 | 数据准备（rosbag → LeRobot、quantile） |
    | 4 | 模仿学习训练（ACT / Diffusion / PI0 / PI05） |
    | 5 | 仿真器测试与真机测试 |

    零基础读者建议先阅读 [零基础完整学习教程](beginner_tutorial.md)，再回到本节按命令执行。

---

## 1. 环境要求

- **系统**：推荐 Ubuntu 20.04（22.04 / 24.04 建议使用 Docker）
- **Python**：Python 3.10
- **ROS**：ROS Noetic + Kuavo Robot ROS 补丁
- **依赖**：Docker、NVIDIA CUDA Toolkit（GPU 训练/推理）
- **本分支（dev）**：建议新建独立 conda 环境，如 `kdc_dev`

---

## 2. 安装步骤

### 2.1 克隆代码

```bash
# SSH 或 HTTPS
git clone https://github.com/LejuRobotics/kuavo_data_challenge.git
cd kuavo_data_challenge

# 切换到 dev 分支
git checkout origin/dev

# 更新 third_party 下的 lerobot 子模块
git submodule init
git submodule update --recursive --progress
```

### 2.2 Python 环境

```bash
# 使用 conda（推荐）
conda create -n kdc_dev python=3.10
conda activate kdc_dev

# 或使用 venv
python3.10 -m venv kdc_dev
source kdc_dev/bin/activate
```

### 2.3 安装依赖

```bash
source /opt/ros/noetic/setup.bash  # 若使用 ROS，建议写入 ~/.bashrc

pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple  # 建议换源加速

# 仅训练（无需 ROS）
pip install -r requirements_ilcode.txt

# 完整功能（数据转换、部署，需 ROS Noetic）
pip install -r requirements_total.txt
```

确认 lerobot 版本（当前约 0.4.2）：

```bash
pip show lerobot
```

### 2.4 ROS 与 NVIDIA（可选）

- **ROS Noetic**：仿真与真机部署需安装；详见 [README 安装指南](https://github.com/LejuRobotics/kuavo_data_challenge/tree/dev) 中的「ROS 环境配置」。
- **NVIDIA 驱动与 Docker**：GPU 训练/推理需配置；详见 README「操作系统环境配置」。

---

## 3. 数据准备

### 3.1 数据格式转换（rosbag → LeRobot）

若已有 Kuavo rosbag 数据，需先转换为 LeRobot parquet 格式：

```bash
python kuavo_data/CvtRosbag2Lerobot.py \
  --config-path=../configs/data/ \
  --config-name=KuavoRosbag2Lerobot.yaml \
  rosbag.rosbag_dir=/path/to/rosbag \
  rosbag.lerobot_dir=/path/to/lerobot_data
```

- `rosbag.rosbag_dir`：原始 rosbag 路径  
- `rosbag.lerobot_dir`：转换后 Lerobot 数据保存路径（通常包含 `lerobot` 子目录）  
- 配置文件中可设置启用的相机及是否使用深度图像

### 3.2 数据集结构

转换后应包含：

- `lerobot/meta/info.json`
- `lerobot/meta/stats.json`
- `lerobot/meta/episodes/...`
- `lerobot/data/`、`lerobot/videos/` 等

### 3.3 Quantile 统计（PI0 / PI05 需要）

使用 PI0 或 PI05 且配置为 QUANTILES 归一化时，需先补充分位数统计：

```bash
python third_party/lerobot/src/lerobot/datasets/v30/augment_dataset_quantile_stats.py \
  --repo-id sim_task2_lerobot \
  --root /path/to/sim_task2_lerobot/lerobot
```

---

## 4. 模仿学习训练

### 4.1 使用 train_policy.py（Hydra 配置）

**Diffusion：**

```bash
python kuavo_train/train_policy.py \
  --config-path=../configs/policy/ \
  --config-name=diffusion_config.yaml \
  task=your_task_name \
  method=your_method_name \
  root=/path/to/lerobot_data/lerobot \
  training.batch_size=32 \
  policy_name=diffusion
```

**ACT：**

```bash
python kuavo_train/train_policy.py \
  --config-path=../configs/policy/ \
  --config-name=act_config.yaml \
  task=your_task_name \
  method=your_method_name \
  root=/path/to/lerobot_data/lerobot \
  training.batch_size=32 \
  policy_name=act
```

**PI0 / PI05：** 使用 `lerobot_train.py`，需指定 `--policy.pretrained_path`：

```bash
CUDA_VISIBLE_DEVICES=0 python third_party/lerobot/src/lerobot/scripts/lerobot_train.py \
  --dataset.repo_id=sim_task2_lerobot \
  --dataset.root=/path/to/sim_task2_lerobot/lerobot \
  --policy.type=pi05 \
  --output_dir=./outputs/train/sim_task2/pi05_rgb \
  --job_name=pi05_training \
  --policy.pretrained_path=/path/to/pi05_base \
  --policy.push_to_hub=false \
  --steps=30000 \
  --batch_size=16 \
  --policy.device=cuda
```

### 4.2 单机多卡（Accelerate）

```bash
# 配置 accelerate
vim configs/accelerate/accelerate_config.yaml

# 启动多卡训练
accelerate launch --config_file configs/accelerate/accelerate_config.yaml \
  kuavo_train/train_policy_with_accelerate.py \
  --config-path=../configs/policy \
  --config-name=diffusion_config.yaml \
  task=your_task_name \
  method=your_method_name \
  root=/path/to/lerobot_data/lerobot
```

### 4.3 输出目录

训练完成后，模型与配置保存在：

```
outputs/train/<task>/<method>/run_<timestamp>/
├── epoch1/、epoch2/、...、epochbest/
├── policy_preprocessor.json
└── config.json
```

---

## 5. 仿真器测试

### 5.1 启动 Mujoco 仿真器

仿真器需单独启动，具体步骤见赛事/仿真器相关 README。

### 5.2 配置部署

编辑 `configs/deploy/kuavo_env.yaml`：

- `env.env_name`：`Kuavo-Sim`
- `inference.policy_type`：`diffusion` / `act` / `pi0` / `pi05`
- `inference.task`、`inference.method`、`inference.timestamp`、`inference.epoch`：对应训练输出路径

### 5.3 运行自动测试

```bash
conda activate kdc_dev
cd /path/to/kuavo_data_challenge

python kuavo_deploy/src/scripts/script_auto_test.py \
  --task auto_test \
  --config configs/deploy/kuavo_env.yaml
```

或使用交互式入口（与 main 分支不同）：

```bash
python kuavo_deploy/eval_kuavo.py
```

选择 `3` 指定 `kuavo_env.yaml` 路径，再选择 `8. auto_test` 进行自动测试。

---

## 6. 真机测试

1. 修改 `configs/deploy/kuavo_env.yaml`：`env.env_name` 设为 `Kuavo-Real`，`eef_type`、`obs_key_map` 等按真机配置；
2. 配置 `inference.go_bag_path`：真机需提供预录轨迹用于到达工作姿态；
3. 运行部署脚本，步骤同仿真；
4. 推理日志：`log/kuavo_deploy/kuavo_deploy.log`。

边侧机 / AGX Orin 推理说明见 [README_AGX_ORIN.md](https://github.com/LejuRobotics/kuavo_data_challenge/blob/dev/README_AGX_ORIN.md)。

---

## 7. kuavo_humanoid_sdk 版本匹配

若出现「机械臂初始化失败」等通信错误，需检查 SDK 版本与下位机一致：

```bash
# 下位机查看版本
ssh lab@192.168.26.1
cd ~/kuavo-ros-opensource
git describe --tag  # 如 1.2.2 或 1.3.1

# 本机安装对应版本
pip install kuavo-humanoid-sdk==1.2.2  # 与下位机版本匹配
```

---

## 8. 核心代码结构

```
kuavo_data_challenge/
├── configs/           # 配置文件
├── kuavo_data/        # 数据转换（rosbag → Lerobot）
├── kuavo_deploy/      # 部署脚本（仿真/真机）
├── kuavo_train/       # 模仿学习训练
├── lerobot_patches/   # Lerobot 兼容补丁
├── third_party/       # Lerobot 子模块
└── outputs/           # 训练输出与评估结果
```

---

## 9. 更多文档

- **完整安装与使用**：[GitHub README (dev 分支)](https://github.com/LejuRobotics/kuavo_data_challenge/tree/dev)
- **训练流水线**：[训练流水线](../training/pipeline.md)
- **仿真自动测试**：[仿真自动测试](../deployment/sim_auto_test.md)
- **真机评测**：[真机评测](../deployment/real_eval.md)
- **策略说明**：[策略概览](../concepts/policy_overview.md)
