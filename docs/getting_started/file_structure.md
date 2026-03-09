# 项目文件构成说明

本文档帮助读者快速了解 Kuavo Data Challenge 仓库的目录与关键文件，便于定位配置、训练、部署相关代码。

!!! note "使用建议"
    可按「顶层结构 → 各目录说明」顺序阅读；需要查具体配置文件或脚本时，直接使用文内表格定位路径。

---

## 1. 顶层结构一览

```
kuavo_data_challenge/
├── configs/                 # 各类 YAML 配置
├── docs/                    # 本在线文档（MkDocs）
├── docker/                  # Docker 相关
├── kuavo_data/              # 数据转换（rosbag → LeRobot）
├── kuavo_deploy/            # 部署与评测（仿真/真机）
├── kuavo_train/             # 训练入口与策略封装
├── lerobot/                 # 转换后的 LeRobot 数据集（本地，可选）
├── lerobot_patches/         # 对 LeRobot 的补丁/扩展
├── scripts/                 # 通用脚本（GPU、内存等）
├── third_party/             # 第三方依赖（含 LeRobot 源码）
├── mkdocs.yml               # 文档站点配置
├── setup.py                 # 包安装
├── install_*.sh             # 环境安装脚本
└── README.md
```

---

## 2. 各目录说明与文件清单

### 2.1 `configs/` — 配置

| 路径 | 说明 |
|------|------|
| `configs/accelerate/` | Accelerate 多卡/分布式配置 |
| `configs/accelerate/accelerate_config.yaml` | 多 GPU 训练配置 |
| `configs/data/` | 数据转换配置 |
| `configs/data/KuavoRosbag2Lerobot.yaml` | rosbag → LeRobot 转换参数 |
| `configs/deploy/` | 部署与推理配置 |
| `configs/deploy/kuavo_env.yaml` | **主部署配置**：任务、策略、checkpoint、推理参数、仿真/真机开关等 |
| `configs/deploy/others_env.yaml` | 其他环境示例 |
| `configs/policy/` | 各策略训练配置 |
| `configs/policy/act_config.yaml` | ACT 策略配置 |
| `configs/policy/diffusion_config.yaml` | Diffusion 策略配置 |
| `configs/policy/pi0_config*.yaml` | PI0 系列（含 RGB、RGB-Depth、CrossAttn 等） |
| `configs/policy/pi05_config*.yaml` | PI05 系列（含 RGB-Depth、CrossAttn 等） |

训练时通过 `--config configs/policy/xxx_config.yaml` 指定；部署时通过 `--config configs/deploy/kuavo_env.yaml` 指定。

---

### 2.2 `kuavo_train/` — 训练

| 路径 | 说明 |
|------|------|
| `kuavo_train/train_policy.py` | **统一训练入口**（ACT / Diffusion / PI0 / PI05） |
| `kuavo_train/train_policy_with_accelerate.py` | 使用 Accelerate 的多卡训练 |
| `kuavo_train/train_policy_with_ray.py` | 使用 Ray 的分布式训练 |
| `kuavo_train/README.md` | 训练模块说明 |
| `kuavo_train/utils/` | 训练用工具函数 |
| `kuavo_train/wrapper/` | 对数据集与策略的封装，供 `train_policy.py` 使用 |
| `kuavo_train/wrapper/dataset/LeRobotDatasetWrapper.py` | LeRobot 数据集封装 |
| `kuavo_train/wrapper/processor/` | 预处理器相关封装 |
| `kuavo_train/wrapper/policy/` | 各策略的 Config/Model/Policy 封装 |
| `kuavo_train/wrapper/policy/act/` | ACT：`ACTConfigWrapper`、`ACTModelWrapper`、`ACTPolicyWrapper` |
| `kuavo_train/wrapper/policy/diffusion/` | Diffusion：Config/Model/Policy 封装及 DiT、Transformer 实现 |
| `kuavo_train/wrapper/policy/pi0/` | PI0：`PI0ConfigWrapper`、`PI0ModelWrapper`、`PI0PolicyWrapper` |
| `kuavo_train/wrapper/policy/pi05/` | PI05：`PI05ConfigWrapper`、`PI05ModelWrapper`、`PI05PolicyWrapper` |

策略实现本身在 `third_party/lerobot` 中；本目录负责接入 Kuavo 数据与部署约定。

---

### 2.3 `kuavo_deploy/` — 部署与评测

| 路径 | 说明 |
|------|------|
| `kuavo_deploy/config.py` | 加载 `kuavo_env.yaml`、解析任务与 checkpoint 等 |
| `kuavo_deploy/eval_kuavo.py` | Kuavo 仿真/真机评测入口（可被脚本调用） |
| `kuavo_deploy/eval_others.py` | 其他环境评测入口 |
| `kuavo_deploy/readme.md` | 部署模块说明 |
| `kuavo_deploy/readme/` | 部署子文档（inference、环境、机器人连接等） |
| `kuavo_deploy/utils/` | 部署用工具 |
| `kuavo_deploy/kuavo_env/` | 环境抽象（仿真/真机） |
| `kuavo_deploy/kuavo_env/KuavoBaseRosEnv.py` | 基于 ROS 的基类环境 |
| `kuavo_deploy/kuavo_env/KuavoSimEnv.py` | 仿真环境 |
| `kuavo_deploy/kuavo_env/KuavoRealEnv.py` | 真机环境 |
| `kuavo_deploy/kuavo_service/` | 可选：与服务端通信的 client/server、bag 测试 |
| `kuavo_deploy/src/eval/` | 实际跑评测的脚本 |
| `kuavo_deploy/src/eval/sim_auto_test.py` | **仿真自动测试**（按任务重置、推理、记录结果） |
| `kuavo_deploy/src/eval/real_single_test.py` | **真机单次评测** |
| `kuavo_deploy/src/scripts/` | 与评测配合的脚本 |
| `kuavo_deploy/src/scripts/controller.py` | 运行时控制（暂停/恢复/停止等） |
| `kuavo_deploy/src/scripts/script.py` | 真机任务脚本（go / run / go_run 等） |
| `kuavo_deploy/src/scripts/script_auto_test.py` | 自动测试脚本 |

仿真评测常用：`configs/deploy/kuavo_env.yaml` + `kuavo_deploy/src/eval/sim_auto_test.py`；真机常用：同一 config + `real_single_test.py` 与 `script.py`。

---

### 2.4 `kuavo_data/` — 数据准备

| 路径 | 说明 |
|------|------|
| `kuavo_data/CvtRosbag2Lerobot.py` | **主入口**：将 Kuavo rosbag 转为 LeRobot 格式数据集 |
| `kuavo_data/common/` | 转换用公共逻辑 |
| `kuavo_data/common/config_dataset.py` | 数据集配置 |
| `kuavo_data/common/kuavo_dataset.py` | Kuavo 数据集读取 |
| `kuavo_data/common/ros_handler.py` | ROS 消息处理 |
| `kuavo_data/common/utils.py` | 工具函数 |

转换时使用 `configs/data/KuavoRosbag2Lerobot.yaml`；输出目录通常为项目下的 `lerobot/` 或自定义路径。

---

### 2.5 `scripts/` — 通用脚本

| 路径 | 说明 |
|------|------|
| `scripts/check_gpu_processes.sh` | 查看 GPU 占用进程 |
| `scripts/clear_gpu_memory.py` | 清理 GPU 显存（调试用） |

---

### 2.6 `docker/` — Docker

| 路径 | 说明 |
|------|------|
| `docker/readme.md` | Docker 使用说明 |
| `docker/run_with_gpu.sh` | 带 GPU 的容器运行示例 |

---

### 2.7 `lerobot_patches/` — 对 LeRobot 的修改

| 路径 | 说明 |
|------|------|
| `lerobot_patches/custom_patches.py` | 对第三方 LeRobot 的补丁或扩展逻辑 |

用于在不直接改 `third_party/lerobot` 的前提下扩展行为。

---

### 2.8 `third_party/lerobot/` — LeRobot 源码

| 路径 | 说明 |
|------|------|
| `third_party/lerobot/src/lerobot/` | LeRobot 核心代码 |
| `third_party/lerobot/src/lerobot/scripts/lerobot_train.py` | LeRobot 官方训练入口（本项目统一用 `kuavo_train/train_policy.py`） |
| `third_party/lerobot/src/lerobot/policies/` | 策略实现（ACT、Diffusion、PI0、PI05 等） |
| `third_party/lerobot/src/lerobot/processor/` | 处理器（tokenizer、normalizer 等） |

策略与数据格式的权威实现在此；`kuavo_train/wrapper` 在此基础上做封装与配置映射。

---

### 2.9 `docs/` — 在线文档

| 路径 | 说明 |
|------|------|
| `docs/index.md` | 文档首页 |
| `docs/mkdocs.yml` | 文档导航（与根目录 `mkdocs.yml` 可能为同一或链接） |
| `docs/getting_started/` | 安装、快速开始、**本文件构成说明** |
| `docs/concepts/` | 架构、策略概览 |
| `docs/dataset/` | 数据集与任务说明 |
| `docs/policy/` | 各策略详细说明（ACT、Diffusion、PI0、PI05 等） |
| `docs/training/` | 训练流程、配置、对比、多 GPU |
| `docs/deployment/` | 仿真自动测试、真机评测、ROS |
| `docs/advanced/` | RGB-Depth 融合、Depth 支持细节 |
| `docs/faq.md` | 常见问题 |

---

## 3. 按用途快速查找

| 你想做的事 | 主要涉及的文件/目录 |
|------------|----------------------|
| 把 rosbag 转成训练数据 | `kuavo_data/CvtRosbag2Lerobot.py`、`configs/data/KuavoRosbag2Lerobot.yaml` |
| 改训练参数（学习率、batch 等） | `configs/policy/*.yaml`，以及 `kuavo_train/train_policy.py` 的参数 |
| 跑 ACT/Diffusion/PI0/PI05 训练 | `kuavo_train/train_policy.py` + 对应 `configs/policy/xxx_config.yaml` |
| 改部署任务、checkpoint、推理设置 | `configs/deploy/kuavo_env.yaml`、`kuavo_deploy/config.py` |
| 跑仿真自动测试 | `kuavo_deploy/src/eval/sim_auto_test.py` + `configs/deploy/kuavo_env.yaml` |
| 跑真机评测 | `kuavo_deploy/src/eval/real_single_test.py`、`kuavo_deploy/src/scripts/script.py` |
| 理解策略结构或改策略封装 | `kuavo_train/wrapper/policy/{act,diffusion,pi0,pi05}/` |
| 理解策略底层实现 | `third_party/lerobot/src/lerobot/policies/` |
| 查报错或常见问题 | `docs/faq.md` |
| 从零开始跑通流程 | `docs/getting_started/quick_start.md`、`docs/getting_started/installation.md` |

---

## 4. 相关文档

- [快速开始](quick_start.md) — 环境、数据、训练、仿真/真机一条龙
- [安装指南](installation.md) — 安装与依赖
- [训练流水线](../training/pipeline.md) — 训练入口与各策略调用方式
- [仿真自动测试](../deployment/sim_auto_test.md) — 仿真评测配置与运行
- [真机评测](../deployment/real_eval.md) — 真机部署与 SDK 版本
- [常见问题](../faq.md) — 常见问题汇总
