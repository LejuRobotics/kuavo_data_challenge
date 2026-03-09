# 安装指南

本文档给出**最低环境要求**与**创建 Conda 环境、安装项目**的简要步骤。完整流程（含 ROS、Docker、数据与训练）见 [快速开始](quick_start.md) 与 [零基础完整学习教程](beginner_tutorial.md)。

!!! abstract "简要说明"
    满足环境要求后，按下面步骤创建环境并安装依赖；详细训练与部署步骤见 [训练流水线](../training/pipeline.md) 与 [仿真自动测试](../deployment/sim_auto_test.md)。

---

## 环境要求

- **系统**：Ubuntu 20.04 / 22.04（推荐）
- **Python**：>= 3.10
- **CUDA**：>= 11.7（训练 / 推理需要 GPU 时）
- **ROS**：Noetic / Melodic（部署 Kuavo 机器人或做 rosbag 转换时需要）

---

## 创建 Conda 环境

```bash
conda create -n kdc python=3.10 -y
conda activate kdc
```

---

## 安装依赖与项目

```bash
cd /path/to/kuavo_data_challenge
pip install -e .
```

按需选择依赖：

- **仅训练**（已有 LeRobot 数据）：`pip install -r requirements_ilcode.txt`
- **含数据转换 / 仿真 / 真机**：先 `source /opt/ros/noetic/setup.bash`，再 `pip install -r requirements_total.txt`

详见 [零基础完整学习教程](beginner_tutorial.md) 第四部分。
