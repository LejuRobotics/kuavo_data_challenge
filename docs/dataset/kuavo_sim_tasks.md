# Kuavo 仿真任务说明

本节描述 Kuavo 仿真任务（sim_task1 / sim_task2 等）在**数据采集、训练与部署**中的角色，以及任务命名、配置与评估流程的对应关系。

---

## 1. 概述

Kuavo 仿真任务指在 **Kuavo-Sim** 仿真环境中定义的操作场景。同一任务名称会出现在：

- **数据集**：转换后的 LeRobot 数据集常以 `sim_taskX_lerobot` 作为 `repo_id`，数据目录如 `sim_task2_lerobot/lerobot`；
- **训练配置**：YAML 或命令行中的 `task`、`method`，以及输出目录 `outputs/train/<task>/<method>/<timestamp>`；
- **部署配置**：`configs/deploy/kuavo_env.yaml` 中的 `inference.task`、`inference.method`、`inference.timestamp`、`inference.epoch`，用于定位要加载的检查点；
- **评估流程**：仿真器通过 ROS 服务（如 `/simulator/reset`、`/simulator/start`、`/simulator/success`）与脚本配合，执行多回合自动测试（`auto_test`）。

环境名 `env_name: Kuavo-Sim` 表示仿真；真机为 `Kuavo-Real`。观测与动作接口（多视角 RGB/深度、关节状态、动作维度）在仿真与真机间保持一致，便于同一策略在仿真验证后迁移到真机。

---

## 2. 常见任务类型

### 2.1 sim_task1

- **定位**：相对简单的仿真场景，常用于算法验证与 smoke test。
- **任务目标**：简单抓取 / 放置等基础操作。
- **数据特点**：轨迹较短，场景相对简单。
- **在配置中的用法**：
  - 数据集：`repo_id` 可为 `sim_task1_lerobot`，`root` 指向对应 `lerobot` 目录；
  - 训练：`task: "sim_task1"`，输出目录示例 `outputs/train/sim_task1/pi0_rgb/...`；
  - 部署：`inference.task: "sim_task1"`，`inference.method` 与 `timestamp`、`epoch` 对应上述训练产出。

### 2.2 sim_task2

- **定位**：更复杂的流水线操作，适合评估大型策略（如 PI0、PI05）的整体性能。
- **任务目标**：抓取 → 称重 → 分拣等多步骤操作。
- **数据特点**：多相机视角（head_cam_h、wrist_cam_l、wrist_cam_r），可选深度（depth_h、depth_l、depth_r）；轨迹与场景更复杂。
- **在配置中的用法**：
  - 数据集：`repo_id` 常为 `sim_task2_lerobot`，如 [快速开始](../getting_started/quick_start.md) 中的示例；
  - 训练：`task: "sim_task2"`，`output_dir=./outputs/train/sim_task2/pi05_rgb`；
  - 部署：`configs/deploy/kuavo_env.yaml` 中默认示例为 `task: "sim_task2"`、`method: "pi05_rgb"`，配合 `timestamp`、`epoch` 指定 run 与检查点。

若赛事或内部约定中引入更多任务（如 sim_task3、sim_task4），其用法与上述一致：用 `task` 区分场景，用 `method` 区分策略或训练配置，用 `repo_id`/`root` 指向对应数据集。

---

## 3. 观测与数据格式

仿真环境通过 ROS 话题发布与训练阶段一致的观测：

- **RGB 图像**：如 `/cam_h/color/image_raw/compressed`、`/cam_l/...`、`/cam_r/...`，对应 LeRobot 中的 `observation.images.head_cam_h`、`observation.images.wrist_cam_l`、`observation.images.wrist_cam_r`；
- **深度图像**（可选）：如 `/cam_*/depth/.../compressedDepth`，对应 `observation.depth_h`、`observation.depth_l`、`observation.depth_r`；
- **状态**：关节与夹爪状态（如 `/sensors_data_raw`、`/gripper/state`），对应 `observation.state`。

部署配置中的 `env.obs_key_map`、`env.image_size`、`env.depth_range` 等需与训练时使用的特征一致。数据集格式详见 [LeRobot v4.3 数据集说明](lerobot_v30.md)。

---

## 4. 部署与评估流程

### 4.1 环境与配置

- **环境名称**：`configs/deploy/kuavo_env.yaml` 中 `env.env_name: Kuavo-Sim`；
- **推理配置**：`inference` 下的 `policy_type`、`task`、`method`、`timestamp`、`epoch`、`max_episode_steps`、`eval_episodes`、`seed` 等。

若使用非默认输出目录的检查点，可设置 `inference.checkpoint_run_dir` 与 `inference.epoch`，详见 [仿真自动测试](../deployment/sim_auto_test.md)。

### 4.2 自动测试（auto_test）

- **入口**：`python kuavo_deploy/src/scripts/script_auto_test.py --task auto_test --config configs/deploy/kuavo_env.yaml`；
- **流程**：加载配置 → 根据 `task`/`method`/`timestamp`/`epoch` 或 `checkpoint_run_dir` 加载策略 → 创建 `Kuavo-Sim` 环境 → 循环执行 `eval_episodes` 个 episode；每轮调用仿真器 reset、start，通过 `/simulator/success` 等判定成功与否；
- **结果**：成功次数、日志与可选保存的观测/轨迹，用于比赛打分或离线分析。

---

## 5. 配置与命名小结

| 用途         | 典型字段 / 位置                     | 示例值                    |
|--------------|-------------------------------------|---------------------------|
| 数据集标识   | `dataset.repo_id` / 数据根目录名    | `sim_task2_lerobot`       |
| 数据集路径   | `dataset.root`                      | `/path/to/sim_task2_lerobot/lerobot` |
| 训练任务名   | 配置中 `task`、`--output_dir` 路径  | `sim_task2`               |
| 训练方法名   | 配置中 `method`、输出子目录         | `pi05_rgb`                |
| 部署任务名   | `inference.task`                    | `sim_task2`               |
| 部署方法名   | `inference.method`                  | `pi05_rgb`                |
| 部署 run     | `inference.timestamp` 或 `checkpoint_run_dir` | `run_20260127_011739`     |
| 部署检查点   | `inference.epoch`                   | `3` 或 `best`             |
| 仿真环境名   | `env.env_name`                      | `Kuavo-Sim`               |

实际任务定义、物体布局与成功判定由仿真器内部实现；本仓库负责数据格式、训练脚本与部署脚本的对接。更多细节请参考：

- [快速开始](../getting_started/quick_start.md)
- [仿真自动测试](../deployment/sim_auto_test.md)
- [LeRobot v4.3 数据集说明](lerobot_v30.md)
- [训练流水线](../training/pipeline.md)、[配置说明](../training/configs.md)
