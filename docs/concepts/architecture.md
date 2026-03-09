# 架构说明

本节从数据流与模块分层两个角度，介绍 Kuavo Data Challenge 的整体架构。

!!! abstract "阅读指引"
    先看**数据流**（ROS/仿真 → LeRobot v4.3），再看**训练与部署**分层及与各策略的对应关系；更多入口与配置见 [训练流水线](../training/pipeline.md)、[配置说明](../training/configs.md)。

---

## 1. Data：从 ROS / 仿真到 LeRobot v4.3

### 1.1 数据来源

- **仿真环境**（sim_task1 / sim_task2 等）：
  - 通过 Kuavo 仿真器提供的 ROS 话题采集：
    - RGB 相机：`/cam_h/color/image_raw/compressed`、`/cam_l/...`、`/cam_r/...`
    - Depth 相机（可选）：`/cam_*/depth/.../compressedDepth`
    - 关节与夹爪状态：`/sensors_data_raw`、`/gripper/state`
- **真机环境**：
  - 同样通过 ROS + Kuavo SDK 发布相同语义的 Topic / Service，以保证仿真与真机数据兼容。

### 1.2 统一为 LeRobot v4.3 数据集

在本项目中，所有训练数据都会被转换为 **LeRobot v4.3** 标准格式：

- 典型目录：
  - `lerobot/meta/info.json`
  - `lerobot/meta/stats.json`
  - `lerobot/meta/episodes/chunk-000/file-000.parquet`
- 统计信息：
  - `stats.json` 中包含状态 / 动作等的 mean/std 或 quantiles；
  - 对于 PI0 / PI05，会使用 quantile 归一化。

数据转换与增强逻辑主要位于：

- `kuavo_data/`：ROS bag 转 LeRobot 的脚本与通用数据处理；
- `third_party/lerobot/datasets/v30/*.py`：LeRobot 官方的数据集工具。

---

## 2. Train：多策略训练层（kuavo_train + LeRobot）

### 2.1 训练入口与第三方框架

- 核心训练入口脚本：
  - `third_party/lerobot/src/lerobot/scripts/lerobot_train.py`
- 利用 LeRobot 的训练框架：
  - 完成数据加载、预处理、优化器 / 学习率调度等通用流程；
  - 通过配置（YAML + CLI）选择不同的 policy 类型。

### 2.2 Kuavo 专用 Wrapper 层

为避免直接修改 LeRobot 源码，本项目在 `kuavo_train/wrapper/` 下增加了一层适配：

- `kuavo_train/wrapper/policy/act/*`：ACT 策略封装；
- `kuavo_train/wrapper/policy/diffusion/*`：Diffusion Policy 封装；
- `kuavo_train/wrapper/policy/pi0/*`：PI0（PaliGemma + Expert）封装；
- `kuavo_train/wrapper/policy/pi05/*`：PI05（OpenPI 风格）封装；
- （预留）Groot / SmolVLA 等策略的封装。

这些 wrapper 负责：

- 映射 Kuavo 观测键（`observation.images.head_cam_h`、`observation.state` 等）到各策略期望的输入；
- 根据配置决定是否启用 depth 分支、如何做 RGB-Depth 融合；
- 注入 Kuavo 特定的预处理逻辑（例如 RGB-only 训练时删除 depth 键）。

### 2.3 推荐训练模式（当前实现）

当前仓库中，针对 PI0 / PI05 的默认实现是：

- **RGB-only 训练**：
  - 在训练循环中统一删除 `observation.depth_*`；
  - 仅使用 RGB 图像 + 状态作为输入模态；
- **量化 / 标准化**：
  - 依赖 LeRobot 的 `stats.json` 与 Processor Pipeline；
  - 对 PI0 / PI05 使用 quantile 归一化。

更多细节见：

- [训练流水线](../training/pipeline.md)
- [配置说明](../training/configs.md)

---

## 3. Deploy：仿真与真机部署层（kuavo_deploy）

### 3.1 仿真部署（Simulation）

相关代码位于 `kuavo_deploy/`：

- 入口脚本：
  - `kuavo_deploy/src/scripts/script_auto_test.py`
- 核心评估逻辑：
  - `kuavo_deploy/src/eval/sim_auto_test.py`

主要流程：

1. 从 `configs/deploy/kuavo_env.yaml` 读取环境与 inference 配置；
2. 解析训练输出目录，加载：
   - 策略 config（`config.json`）；
   - 预处理配置（`policy_preprocessor.json`）；
   - 模型权重（`epochX/model.safetensors`）。
3. 通过 `setup_policy(...)` 统一构建策略实例：
   - 对 PI0 / PI05：部署期裁剪为 RGB-only、关闭 `compile_model`、强制使用 float32；
4. 使用 Gym + ROS：
   - `KuavoBaseRosEnv` 对接 ROS Topic 与 Service；
   - 循环调用 `policy.select_action(observation)` 与 `env.step(action)`；
   - 记录成功率与视频结果。

详见：`deployment/sim_auto_test.md`。

### 3.2 真机部署（Real Robot）

真机相关模块同样在 `kuavo_deploy/` 下：

- `eval_kuavo.py`：真机评估入口脚本；
- `kuavo_service/*`：推理服务端 / 机器人客户端的解耦实现（通常基于 ZeroMQ + ROS）。

职责划分：

- **Policy Server**：
  - 在上位机 / GPU 服务器上运行策略推理；
  - 接收来自机器人端的观测，返回动作；
- **Robot Client**：
  - 在靠近机器人侧的机器上运行；
  - 通过 Kuavo SDK / ROS 将动作转换为关节命令并执行；
  - 处理安全相关逻辑（急停 / 限幅等）。

详见：`deployment/real_eval.md`。

---

## 4. 模块分层小结

- **数据层（Data）**：`kuavo_data/` + `lerobot/`  
  负责从 ROS / 仿真采集数据并转换为 LeRobot 数据集。

- **训练层（Train）**：`third_party/lerobot` + `kuavo_train/`  
  利用 LeRobot 通用训练框架，通过 wrapper 适配 Kuavo 观测与多种 policy。

- **部署层（Deploy）**：`kuavo_deploy/`  
  将训练好的策略部署到仿真与真机环境，通过 ROS / Kuavo SDK 控制机器人。

> 各种 policy 的详细说明见 `concepts/policy_overview.md` 与 `policy/*.md`。  
> 推荐阅读顺序：本页 → `policy_overview.md` → 对应的 `policy/*.md` → `training/` 与 `deployment/` 章节。

