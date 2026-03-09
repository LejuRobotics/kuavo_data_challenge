# 策略概览

本节对项目中支持的主要策略进行整体介绍与对比，便于按需求选型并衔接训练与部署。

---

## 1. 概述

Kuavo Data Challenge 在 **LeRobot** 训练框架之上，通过 `kuavo_train/wrapper/policy/` 下的封装层，适配多种策略以支持：

- **多模态输入**：RGB 图像、可选深度图、机器人状态、可选语言指令；
- **统一数据格式**：LeRobot v4.3 数据集与 `stats.json` / quantile 统计；
- **仿真与真机部署**：同一套权重可在仿真（`sim_auto_test`）与真机（`real_single_test`）中加载。

策略类型由配置中的 `policy_type` 指定（如 `diffusion`、`act`、`pi0`、`pi05`），训练时则通过 `--policy.type` 选择。

---

## 2. 策略对比简表

| 策略 | 输入模态 | 输出形式 | 部署支持 | 典型用途 |
|------|----------|----------|----------|----------|
| **ACT** | RGB（+ 可选 Depth）、状态 | Action Chunk（一段动作序列） | 是 | 通用模仿学习，VAE 隐空间 |
| **Diffusion** | RGB（+ 可选 Depth）、状态 | Action Chunk（扩散去噪生成） | 是 | 多模态、高维动作分布 |
| **PI0** | RGB（+ 可选 Depth）、状态、可选语言 | 动作序列（Transformer 解码） | 是 | 大视觉语言模型 + 专家头，可弱语言/无语言 |
| **PI05** | 同 PI0 | 同 PI0（OpenPI 风格） | 是 | 与 PI0 类似，不同底座与接口 |
| **Groot** | 多模态 VLA | 动作 / 决策 | 文档/预留 | 多任务、具身推理 |
| **SmolVLA** | 轻量 VLA | 动作 | 文档/预留 | 轻量化 VLA 部署 |

**说明**：当前仿真与真机部署脚本中已接通的为 **diffusion、act、pi0、pi05**；Groot、SmolVLA 在文档与结构中预留，具体训练/部署以各 policy 文档为准。

---

## 3. 各策略简介

### 3.1 ACT（Action Chunking Transformer）

- **核心思想**：用 Transformer 编码观测（多视角图像 + 状态，可选深度），在动作空间预测**整段动作序列**（action chunk），可选 VAE 学习动作分布。
- **输入**：`image_features`（RGB）、可选 `depth_features`、可选 `env_state_feature`；由 `CustomACTConfigWrapper` 的 `input_features` 决定。
- **输出**：`(B, chunk_size, action_dim)` 动作块；推理时按步取用，与 Diffusion 类似。
- **训练/部署**：LeRobot 标准 ACT 流程 + Kuavo 多相机/深度封装；部署见 `sim_auto_test`、`real_single_test`，`policy_type=act`。
- **详见**：[ACT 策略说明](../policy/act.md)。

### 3.2 Diffusion Policy

- **核心思想**：将动作序列视为扩散过程的目标，通过去噪网络生成 action chunk；支持多视角图像与可选深度融合。
- **输入**：RGB `image_features`、可选 `depth_features`、状态；推理前常做 crop/resize 等预处理。
- **输出**：Action chunk，通过内部 queue 按步输出单步动作。
- **训练/部署**：Wrapper 位于 `kuavo_train/wrapper/policy/diffusion/`；部署时 `policy_type=diffusion`。
- **详见**：[Diffusion Policy 说明](../policy/diffusion.md)。

### 3.3 PI0（PaliGemma + Expert）

- **核心思想**：基于视觉-语言模型（如 PaliGemma）编码图像与可选语言，再通过专家头在状态与动作空间进行模仿学习；支持纯 RGB 或 RGB+Depth。
- **输入**：多视角 RGB（+ 可选深度）、机器人状态、可选 `language_instruction`（部署时可置空实现“无语言”）。
- **输出**：解码得到的动作序列；使用 quantile 归一化时依赖数据集的 quantile 统计。
- **训练/部署**：当前默认 **RGB-only**、弱语言/无语言；部署时在观测中注入空字符串 `task` 以满足处理器契约，且会去掉 depth 观测键。`policy_type=pi0`。
- **详见**：[PI0 策略说明](../policy/pi0.md)。

### 3.4 PI05（OpenPI 风格）

- **核心思想**：与 PI0 同属“大模型 + 专家头”范式，接口与配置为 OpenPI 风格；同样支持 RGB-only 或 RGB+Depth，以及可选语言。
- **输入/输出**：与 PI0 类似；配置与 backbone 存在差异。
- **训练/部署**：同样支持 RGB-only、无语义 prompt；部署时 `policy_type=pi05`，同样会过滤 depth、注入空 `task`。
- **详见**：[PI05 策略说明](../policy/pi05.md)。

### 3.5 Groot

- **定位**：面向多任务与具身推理的 VLA 策略。
- **当前状态**：在文档与 nav 中预留；具体模型结构、输入输出及训练/部署流程见专门文档。
- **详见**：[Groot 策略说明](../policy/groot.md)。

### 3.6 SmolVLA

- **定位**：轻量化 Vision-Language-Action 模型，便于资源受限场景。
- **当前状态**：在文档与结构中预留；具体说明见专门文档。
- **详见**：[SmolVLA 策略说明](../policy/smolvla.md)。

---

## 4. 部署中的策略选择

在 `kuavo_deploy` 中：

- **仿真**：`kuavo_deploy/src/eval/sim_auto_test.py` 通过 `cfg.policy_type` 调用 `setup_policy()`，支持 `diffusion`、`act`、`pi0`、`pi05`、`client`（仅客户端模式，不加载本地策略）。
- **真机**：`real_single_test.py` 同样通过 `policy_type` 加载对应策略；PI0/PI05 会做 tokenizer 路径覆盖与 `task` 占位处理。
- **配置**：`kuavo_deploy/config.py` 中 `ConfigInference.policy_type` 默认 `"diffusion"`，可改为 `"act"`、`"pi0"`、`"pi05"` 以匹配当前权重。

训练时使用 `third_party/lerobot` 的 `lerobot_train.py`，通过 `--policy.type=act|diffusion|pi0|pi05` 等选择策略类型，并与 `kuavo_train/wrapper` 中的封装配合。

---

## 5. 建议阅读顺序

1. **先读本节**：了解各策略的定位、输入输出形式及部署支持。
2. **按需阅读具体策略**：在 [策略](../policy/act.md) 下打开对应 `policy/*.md`（如 ACT、Diffusion、PI0、PI05）。
3. **训练与部署流程**：见 [训练流水线](../training/pipeline.md)、[配置说明](../training/configs.md)、[仿真自动测试](../deployment/sim_auto_test.md)、[真机评测](../deployment/real_eval.md)。
