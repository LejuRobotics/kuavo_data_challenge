# 更新与新增功能

本文说明本项目相较于初始版本所增加的主要功能与改进，便于读者快速了解当前代码库的能力与文档入口。

!!! abstract "说明"
    以下每项为**独立功能模块**，可点击「详见」中的链接进入对应文档查看实现与配置细节。

---

## 1. ACT / Diffusion 策略的深度图像支持

!!! info "功能简述"
    为 **ACT** 与 **Diffusion Policy** 增加了对深度图像（depth）的完整支持，在原有 RGB 多视角基础上，可为每个相机提供一种 **RGB 与 Depth 的融合方式**，提升在复杂光照或几何场景下的表现。

- **ACT**：独立 ResNet 编码深度图（1 通道），再通过 **Cross-Modal Attention** 与同相机的 RGB 特征做双向融合，融合结果与状态、latent 等一起送入 Transformer。
- **Diffusion**：同样使用独立深度编码器，通过 **multimodalfuse**（双向 Cross-Attention + concat）将 RGB 与 Depth 特征融合后作为全局条件参与去噪。

**详见**：

- [ACT 策略说明](policy/act.md)（含 depth 配置与融合说明）
- [Diffusion Policy 说明](policy/diffusion.md)（含 depth 与 multimodalfuse）
- [RGB-Depth 融合方案](advanced/rgb_depth_fusion.md)（各策略融合方式对比与实现细节）
- [Depth 支持内部说明](advanced/depth_support.md)（数据与预处理、各 policy 的 depth 配置）

---

## 2. 多卡并行加速

!!! info "功能简述"
    支持使用 **Accelerate** 进行多 GPU 并行训练，显著缩短 ACT、Diffusion、PI0、PI05 等策略在大规模数据上的训练时间。

- 配置位于 `configs/accelerate/accelerate_config.yaml`。
- 入口脚本：`kuavo_train/train_policy_with_accelerate.py`，可与 `train_policy.py` 的同一套 policy 配置配合使用。
- 支持数据并行与混合精度，便于在 2/4/8 卡等环境下扩展。

**详见**：[多卡并行加速](training/multi_gpu.md)。

---

## 3. LeRobot 最新版本 v0.4.3

!!! info "功能简述"
    项目已跟进 **LeRobot 最新版本 v0.4.3**，数据集格式、训练管线与策略实现均与该版本对齐。

- 数据集采用 LeRobot v4.3 约定：`meta/info.json`、`meta/stats.json`、`meta/episodes/`、`data/`、`videos/` 等目录与字段与官方一致。
- 策略与处理器（tokenizer、normalizer、quantile 等）与 LeRobot 0.4.3 兼容，便于复用官方脚本与生态。

**详见**：[LeRobot v4.3 数据集说明](dataset/lerobot_v30.md)、[训练流程与入口](training/pipeline.md)。

---

## 4. LeRobot 帧对齐与目录/文件结构

!!! info "功能简述"
    在数据转换与训练管线中支持 **LeRobot 帧对齐** 与清晰的 **目录/文件结构**，保证多相机图像、深度、状态与动作在时间维上一致，并便于按 chunk 流式读取。

- 转换脚本 `kuavo_data/CvtRosbag2Lerobot.py` 按 LeRobot 规范生成 `meta/`、`data/`、`videos/`，各特征在 `frame_index` / `timestamp` 上对齐。
- 数据集目录结构、`info.json` 中的 `data_path` / `video_path` 模板及 `features` 定义均遵循 LeRobot 约定，便于与官方工具和本仓库训练脚本对接。

**详见**：

- [LeRobot v4.3 数据集说明](dataset/lerobot_v30.md)（目录结构、meta、帧索引）
- [项目文件构成说明](getting_started/file_structure.md)（整体目录与数据/训练/部署文件对应关系）

---

## 5. 末端增量式控制

!!! info "功能简述"
    在部署与机器人控制层面支持 **末端执行器（end-effector）的增量式控制**，使策略输出可以以相对位移/增量的形式作用于末端，更适合实际机器人的伺服与安全约束。

- 与动作空间定义、归一化及部署脚本中的动作解析配合，可将网络输出的动作解释为增量指令并下发到底层控制。
- 具体接口与参数依赖 `kuavo_deploy` 与底层 SDK/ROS 的配置，可在部署配置中指定控制模式（绝对/增量）。

**详见**：[仿真自动测试](deployment/sim_auto_test.md)、[真机评测](deployment/real_eval.md) 中的控制与动作相关说明。

---

## 6. 更多模仿学习与 VLA 模型（PI0、PI05、Groot、SmolVLA）

!!! info "功能简述"
    在原有 ACT、Diffusion 之外，集成了更多 **模仿学习** 与 **VLA（Vision-Language-Action）** 类策略，覆盖从轻量到大规模多模态的不同需求。

| 模型 | 类型 | 简要说明 |
|------|------|----------|
| **PI0** | 模仿学习 + VLM | 基于 PaliGemma 视觉编码器 + 专家头，支持多视角 RGB、可选深度与可选语言；支持无语言/弱语言部署，可选 Cross-Attention 融合 Depth。 |
| **PI05** | 模仿学习 + VLM | OpenPI 风格，与 PI0 类似但底座与接口不同；支持 `freeze_vision_encoder`、`train_expert_only` 等训练选项，可选 RGB-Depth Cross-Attention。 |
| **Groot** | 多模态 VLA | 面向多任务与具身推理的多模态 VLA，文档与结构中已预留，具体训练/部署以 Groot 策略文档为准。 |
| **SmolVLA** | 轻量 VLA | 轻量化 VLA 模型，便于在资源受限设备上部署，文档与结构中已预留。 |

- 训练统一入口：`kuavo_train/train_policy.py`，通过 `--policy.type` 选择 `act`、`diffusion`、`pi0`、`pi05` 等。
- 部署：仿真（`sim_auto_test`）与真机（`real_single_test`）已接通 **ACT、Diffusion、PI0、PI05**；Groot、SmolVLA 的完整流程见各策略文档。

**详见**：

- [策略概览](concepts/policy_overview.md)
- [PI0 策略说明](policy/pi0.md)
- [PI05 策略说明](policy/pi05.md)
- [Groot 策略说明](policy/groot.md)
- [SmolVLA 策略说明](policy/smolvla.md)
- [训练策略对比](training/comparison.md)

---

## 7. 文档与结构改进

!!! note "文档与入口"
    - **项目文件构成**：[项目文件构成说明](getting_started/file_structure.md) 提供仓库目录与关键文件清单，便于快速定位配置、训练与部署代码。
    - **常见问题**：[常见问题](faq.md) 汇总环境、训练、部署、模型权重及 PI0/PI05 相关常见问题与解决办法。
    - **统一训练入口**：ACT、Diffusion、PI0、PI05 均通过 `kuavo_train/train_policy.py` 训练，详见 [训练流程](training/pipeline.md)。

---

## 相关文档索引

| 主题 | 文档 |
|------|------|
| 快速上手 | [快速开始](getting_started/quick_start.md) |
| 安装与环境 | [安装指南](getting_started/installation.md) |
| 文件与目录 | [项目文件构成](getting_started/file_structure.md) |
| 策略对比与选型 | [策略概览](concepts/policy_overview.md)、[训练策略对比](training/comparison.md) |
| RGB-Depth 融合 | [RGB-深度融合](advanced/rgb_depth_fusion.md)、[深度支持说明](advanced/depth_support.md) |
| 训练与多卡 | [训练流水线](training/pipeline.md)、[多卡训练](training/multi_gpu.md) |
| 部署 | [仿真自动测试](deployment/sim_auto_test.md)、[真机评测](deployment/real_eval.md) |
| 常见问题 | [常见问题](faq.md) |

---

## 更多资讯

乐聚 OpenLET 社区会持续发布赛事动态、技术经验分享与开源资源更新，可前往 [OpenLET 资讯](https://openlet.openatom.tech/explore/journalism) 浏览。
