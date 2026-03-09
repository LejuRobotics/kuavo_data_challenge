# 训练对比（Training Comparison）

本节从**策略类型**、**训练与表现指标**、**配置与模态**、**RGB-Depth 融合方案**等维度整理对比，便于选型与复现。各策略在 Kuavo 仿真任务上的具体数值需结合实际训练与评估结果填写。

---

## 1. 策略类型对比总表

| 策略 | 输入模态 | 输出形式 | 归一化方式 | 部署支持 | 典型用途 |
|------|----------|----------|------------|----------|----------|
| **ACT** | RGB（+ 可选 Depth）、状态 | Action Chunk（一段动作序列） | mean/std | 是 | 通用模仿学习，VAE 隐空间 |
| **Diffusion** | RGB（+ 可选 Depth）、状态 | Action Chunk（扩散去噪生成） | mean/std | 是 | 多模态、高维动作分布 |
| **PI0** | RGB（+ 可选 Depth）、状态、可选语言 | 动作序列（Transformer 解码） | quantile（或 mean/std） | 是 | 大视觉语言模型 + 专家头，可弱语言/无语言 |
| **PI05** | 同 PI0 | 同 PI0（OpenPI 风格） | quantile（或 mean/std） | 是 | 与 PI0 类似，不同底座与接口 |
| **Groot** | 多模态 VLA | 动作 / 决策 | 见文档 | 文档/预留 | 多任务、具身推理 |
| **SmolVLA** | 轻量 VLA | 动作 | 见文档 | 文档/预留 | 轻量化 VLA 部署 |

- **部署支持**：当前仿真与真机脚本已接通 **diffusion、act、pi0、pi05**；Groot、SmolVLA 为预留。
- **归一化**：PI0/PI05 使用 quantile 时需事先运行 `augment_dataset_quantile_stats.py`；ACT/Diffusion 使用 `stats.json` 中的 mean/std。

---

## 2. 训练与表现指标对比

以下指标需在**相同任务、相同数据与评估协议**下对比才有意义；表中“—”表示暂无统一实测数据，可随实验补充。

### 2.1 资源与效率

| 策略 | 参数量级 | 典型 batch_size | 显存占用（约） | 收敛速度（相对） | 推理速度（相对） |
|------|------------|-----------------|----------------|------------------|------------------|
| **ACT** | 中 | 32–64 | 中 | 较快 | 快 |
| **Diffusion** | 中 | 16–32 | 中–高 | 中等 | 中等（去噪步数影响） |
| **PI0** | 大（VLM+专家） | 8–16 | 高 | 依赖预训练与数据 | 中等 |
| **PI05** | 大（同 PI0 量级） | 8–16 | 高 | 同 PI0 | 中等 |
| **Groot** | 大 | — | — | — | — |
| **SmolVLA** | 小–中 | — | 低–中 | — | 快（设计目标） |

- **PI0/PI05**：需 `policy.pretrained_path` 预训练底座，显存与 batch 受 VLM 限制；可开 `gradient_checkpointing` 省显存。
- **Diffusion**：去噪步数、chunk 长度等会直接影响训练与推理耗时。

### 2.2 任务表现（需按实验填写）

建议在 **sim_task1 / sim_task2 / sim_task3** 等固定任务上，用相同 episode 数、相同评估脚本（如 `sim_auto_test`）跑多轮后填表。

| 策略 | 任务成功率（示例） | 备注 |
|------|-------------------|------|
| **ACT** | — | 可注明任务名与 epoch |
| **Diffusion** | — | 可注明任务名与 epoch |
| **PI0（RGB-only）** | — | 无语言/弱语言时需与部署一致 |
| **PI05（RGB-only）** | — | 同上 |
| **PI0（RGB+Depth）** | — | 需与融合方案（见下）一起记录 |
| **PI05（RGB+Depth）** | — | 同上 |

- **评估方式**：在仿真中运行 `script_auto_test.py`，根据任务判定成功/失败，汇总成功率；详见 [仿真自动测试](../deployment/sim_auto_test.md)。
- **一致性**：训练若为**无语言 / 仅 RGB**，部署时也需**无语言 + 仅 RGB**，否则存在分布偏移，影响成功率。

---

## 3. 配置与模态对比

### 3.1 输入模态

| 配置 | 图像 | 状态 | 语言 | 说明 |
|------|------|------|------|------|
| **仅 RGB** | 多视角 RGB | 是 | 可选（可置空） | 当前默认；与部署一致 |
| **RGB + Depth** | RGB + 深度图 | 是 | 可选 | 需在策略与部署中同时开启 depth |

- 训练时在 `lerobot_train.py` 中若删除 `observation.depth_*`，即变为仅 RGB；部署时 `sim_auto_test` 会按策略类型过滤 depth 键。
- 语言：有语言时需在 batch 中提供 `task`/语言键；无语言时部署侧注入空字符串等以满足 processor 契约，见 [训练流水线](pipeline.md) 与 [快速开始](../getting_started/quick_start.md)。

### 3.2 常用训练参数示例

| 策略 | policy.type | 典型 steps | 典型 batch_size | 备注 |
|------|-------------|------------|-----------------|------|
| ACT | `act` | 30k–50k | 32–64 | chunk_size 等见 config |
| Diffusion | `diffusion` | 30k–50k | 16–32 | 去噪步数、chunk 见 config |
| PI0 | `pi0` | 20k–40k | 8–16 | 需 `policy.pretrained_path`，PI0 无 `freeze_vision_encoder` 等 |
| PI05 | `pi05` | 20k–40k | 8–16 | 需 `policy.pretrained_path`，可选 freeze/train_expert_only |

---

## 4. RGB-Depth 融合方案对比（PI0 系）

若使用 **RGB+Depth**，PI0 系有两种常见融合方式，可在同任务上对比训练与验证表现。

| 特性 | Sequence Concat（方案一） | Cross-Attention（方案二） |
|------|---------------------------|----------------------------|
| **深度编码器** | 独立 ViT Backbone | PaliGemma Vision Tower（与 RGB 共享） |
| **融合方式** | 序列拼接 `[RGB, Depth]` | 双向 Cross-Attention |
| **参数量** | 较多（多一个 ViT） | 较少 |
| **计算开销** | 较低 | 较高 |
| **实现/配置** | `depth_fusion_method=sequence_concat` | `depth_fusion_method=cross_attention` |

更详细的训练命令、检查点与结果分析模板见项目根目录 **TRAINING_COMPARISON_GUIDE.md**；方案架构与选型见 **RGB_DEPTH_FUSION_COMPARISON.md**。

---

## 5. 选型与实验建议

1. **资源紧张**：优先考虑 ACT 或 SmolVLA（若已接通）；PI0/PI05 可减小 batch、开 gradient checkpointing。
2. **要语言/多任务**：PI0、PI05 支持语言条件；训练与部署需一致（有语言/无语言、键名一致）。
3. **仅 RGB**：当前默认；若加 Depth，需同时改训练与部署，并选融合方案（见上表）。
4. **对比实验**：固定任务与数据，只改 `policy.type` 或融合方案，记录成功率、收敛曲线、显存与推理时间，再填入本文档表格。

---

## 6. 相关文档

- [策略概览](../concepts/policy_overview.md)：各策略简介与部署选择
- [训练流水线](pipeline.md)：数据流、入口脚本与 Kuavo 约定
- [快速开始](../getting_started/quick_start.md)：从数据到训练与仿真评估
- 项目根目录 **TRAINING_COMPARISON_GUIDE.md**：RGB-Depth 两种融合方案的训练命令与对比步骤
- 项目根目录 **RGB_DEPTH_FUSION_COMPARISON.md**：RGB-Depth 融合架构与特性对比
