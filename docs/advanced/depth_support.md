# 深度支持实现细节

本节总结项目中 **ACT、Diffusion、PI0、PI05、Groot、SmolVLA** 对深度图像（depth）的支持方式。各策略均支持 depth 输入，但**融合方式与实现细节**各不相同。

---

## 1. 概述

- **数据来源**：多相机深度图（如 `observation.depth_h`、`observation.depth_l`、`observation.depth_r`），与 RGB 对齐；
- **预处理**：uint16/mm 转 float、归一化（min-max 或 quantile）、单通道转 3 通道（适配 RGB backbone）等；
- **融合位置**：在 wrapper / model 层，通过独立 backbone、共享 backbone 或 Cross-Attention 等方式融合 RGB 与 Depth 特征。

---

## 2. 各策略 Depth 支持对比

| 策略 | 深度编码器 | 融合方式 | 配置字段 | 备注 |
|------|------------|----------|----------|------|
| **ACT** | 独立 ResNet（1-channel conv1） | Cross-Modal Attention | `use_depth`, `depth_features`, `depth_backbone` | RGB/Depth 分别编码后交叉注意力 |
| **Diffusion** | 独立 ResNet 或每相机独立 encoder | Concat / Cross-Attn | `use_depth`, `depth_features`, `depth_backbone` | 可选 `use_separate_depth_encoder_per_camera` |
| **PI0** | 独立 ViT 或 PaliGemma 共享 | Sequence Concat / Cross-Attention | `use_depth`, `depth_features`, `depth_fusion_method` | 方案一/方案二，见 [RGB-深度融合](rgb_depth_fusion.md) |
| **PI05** | 同 PI0 | 同 PI0 | 同 PI0 | 与 PI0 共用两套融合方案 |
| **Groot** | 多模态 VLA 内置 | 上游设计 | 见 Groot 文档 | 预留，按 upstream 支持 depth |
| **SmolVLA** | 轻量 VLA 内置 | 上游设计 | 见 SmolVLA 文档 | 预留，按 upstream 支持 depth |

---

## 3. 数据来源与预处理

### 3.1 Depth 数据来源

- **ROS 话题**：`obs_key_map` 中配置（如 `/cam_h/depth/...`、`/cam_l/depth/...`），单位通常为 mm；
- **数据集**：LeRobot 的 `observation.depth_h`、`observation.depth_l` 等，需与 `meta/info.json`、`stats.json` 中定义一致；
- **归一化**：`normalization_mapping.DEPTH` 常用 `MIN_MAX`；PI0/PI05 可配合 quantile 统计。

### 3.2 预处理流程（通用）

1. **格式**：支持 `[B, 1, H, W]`、`[B, H, W]`、`[B, H, W, 1]` 等；
2. **数值**：uint16 → float，按 `depth_range`（如 [0, 1500] mm）裁剪与归一化；
3. **通道**：单通道 depth 重复为 3 通道，以接入预训练 RGB backbone（如 PaliGemma Vision Tower）；
4. **尺寸**：resize/crop 与 RGB 一致，保证时空对齐。

### 3.3 与 RGB 对齐与同步

- **时间同步**：观测 buffer 中 RGB 与 depth 使用同一时间戳或最近帧；
- **空间对齐**：同一相机的 RGB 与 depth 已标定对齐；不同相机按 `obs_key_map` 顺序组织；
- **训练时**：若仅用 RGB，需在 batch 中删除 `observation.depth_*`；部署时同理。

---

## 4. 各策略融合实现

### 4.1 ACT

- **深度编码器**：ResNet（如 `resnet18`），首层 `conv1` 改为 1-channel 输入，权重由 3-channel 平均得到；
- **特征提取**：`IntermediateLayerGetter` 取 `layer4` 特征图 → `encoder_depth_feat_input_proj` 投影到 `dim_model`；
- **融合**：`CrossModalAttentionFusion`（双向 Cross-Attention），RGB 与 Depth 特征互相 query/key/value；
- **输出**：融合后与 state token 一起送入 Transformer encoder。

配置示例（`act_config.yaml`）：

```yaml
custom:
  use_depth: true
  depth_features: ["observation.depth_h", "observation.depth_l"]
  depth_backbone: resnet18
```

### 4.2 Diffusion Policy

- **深度编码器**：`ResnetDepthEncoder`（1-channel 首层 ResNet），或 `use_separate_depth_encoder_per_camera` 时每相机独立 encoder；
- **特征流程**：Depth 经 encoder 得到 token，`depth_attn_layer` 自注意力；
- **融合**：RGB 与 Depth token 在 `_prepare_global_conditioning` 中 Concat 或 Cross-Attn 融合，再作为全局条件输入去噪网络。

配置示例：

```yaml
use_depth: true
depth_features: ["observation.depth_h", "observation.depth_l"]
depth_backbone: resnet18
# use_separate_depth_encoder_per_camera: false  # 可选
```

### 4.3 PI0 / PI05（两种方案）

#### 方案一：Sequence Concat（OpenVLA-Depth 风格）

- **深度编码器**：独立 ViT（如 `vit_base_patch16_224`），`timm` 创建；
- **流程**：RGB → PaliGemma Vision Tower；Depth → 独立 ViT → 投影到 LLM 维度；
- **融合**：`[RGB特征, Depth特征]` 序列拼接后输入 LLM。

配置：`depth_fusion_method: "sequence_concat"`，`depth_backbone: "vit_base_patch16_224"`。

#### 方案二：Cross-Attention

- **深度编码器**：与 RGB 共享 PaliGemma Vision Tower（depth 复制为 3 通道后输入）；
- **流程**：RGB/Depth 分别经 Vision Tower 编码，再经 `CrossModalAttentionFusion` 双向 Cross-Attention；
- **融合**：融合后的 RGB 与 Depth 特征均送入 LLM。

配置：`depth_fusion_method: "cross_attention"`；`depth_backbone` 不实际使用。

详见 [RGB-深度融合](rgb_depth_fusion.md)、项目根目录 `depth_support_implementation_summary.md`。

### 4.4 Groot / SmolVLA

- **Groot**：多模态 VLA，支持多模态输入；depth 融合由 upstream Groot/Eagle 设计决定；
- **SmolVLA**：轻量 VLA，多模态输入；具体 depth 接入方式见 LeRobot 与上游文档。

本项目中 Groot、SmolVLA 为预留，尚未接入 Kuavo wrapper；depth 支持以各 policy 的官方实现为准。

---

## 5. Wrapper 与 Config 控制

### 5.1 通用配置字段

| 字段 | 说明 | 示例 |
|------|------|------|
| `use_depth` | 是否启用 depth | `true` |
| `depth_features` | depth 观测键列表 | `["observation.depth_h", "observation.depth_l"]` |
| `depth_backbone` | 独立 depth 编码器（ACT/Diffusion/PI0 方案一） | `resnet18` / `vit_base_patch16_224` |

### 5.2 PI0/PI05 专属

| 字段 | 说明 | 取值 |
|------|------|------|
| `depth_fusion_method` | 融合方法 | `sequence_concat` / `cross_attention` |
| `depth_fusion_dim` | 融合维度（可选） | 整数或 `null`（自动） |
| `depth_preprocessing` | 预处理方式 | `minmax` 等 |
| `depth_scale` | 缩放因子 | 如 `0.0001` |

### 5.3 训练与部署注意事项

1. **训练**：在 `lerobot_train.py` 中，若使用 RGB-only，需删除 batch 中的 `observation.depth_*`；反之则保留并在 config 中正确配置 `depth_features`。
2. **部署**：`sim_auto_test` / `real_single_test` 中，若策略为 RGB-only 部署，会主动删除 depth 键；训练与部署的模态须一致。
3. **归一化**：`stats.json` 中需有 depth 的统计（mean/std 或 quantile）；`augment_dataset_quantile_stats.py` 可为 PI0/PI05 补充 quantile。

---

## 6. 相关文档

- [RGB-深度融合](rgb_depth_fusion.md)：ACT、Diffusion、PI0、PI05 的 depth 融合方法与流程
- [策略对比](../training/comparison.md)：RGB-only 与 RGB+Depth 的配置与表现对比
- [仿真自动测试](../deployment/sim_auto_test.md)：部署时 depth 的过滤与注入
- 项目根目录 **depth_support_implementation_summary.md**：PI0/PI05 实现细节
- 项目根目录 **RGB_DEPTH_FUSION_COMPARISON.md**：融合方案对比
