# RGB-深度融合方案

本节讲解 **ACT、Diffusion、PI0、PI05** 四种策略的 depth 融合方法。各策略均通过 Cross-Attention 类机制实现 RGB 与 Depth 的双向交互，但编码器与融合细节不同。

---

## 1. 总览

| 策略 | 深度编码器 | 融合方式 | 融合模块 |
|------|------------|----------|----------|
| **ACT** | 独立 ResNet（1-channel） | 双向 Cross-Attention（按相机） | `CrossModalAttentionFusion` |
| **Diffusion** | 独立 ResNet（1-channel） | 双向 Cross-Attention + Concat | `multimodalfuse`（rgb_q / depth_q） |
| **PI0** | PaliGemma 共享 | 双向 Cross-Attention | `CrossModalAttentionFusion` |
| **PI05** | PaliGemma 共享 | 双向 Cross-Attention | `CrossModalAttentionFusion` |

---

## 2. ACT：Cross-Modal Attention 融合

### 2.1 架构流程

```
RGB 图像 [B, 3, H, W]  →  ResNet backbone  →  layer4 特征  →  encoder_img_feat_input_proj  →  RGB tokens [H*W, B, C]
Depth 图像 [B, 1, H, W]  →  ResNet depth_backbone (1ch conv1)  →  layer4 特征  →  encoder_depth_feat_input_proj  →  Depth tokens [H*W, B, C]
                                                                                        ↓
                                                    按相机逐一：CrossModalAttentionFusion
                                                    - RGB as query, Depth as key/value → fused_rgb
                                                    - Depth as query, RGB as key/value → fused_depth
                                                                                        ↓
                                                    concat([fused_rgb, fused_depth]) → cross_modal_fusion_proj → [H*W, B, C]
                                                                                        ↓
                                                        送入 Transformer Encoder（与 state、latent 等一起）
```

### 2.2 融合逻辑

- **按相机一一对应**：每个相机的 RGB token 与同相机的 Depth token 做 Cross-Attention；
- **双向**：RGB 作 query 访问 Depth，Depth 作 query 访问 RGB；
- **后处理**：融合后的 RGB 与 Depth 沿 channel 维 concat，再经 `cross_modal_fusion_proj` 映射回 `dim_model`；
- **输出**：融合 token 与 robot state、 latent 等一起进入 Transformer encoder。

### 2.3 配置与 Wrapper

| 字段 | 说明 |
|------|------|
| `custom.use_depth` | 是否启用 depth |
| `custom.depth_features` | depth 观测键，如 `["observation.depth_h", "observation.depth_l"]` |
| `custom.depth_backbone` | 深度编码器，如 `resnet18` |

实现：`kuavo_train/wrapper/policy/act/ACTModelWrapper.py`（`CustomACTModelWrapper`、`CrossModalAttentionFusion`）

---

## 3. Diffusion Policy：Cross-Attention + Concat 融合

### 3.1 架构流程

```
RGB 图像  →  ResnetRgbEncoder  →  rgb_attn_layer (自注意力)  →  img_features [B*S, n_cam, feat]
Depth 图像  →  ResnetDepthEncoder (1ch conv1)  →  depth_attn_layer (自注意力)  →  depth_features [B*S, n_cam, feat]
                                                                                        ↓
                                                    multimodalfuse:
                                                    - rgb_q: RGB as query, Depth as key/value  →  rgb_q_tokens
                                                    - depth_q: Depth as query, RGB as key/value  →  dep_q_tokens
                                                                                        ↓
                                                    concat([rgb_q_flat, dep_q_flat]) 与 state 一起  →  全局条件
                                                                                        ↓
                                                        输入去噪网络（Unet / Transformer）
```

### 3.2 融合逻辑

- **独立编码**：RGB 与 Depth 各自经 ResNet encoder 与自注意力得到 token；
- **双向 Cross-Attention**：`multimodalfuse["rgb_q"]` 和 `multimodalfuse["depth_q"]` 分别做一次 cross-attn；
- **Concat 作为全局条件**：融合后的 RGB 与 Depth token 展平后 concat，与 state、env_state 等一起作为去噪网络的 `global_cond`；
- **可选**：`use_separate_depth_encoder_per_camera=true` 时为每个 depth 相机使用独立 encoder。

### 3.3 配置与 Wrapper

| 字段 | 说明 |
|------|------|
| `use_depth` | 是否启用 depth |
| `depth_features` | depth 观测键 |
| `depth_backbone` | 深度编码器 backbone |
| `use_separate_depth_encoder_per_camera` | 是否每相机独立 depth encoder |
| `multimodal_heads` | Cross-Attention 的 num_heads |

实现：`kuavo_train/wrapper/policy/diffusion/DiffusionModelWrapper.py`（`CustomDiffusionModelWrapper`、`multimodalfuse`）

---

## 4. PI0 / PI05：Cross-Attention 融合

### 4.1 架构流程

```
RGB 图像 [B, 3, H, W]  →  PaliGemma Vision Tower  →  RGB features [B, N_rgb, C]
Depth 图像 [B, 1, H, W]  →  复制为 3 通道  →  PaliGemma Vision Tower  →  Depth features [B, N_depth, C]
                                                                                        ↓
                                                    CrossModalAttentionFusion:
                                                    - RGB as query, Depth as key/value  →  fused_rgb
                                                    - Depth as query, RGB as key/value  →  fused_depth
                                                                                        ↓
                                                    [fused_rgb, fused_depth] 均加入 prefix embeddings
                                                                                        ↓
                                                        输入 LLM（与语言、state 等）
```

### 4.2 融合逻辑

- **共享 Vision Tower**：Depth 复制为 3 通道后，与 RGB 共用 PaliGemma Vision Tower 编码；
- **双向 Cross-Attention**：`CrossModalAttentionFusion` 中 `rgb_depth_attn`、`depth_rgb_attn` 分别做一次 cross-attn；
- **残差 + LayerNorm**：每路输出加残差再 LayerNorm；
- **输出**：融合后的 RGB 与 Depth 特征都加入 prefix embeddings，一并送入 LLM。

### 4.3 Cross-Attention 细节

```python
# rgb_depth_attn: RGB as query, Depth as key/value
rgb_fused = norm_rgb(rgb_seq + rgb_depth_attn(query=rgb_seq, key=depth_seq, value=depth_seq))

# depth_rgb_attn: Depth as query, RGB as key/value
depth_fused = norm_depth(depth_seq + depth_rgb_attn(query=depth_seq, key=rgb_seq, value=rgb_seq))
```

### 4.4 配置与 Wrapper

| 字段 | 说明 |
|------|------|
| `use_depth` | 是否启用 depth |
| `depth_features` | depth 观测键，需与 RGB 相机一一对应 |
| `depth_fusion_method` | 固定为 `"cross_attention"` |
| `depth_fusion_dim` | 融合维度（可选，默认自动） |
| `depth_preprocessing` | 预处理方式，如 `minmax` |
| `depth_scale` | 缩放因子 |

实现：`kuavo_train/wrapper/policy/pi0/PI0ModelWrapper.py`、`kuavo_train/wrapper/policy/pi05/PI05ModelWrapper.py`（`CustomPI0Pytorch` / `CustomPI05Pytorch`、`CrossModalAttentionFusion`）

---

## 5. Depth 预处理（通用）

1. **格式**：支持 `[B, 1, H, W]`、`[B, H, W]`，多通道时取第一通道；
2. **数值**：uint16/mm → float，按 `depth_range` 裁剪并归一化到 [0, 1] 或 [-1, 1]；
3. **通道**：PI0/PI05 需将单通道扩展为 3 通道以输入 PaliGemma；ACT/Diffusion 使用 1-channel ResNet；
4. **尺寸**：与 RGB 保持一致的 resize/crop。

---

## 6. 训练与部署注意事项

1. **训练**：在 batch 中保留 `depth_features` 所列键，config 中正确设置 `use_depth`、`depth_features`；若用 RGB-only，则删除 depth 键。
2. **部署**：训练用 RGB+Depth 时，部署也需提供 depth；若训练为 RGB-only，部署时需删除 depth 键，与 [仿真自动测试](../deployment/sim_auto_test.md) 中逻辑一致。
3. **归一化**：`stats.json` 中需有 depth 统计；`normalization_mapping.DEPTH` 常为 `MIN_MAX`。
4. **相机对应**：`depth_features` 顺序与 `image_features` 对应（如 head、left、right），保证同一相机的 RGB 与 Depth 配对。

---

## 7. 相关文档

- [深度支持说明](depth_support.md)：各策略 depth 支持对比与实现细节
- [策略对比](../training/comparison.md)：RGB-only 与 RGB+Depth 配置对比
- [仿真自动测试](../deployment/sim_auto_test.md)：部署时 depth 过滤
