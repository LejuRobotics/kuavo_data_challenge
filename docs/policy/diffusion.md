# Diffusion Policy 说明

Diffusion Policy 将动作序列建模为扩散过程，通过去噪网络从噪声逐步恢复出 action chunk。支持多视角 RGB 与可选深度图的 Cross-Attention 融合，适用于高维、多模态动作分布。本项目通过 `CustomDiffusionConfigWrapper` 与 `CustomDiffusionModelWrapper` 扩展，支持 depth 融合及 Transformer / UNet 两种去噪骨干。

---

## 1. 核心思想

- **扩散建模**：在动作空间定义前向扩散（加噪）与反向去噪过程，去噪网络以观测为条件预测噪声；
- **Action Chunking**：预测 `horizon` 步动作，推理时通过 DDPM/DDIM 采样得到 chunk，按 `n_action_steps` 逐步执行；
- **多模态条件**：RGB、Depth、state 经编码与融合后作为全局条件（global_cond）输入去噪网络。

---

## 2. 模型架构概览

```
观测（RGB + 可选 Depth + state）
        ↓
┌─────────────────────────────────────────────────────────────┐
│ RGB: ResnetRgbEncoder → rgb_attn_layer (自注意力)           │
│ Depth: ResnetDepthEncoder (1ch) → depth_attn_layer          │
│        → multimodalfuse (rgb_q / depth_q 双向 Cross-Attn)    │
│        → concat 与 state 一起  →  global_cond                 │
└─────────────────────────────────────────────────────────────┘
        ↓
去噪网络（Transformer 或 UNet）以 global_cond 为条件
        ↓
DDPM/DDIM 采样 → (B, horizon, action_dim)
```

---

## 3. 输入与输出

### 3.1 输入

| 类型 | 键名 | 说明 |
|------|------|------|
| RGB | `image_features` | 多视角 RGB，经 crop/resize 预处理 |
| Depth | `depth_features` | 可选，与 RGB 相机对应 |
| State | `robot_state_feature` | 关节、夹爪等，可选 `use_state_encoder` |
| Env | `env_state_feature` | 可选环境状态 |

### 3.2 输出

- **形状**：`(B, horizon, action_dim)`
- **推理**：内部 queue 按步输出动作，与 ACT 类似。

---

## 4. 关键配置参数

### 4.1 观测与动作

| 参数 | 说明 | 典型值 |
|------|------|--------|
| `n_obs_steps` | 观测历史步数 | 2 |
| `horizon` | 预测动作步数 | 16 |
| `n_action_steps` | 每步执行动作数 | 8 |
| `drop_n_last_frames` | 丢弃最后 n 帧 | 7 |

### 4.2 扩散与去噪

| 参数 | 说明 | 典型值 |
|------|------|--------|
| `noise_scheduler_type` | 调度器 | DDPM |
| `num_train_timesteps` | 训练步数 | 100 |
| `beta_schedule` | β 调度 | squaredcos_cap_v2 |
| `prediction_type` | 预测目标 | epsilon |
| `num_inference_steps` | 推理步数（null 则用训练值） | null |
| `custom.use_unet` | 使用 UNet | false |
| `custom.use_transformer` | 使用 Transformer | true |
| `transformer_n_emb` | Transformer 维度 | 384 |
| `transformer_n_layer` | 层数 | 12 |

### 4.3 视觉与 Depth

| 参数 | 说明 | 典型值 |
|------|------|--------|
| `vision_backbone` | RGB 编码器 | resnet18 |
| `spatial_softmax_num_keypoints` | 空间 softmax 关键点数 | 64 |
| `crop_shape` | 图像裁剪尺寸 | [420, 560] |
| `custom.use_depth` | 是否启用 depth | true |
| `custom.depth_backbone` | depth 编码器 | resnet18 |
| `custom.use_separate_depth_encoder_per_camera` | 每相机独立 depth 编码器 | false |
| `custom.resize_shape` | resize 尺寸 | [210, 280] |

### 4.4 State 编码

| 参数 | 说明 | 典型值 |
|------|------|--------|
| `custom.use_state_encoder` | 是否用 MLP 编码 state | true |
| `custom.state_feature_dim` | state 编码维度 | 128 |
| `custom.state_fuse` | 是否使用 state 引导的融合块 | false |

### 4.5 归一化

- **RGB**：`MEAN_STD`
- **STATE**：`MIN_MAX`
- **ACTION**：`MIN_MAX`
- **DEPTH**：`MIN_MAX`

---

## 5. Depth 支持

启用 `custom.use_depth` 且配置 `depth_features` 时，使用 `ResnetDepthEncoder` 编码 depth，经 `depth_attn_layer` 自注意力后，与 RGB 通过 `multimodalfuse`（`rgb_q`、`depth_q`）做双向 Cross-Attention，融合结果与 state 一起作为 `global_cond`。详见 [RGB-深度融合](../advanced/rgb_depth_fusion.md)。

---

## 6. 训练

### 6.1 配置与入口

- **Config**：`configs/policy/diffusion_config.yaml`，`_target_: kuavo_train.wrapper.policy.diffusion.DiffusionConfigWrapper.CustomDiffusionConfigWrapper`
- **入口**：`third_party/lerobot/src/lerobot/scripts/lerobot_train.py`

### 6.2 训练命令示例

```bash
CUDA_VISIBLE_DEVICES=0 python third_party/lerobot/src/lerobot/scripts/lerobot_train.py \
  --dataset.repo_id=sim_task2_lerobot \
  --dataset.root=/path/to/sim_task2_lerobot/lerobot \
  --policy.type=diffusion \
  --output_dir=./outputs/train/sim_task2/diffusion_rgb \
  --job_name=diffusion_training \
  --steps=30000 \
  --batch_size=32 \
  --policy.device=cuda
```

### 6.3 常用超参数

- **optimizer_lr**：0.0001
- **optimizer_betas**：[0.95, 0.999]
- **optimizer_weight_decay**：1e-3
- **scheduler_name**：cosine
- **scheduler_warmup_steps**：500

---

## 7. 部署

- **policy_type**：`diffusion`
- **仿真**：`sim_auto_test.py` 通过 `setup_policy(pretrained_path, "diffusion", device)` 加载；
- **真机**：`real_single_test.py` 同样支持，在 `kuavo_env.yaml` 中设置 `policy_type: diffusion`。

Diffusion 使用 mean/std 与 min-max 归一化，无需 quantile；preprocessor 从 run_dir 加载。

---

## 8. UNet vs Transformer

- **use_unet=true**：使用 1D UNet 作为去噪网络，`down_dims`、`kernel_size`、`n_groups` 等生效；
- **use_transformer=true**：使用 `TransformerForDiffusion`，`transformer_n_emb`、`transformer_n_layer`、`transformer_n_head` 等生效。

当前默认推荐 `use_transformer: true`。

---

## 9. 相关文档

- [策略概览](../concepts/policy_overview.md)：策略对比与选型
- [RGB-深度融合](../advanced/rgb_depth_fusion.md)：Diffusion 的 depth 融合流程
- [深度支持说明](../advanced/depth_support.md)：Depth 支持实现细节
- [训练流水线](../training/pipeline.md)：训练数据流与入口
- [仿真自动测试](../deployment/sim_auto_test.md)：仿真部署
