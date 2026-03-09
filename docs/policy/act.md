# ACT 策略说明（Action Chunking Transformer）

ACT 是一种基于 Transformer 的模仿学习策略，通过预测**整段动作序列**（action chunk）实现端到端控制，可选 VAE 学习动作分布以提升多样性。本项目通过 `CustomACTConfigWrapper` 与 `CustomACTModelWrapper` 扩展，支持多相机 RGB 与可选深度图的 Cross-Modal Attention 融合。

---

## 1. 核心思想

- **Action Chunking**：一次预测 `chunk_size` 步动作，推理时按步取用，减少重复推理；
- **Transformer 编码**：用 Encoder-Decoder 结构编码观测（图像、状态、可选深度），解码得到动作序列；
- **可选 VAE**：在隐空间建模动作分布，通过 KL 散度正则化提升动作多样性。

---

## 2. 模型架构概览

```
观测（RGB + 可选 Depth + state）
        ↓
┌─────────────────────────────────────────────────────────────┐
│ RGB: ResNet backbone → layer4 特征 → proj → RGB tokens      │
│ Depth: ResNet (1ch conv1) → layer4 → proj → Depth tokens   │
│        → CrossModalAttentionFusion（按相机逐一融合）          │
│        → concat + proj → 融合视觉 tokens                     │
└─────────────────────────────────────────────────────────────┘
        ↓
┌─────────────────────────────────────────────────────────────┐
│ 若 use_vae: [cls, state, action] → VAE Encoder → latent    │
│ 否则: latent = zeros                                         │
└─────────────────────────────────────────────────────────────┘
        ↓
[ latent_proj, state_embed, (可选) env_state, 融合视觉 tokens ]
        ↓
Transformer Encoder → Decoder（自回归 / 一步解码）→ (B, chunk_size, action_dim)
```

---

## 3. 输入与输出

### 3.1 输入

| 类型 | 键名 | 说明 |
|------|------|------|
| RGB | `image_features` | 多视角 RGB，如 `observation.images.head_cam_h`、`wrist_cam_l`、`wrist_cam_r` |
| Depth | `depth_features` | 可选，如 `observation.depth_h`、`observation.depth_l` |
| State | `robot_state_feature` | 关节角度、夹爪等 |
| Env | `env_state_feature` | 可选环境状态 |

由 `input_features` 定义，在 `info.json` 中需与数据集一致。

### 3.2 输出

- **形状**：`(B, chunk_size, action_dim)`
- **推理**：`select_action` 返回 action chunk，按 `n_action_steps` 取用，与 Diffusion 类似。

---

## 4. 关键配置参数

### 4.1 结构与动作

| 参数 | 说明 | 典型值 |
|------|------|--------|
| `n_obs_steps` | 观测历史步数 | 1 |
| `chunk_size` | 动作块长度 | 100 |
| `n_action_steps` | 每步执行动作数 | 1 |
| `dim_model` | Transformer 隐维度 | 512 |
| `n_encoder_layers` | Encoder 层数 | 4 |
| `n_decoder_layers` | Decoder 层数 | 1 |
| `n_heads` | 注意力头数 | 8 |

### 4.2 VAE

| 参数 | 说明 | 典型值 |
|------|------|--------|
| `use_vae` | 是否使用 VAE | true |
| `latent_dim` | 隐空间维度 | 32 |
| `n_vae_encoder_layers` | VAE encoder 层数 | 4 |
| `kl_weight` | KL 散度权重 | 10.0 |

### 4.3 视觉与 Depth

| 参数 | 说明 | 典型值 |
|------|------|--------|
| `vision_backbone` | RGB 编码器 | resnet18 |
| `custom.use_depth` | 是否启用 depth | true |
| `custom.depth_features` | depth 观测键 | 由 input_features 推断 |
| `custom.depth_backbone` | depth 编码器 | resnet18 |

### 4.4 归一化

- **RGB**：`MEAN_STD`
- **DEPTH**：`MIN_MAX`

---

## 5. Depth 支持

启用 `custom.use_depth` 且配置 `depth_features` 时，使用独立 ResNet（1-channel 首层）编码 depth，再通过 `CrossModalAttentionFusion` 与 RGB 做双向 Cross-Attention，按相机逐一融合。详见 [RGB-深度融合](../advanced/rgb_depth_fusion.md)。

---

## 6. 训练

### 6.1 配置与入口

- **Config**：`configs/policy/act_config.yaml`，`_target_: kuavo_train.wrapper.policy.act.ACTConfigWrapper.CustomACTConfigWrapper`
- **入口**：`third_party/lerobot/src/lerobot/scripts/lerobot_train.py`

### 6.2 训练命令示例

```bash
CUDA_VISIBLE_DEVICES=0 python third_party/lerobot/src/lerobot/scripts/lerobot_train.py \
  --dataset.repo_id=sim_task2_lerobot \
  --dataset.root=/path/to/sim_task2_lerobot/lerobot \
  --policy.type=act \
  --output_dir=./outputs/train/sim_task2/act_rgb \
  --job_name=act_training \
  --steps=30000 \
  --batch_size=32 \
  --policy.device=cuda
```

若仅用 RGB 训练，需在 `lerobot_train.py` 中删除 depth 键，或在 `input_features` 中不包含 depth。

### 6.3 常用超参数

- **optimizer_lr**：1e-5
- **optimizer_lr_backbone**：1e-5
- **kl_weight**：10.0
- **dropout**：0.1

---

## 7. 部署

- **policy_type**：`act`
- **仿真**：`sim_auto_test.py` 通过 `setup_policy(pretrained_path, "act", device)` 加载；
- **真机**：`real_single_test.py` 同样支持，需在 `kuavo_env.yaml` 中设置 `policy_type: act` 及 `task`、`method`、`timestamp`、`epoch`。

ACT 使用 mean/std 归一化，无需 quantile 统计；部署时 preprocessor 会从 run_dir 加载。

---

## 8. 相关文档

- [策略概览](../concepts/policy_overview.md)：策略对比与选型
- [RGB-深度融合](../advanced/rgb_depth_fusion.md)：ACT 的 depth 融合流程
- [深度支持说明](../advanced/depth_support.md)：Depth 支持实现细节
- [训练流水线](../training/pipeline.md)：训练数据流与入口
- [仿真自动测试](../deployment/sim_auto_test.md)：仿真部署
