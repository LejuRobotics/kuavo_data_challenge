# PI0 策略说明（PaliGemma + Expert）

PI0 基于视觉-语言模型 **PaliGemma** 编码图像与可选语言指令，再通过专家头（action expert）在状态与动作空间进行模仿学习。支持纯 RGB 或 RGB+Depth（Cross-Attention 融合），可选语言条件；部署时可置空 `task` 实现无语言条件。本项目通过 `CustomPI0ConfigWrapper` 与 `CustomPI0Pytorch` 扩展，支持 depth 的 Cross-Attention 融合。

---

## 1. 核心思想

- **VLM + Expert**：PaliGemma 作为视觉-语言编码器，专家头负责状态→动作映射；
- **Flow Matching**：动作通过 flow matching 生成，推理时多步去噪；
- **多模态**：RGB、可选 Depth、可选语言、机器人状态；
- **无语言条件**：训练与部署均可使用空 `task`，实现纯视觉/状态条件。

---

## 2. 模型架构概览

```
观测（RGB + 可选 Depth + state + 可选 language）
        ↓
┌─────────────────────────────────────────────────────────────┐
│ RGB: PaliGemma Vision Tower → RGB features                  │
│ Depth（若 use_depth）: 复制3通道 → PaliGemma Vision Tower   │
│        → CrossModalAttentionFusion 双向 Cross-Attention     │
│        → [fused_rgb, fused_depth] 加入 prefix embeddings    │
└─────────────────────────────────────────────────────────────┘
        ↓
[ image_embeddings, (可选) language_embeddings, state_embeddings ]
        ↓
PaliGemma LLM + Expert Head
        ↓
Flow Matching 解码 → (B, chunk_size, action_dim)
```

---

## 3. 输入与输出

### 3.1 输入

| 类型 | 键名 | 说明 |
|------|------|------|
| RGB | `image_features` | 多视角 RGB，如 `observation.images.head_cam_h`、`wrist_cam_l`、`wrist_cam_r` |
| Depth | `depth_features` | 可选，Cross-Attention 时与 RGB 共享 PaliGemma |
| State | `robot_state_feature` | 关节角度、夹爪等 |
| Language | `task` / 语言键 | 可选，部署时可注入空字符串 `""` |

### 3.2 输出

- **形状**：`(B, chunk_size, action_dim)`
- **推理**：`select_action` 返回 action chunk，按 `n_action_steps` 逐步执行。

---

## 4. 关键配置参数

### 4.1 结构与动作

| 参数 | 说明 | 典型值 |
|------|------|--------|
| `n_obs_steps` | 观测历史步数 | 1 |
| `chunk_size` | 动作块长度 | 50 |
| `n_action_steps` | 每步执行动作数 | 50 |
| `max_state_dim` | 状态维度上限 | 32 |
| `max_action_dim` | 动作维度上限 | 32 |
| `image_resolution` | 图像分辨率 | [224, 224] |
| `empty_cameras` | 空相机数量 | 0 |

### 4.2 模型变体

| 参数 | 说明 | 典型值 |
|------|------|--------|
| `paligemma_variant` | PaliGemma 变体 | gemma_2b |
| `action_expert_variant` | 专家头变体 | gemma_300m |
| `dtype` | 精度 | float32 / bfloat16 |

### 4.3 Flow Matching

| 参数 | 说明 | 典型值 |
|------|------|--------|
| `num_inference_steps` | 推理去噪步数 | 10 |
| `time_sampling_beta_alpha` | 时间采样参数 | 1.5 |
| `time_sampling_beta_beta` | 时间采样参数 | 1.0 |
| `min_period` / `max_period` | 周期范围 | 4e-3 / 4.0 |

### 4.4 Depth（可选）

| 参数 | 说明 | 典型值 |
|------|------|--------|
| `use_depth` | 是否启用 depth | false |
| `depth_features` | depth 观测键 | [] 或与 RGB 对应 |
| `depth_fusion_method` | 融合方法 | cross_attention |
| `depth_preprocessing` | 预处理 | minmax |
| `depth_scale` | 缩放因子 | 0.0001 |

### 4.5 归一化

- **VISUAL**：`IDENTITY`（PaliGemma 内部处理）
- **STATE**：`MEAN_STD` 或 `QUANTILES`
- **ACTION**：`MEAN_STD` 或 `QUANTILES`

### 4.6 训练优化

| 参数 | 说明 | 典型值 |
|------|------|--------|
| `gradient_checkpointing` | 梯度检查点 | true |
| `compile_model` | torch.compile | false |
| `optimizer_lr` | 学习率 | 2.5e-5 |
| `optimizer_weight_decay` | 权重衰减 | 0.01 |
| `optimizer_grad_clip_norm` | 梯度裁剪 | 1.0 |
| `tokenizer_max_length` | Tokenizer 最大长度 | 48 |

---

## 5. 预训练模型下载（HuggingFace）

训练与部署 PI0 需要两类模型：**PaliGemma**（视觉-语言底座，含 tokenizer）与 **LeRobot π₀ 预训练策略**（pi0_base）。以下为从 HuggingFace 获取的常用方式。

### 5.1 PaliGemma 开源模型（含 Tokenizer）

PI0 使用的 tokenizer 与视觉编码器来自 Google 的 PaliGemma。部署时若使用本地 tokenizer，需单独下载。

- **HuggingFace 模型 ID**：[google/paligemma-3b-pt-224](https://huggingface.co/google/paligemma-3b-pt-224)
- **说明**：使用前需在 [HuggingFace 模型页](https://huggingface.co/google/paligemma-3b-pt-224) 同意 Google 的使用条款。

**方式一：命令行下载到本地目录**

```bash
# 安装 huggingface_hub（若未安装）
pip install huggingface_hub

# 登录（首次使用需 token，见 https://huggingface.co/settings/tokens）
huggingface-cli login

# 下载到指定目录（例如 /path/to/models/paligemma-3b-pt-224）
huggingface-cli download google/paligemma-3b-pt-224 \
  --local-dir /path/to/models/paligemma-3b-pt-224 \
  --local-dir-use-symlinks False
```

**方式二：Python 预下载（缓存到默认目录）**

```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="google/paligemma-3b-pt-224",
    local_dir="/path/to/models/paligemma-3b-pt-224",  # 可选，不填则用默认 cache
    local_dir_use_symlinks=False,
)
```

部署时通过环境变量指定 tokenizer 路径（与训练/checkpoint 中路径不一致时）：

```bash
export PALIGEMMA_TOKENIZER_PATH=/path/to/models/paligemma-3b-pt-224
```

### 5.2 LeRobot π₀ 预训练底座（pi0_base）

PI0 的完整预训练策略（含 PaliGemma + 专家头）由 LeRobot 提供，训练时 `pretrained_path` 可直接使用 Hub ID，首次运行会自动下载。

- **HuggingFace 模型 ID**：[lerobot/pi0_base](https://huggingface.co/lerobot/pi0_base)
- **可选**：[lerobot/pi0_libero](https://huggingface.co/lerobot/pi0_libero)（在 Libero 上微调）

**方式一：训练时直接使用 Hub ID（自动下载）**

```bash
--policy.pretrained_path=lerobot/pi0_base
```

**方式二：预先下载到本地再指定路径**

```bash
# 命令行
huggingface-cli download lerobot/pi0_base \
  --local-dir /path/to/pi0_base \
  --local-dir-use-symlinks False
```

```python
# Python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="lerobot/pi0_base",
    local_dir="/path/to/pi0_base",
    local_dir_use_symlinks=False,
)
```

然后在配置或命令行中设置 `pretrained_path=/path/to/pi0_base`。

**离线 / 仅用缓存**：若已下载过，可设置 `HF_HUB_OFFLINE=1` 使用本地缓存，避免联网。

---

## 6. 预训练与 Quantile

- **pretrained_path**：**必填**，指向 PI0 预训练底座（如 `lerobot/pi0_base` 或本地路径）；
- **Quantile**：若使用 `QUANTILES` 归一化，需先运行 `augment_dataset_quantile_stats.py` 为 `stats.json` 补充分位数统计；
- **PI0** 无 `freeze_vision_encoder`、`train_expert_only` 等字段（与 PI05 不同）。

---

## 7. Depth 支持（Cross-Attention）

启用 `use_depth` 且 `depth_fusion_method: "cross_attention"` 时：
- Depth 复制为 3 通道后，与 RGB 共享 PaliGemma Vision Tower；
- `CrossModalAttentionFusion` 做双向 Cross-Attention，融合结果加入 prefix embeddings。
详见 [RGB-深度融合](../advanced/rgb_depth_fusion.md)。

---

## 8. 训练

### 8.1 配置与入口

- **Config**：`configs/policy/pi0_rgb_config.yaml`（RGB-only）或 `pi0_crossattn_config.yaml`（RGB+Depth）
- **入口**：`third_party/lerobot/src/lerobot/scripts/lerobot_train.py`

### 8.2 训练命令示例（RGB-only）

```bash
CUDA_VISIBLE_DEVICES=0 python third_party/lerobot/src/lerobot/scripts/lerobot_train.py \
  --dataset.repo_id=sim_task2_lerobot \
  --dataset.root=/path/to/sim_task2_lerobot/lerobot \
  --policy.type=pi0 \
  --output_dir=./outputs/train/sim_task2/pi0_rgb \
  --job_name=pi0_training \
  --policy.pretrained_path=/path/to/pi0_base \
  --policy.push_to_hub=false \
  --steps=30000 \
  --batch_size=16 \
  --policy.device=cuda
```

### 8.3 数据与 Quantile

- 使用 quantile 时：先运行 `augment_dataset_quantile_stats.py`；
- RGB-only 训练时：在 `lerobot_train.py` 中会删除 batch 的 `observation.depth_*` 键。

---

## 9. 部署

- **policy_type**：`pi0`
- **仿真 / 真机**：`sim_auto_test`、`real_single_test` 通过 `setup_policy(..., "pi0", device)` 加载；
- **RGB-only 部署**：自动裁剪 `input_features` 中的 depth 键，删除 `observation.depth_*`，注入 `observation["task"] = [""]`；
- **Tokenizer**：可设置 `PALIGEMMA_TOKENIZER_PATH` 覆盖 checkpoint 中的路径；
- **dtype / compile**：部署时强制 `float32`、`compile_model=false`。

---

## 10. 相关文档

- [策略概览](../concepts/policy_overview.md)：策略对比与选型
- [RGB-深度融合](../advanced/rgb_depth_fusion.md)：PI0 的 Cross-Attention depth 融合
- [深度支持说明](../advanced/depth_support.md)：Depth 支持实现细节
- [训练流水线](../training/pipeline.md)：训练数据流与 quantile
- [仿真自动测试](../deployment/sim_auto_test.md)：仿真部署与 task 注入
