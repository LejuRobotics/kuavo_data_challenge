# SmolVLA 策略说明

SmolVLA 是 Hugging Face 提出的**轻量化视觉-语言-动作（VLA）**模型，基于 [SmolVLM2](https://huggingface.co/papers/2506.01844) 视觉-语言底座与 **Flow Matching** 动作专家头，在保持较低计算成本的同时支持图像、语言与机器人状态的多模态输入，并输出动作序列。

!!! abstract "简要说明"
    适合在**消费级 GPU 或边侧设备**上训练与部署；输入为多视角图像、语言 token、状态，输出为固定长度 action chunk。本仓库中实现位于 LeRobot 的 `lerobot/policies/smolvla`，当前 **Kuavo 训练/部署脚本尚未注册**，若需使用需自行接 config 与 wrapper。

---

## 1. 核心思想

| 要点 | 说明 |
|------|------|
| **VLM + Action Expert** | 预训练 SmolVLM2 编码图像与语言；独立专家 Transformer 在 VLM 的 KV cache 之上，结合状态与带时间编码的噪声动作，通过 **Flow Matching** 去噪得到动作序列。 |
| **轻量化** | 参数量小于 PI0/PI05 等大 VLM，便于在资源受限环境运行。 |
| **多模态** | 图像（多视角）、语言 token、机器人状态 → 固定长度 action chunk。 |

---

## 2. 模型架构概览

```
图像 + 语言 token + 状态
        ↓
  embed_prefix：SigLIP 编码图像 → 与语言/状态 embedding 拼接为 prefix
        ↓
  VLM（SmolVLM） + Action Expert（交叉/自注意力）
        ↓
  embed_suffix：噪声动作 + 时间步 → 专家输入
        ↓
  Flow Matching 多步去噪 → (B, chunk_size, action_dim)
```

- **Prefix**：图像特征（resize+padding、[0,1]→[-1,1]）、语言 embedding、状态投影。
- **Suffix**：噪声动作 + 时间正弦位置编码，专家通过 cross-attention 使用 VLM KV cache。
- **解码**：推理时从纯噪声出发，按 `num_steps` 步迭代；训练时 MSE 拟合速度场。

---

## 3. 输入与输出

### 3.1 输入

| 类型 | 说明 |
|------|------|
| **图像** | 多路相机，`resize_imgs_with_padding` 默认 (512, 512)，[0,1]→[-1,1]；支持 `empty_cameras` 占位。 |
| **语言** | tokenizer 编码为 `observation.language_tokens` / `observation.language_attention_mask`；processor 保证 task 以换行结尾。 |
| **状态** | `observation.state`，pad 到 `max_state_dim`（默认 32）。 |

### 3.2 输出

- **形状**：`(B, chunk_size, action_dim)`，默认 `chunk_size=50`、`n_action_steps=50`。
- **推理**：`select_action` 维护 action queue，queue 空时调用 `sample_actions` 生成新 chunk；支持 RTC 时使用 `predict_action_chunk`。

---

## 4. 关键配置参数

### 4.1 结构与动作

| 参数 | 说明 | 典型值 |
|------|------|--------|
| `n_obs_steps` | 观测历史步数 | 1 |
| `chunk_size` | 动作块长度 | 50 |
| `n_action_steps` | 每步执行动作数 | 50 |
| `max_state_dim` / `max_action_dim` | 状态/动作最大维度（不足则 pad） | 32 |
| `resize_imgs_with_padding` | 图像目标尺寸 | (512, 512) |
| `empty_cameras` | 空相机数量 | 0 |

### 4.2 模型与解码

| 参数 | 说明 | 典型值 |
|------|------|--------|
| `vlm_model_name` | VLM 底座 HuggingFace ID | `HuggingFaceTB/SmolVLM2-500M-Video-Instruct` |
| `load_vlm_weights` | 是否加载 VLM 权重（False 可仅训专家） | False |
| `num_steps` | Flow Matching 去噪步数 | 10 |
| `attention_mode` | 专家与 VLM 注意力方式 | `"cross_attn"` |
| `num_vlm_layers` / `num_expert_layers` | VLM 层数 / 专家层数（≤0 表示与 VLM 一致） | 16 / -1 |
| `self_attn_every_n_layers` | 每 N 层用 self-attn | 2 |
| `expert_width_multiplier` | 专家隐藏层相对 VLM 比例 | 0.75 |
| `min_period` / `max_period` | 时间正弦位置编码周期 | 4e-3 / 4.0 |

### 4.3 微调控制

| 参数 | 说明 | 典型值 |
|------|------|--------|
| `freeze_vision_encoder` | 是否冻结视觉编码器 | True |
| `train_expert_only` | 是否只训练专家 | True |
| `train_state_proj` | 是否训练状态投影层 | True |

!!! note "归一化与 Aloha"
    **normalization_mapping**：VISUAL 常用 IDENTITY，STATE/ACTION 常用 MEAN_STD。**adapt_to_pi_aloha** 用于 Aloha 空间与 SmolVLA 预训练空间转换，非 Aloha 场景可保持 False。

---

## 5. 预处理与后处理

**预处理链路**：RenameObservations → AddBatchDimension → **SmolVLANewLineProcessor**（task 以 `\n` 结尾）→ **TokenizerProcessorStep** → Device → **Normalizer**（STATE/ACTION 按 dataset_stats）。

**后处理**：Unnormalizer（ACTION）→ Device(cpu)。

---

## 6. 训练与推理

### 6.1 依赖安装

```bash
pip install -e ".[smolvla]"
```

（若 requirements 已包含 smolvla 可跳过。）

### 6.2 使用预训练底座微调

```bash
lerobot-train \
  --policy.path=lerobot/smolvla_base \
  --dataset.repo_id=<your_dataset> \
  --batch_size=64 \
  --steps=200000
```

### 6.3 从零指定类型训练

```bash
lerobot-train \
  --policy.type=smolvla \
  --dataset.repo_id=<your_dataset> \
  --batch_size=64 \
  --steps=200000
```

### 6.4 代码中加载

```python
from lerobot.policies.smolvla import SmolVLAPolicy

policy = SmolVLAPolicy.from_pretrained("lerobot/smolvla_base")
```

---

## 7. 与 Kuavo 项目的关系

!!! info "当前状态"
    本仓库 **kuavo_train** 与 **kuavo_deploy** 主要对接 ACT、Diffusion、PI0、PI05；SmolVLA 在 LeRobot 侧已有实现，但**尚未在训练入口与仿真/真机部署中注册**。

若要在 Kuavo 数据上使用 SmolVLA，需：

1. 在 `configs/policy/` 下增加 `smolvla_config.yaml`；
2. 在 `kuavo_train` 中支持 `policy_name=smolvla` 并指向 LeRobot SmolVLA；
3. 在 `kuavo_deploy` 的 `setup_policy` 中增加对 `smolvla` 的加载与预处理。

可参考 [ACT](act.md)、[PI0](pi0.md) 的集成方式。

---

## 8. 相关文档与资源

| 类型 | 链接 |
|------|------|
| 论文 | [SmolVLA (Hugging Face)](https://huggingface.co/papers/2506.01844) |
| 源码 | `third_party/lerobot/src/lerobot/policies/smolvla/` |
| 策略概览 | [策略概览](../concepts/policy_overview.md) |
| 训练流水线 | [训练流水线](../training/pipeline.md) |
