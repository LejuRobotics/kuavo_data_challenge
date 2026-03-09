# Groot 策略说明

Groot 是 **NVIDIA** 提出的**多模态视觉-语言-动作（VLA）**模型，面向多任务与具身推理，支持多 embodiment（如 SO100、GR1、Unitree G1 等）。LeRobot 通过 `lerobot/policies/groot` 集成 **GR00T-N1.5**：使用 **Eagle** 视觉-语言 backbone 与 **Flow Matching** 动作头，可加载 HuggingFace 预训练权重并在 LeRobot 数据格式下微调与推理。

!!! abstract "简要说明"
    输入为图像、语言、状态与 **embodiment_id**，输出为动作序列；STATE/ACTION 使用 **MIN_MAX** 归一化（需数据集 min/max 统计）。预训练模型 **action_horizon 最大为 16**。本仓库中实现位于 `third_party/lerobot/.../groot`，当前 **Kuavo 训练/部署脚本尚未注册**，若需使用需自行接 config 与 processor。

---

## 1. 核心思想

| 要点 | 说明 |
|------|------|
| **Eagle + Flow Matching** | 视觉与语言由 **Eagle** 编码为统一表示；动作由 **FlowmatchingActionHead** 在给定状态与 embodiment 下生成。 |
| **多 embodiment** | 通过 `embodiment_tag` / `embodiment_mapping` 区分不同机器人，预训练支持多种 embodiment ID。 |
| **归一化** | STATE 与 ACTION 使用 **MIN_MAX** 归一化到 [-1, 1]，与 SO100 等一致；需提供 dataset min/max。 |

---

## 2. 模型架构概览

```
图像 + 语言 + 状态 + embodiment_id
        ↓
  Eagle backbone（vision + language）→ 投影到统一维度
        ↓
  GR00T 主体（Eagle 特征 + 状态 + embodiment）
        ↓
  FlowmatchingActionHead
        ↓
  动作序列 (B, action_horizon, action_dim)，action_horizon ≤ 16
```

- **Eagle**：图像与文本编码，输出再经线性投影；可配置 `tune_visual`、`tune_llm`。
- **Action Head**：接收 backbone 输出、状态等，flow matching 去噪生成动作；**action_horizon 在预训练模型中最大为 16**。
- **Processor**：将 LeRobot 的 observation/action 转为 Eagle 输入（video、language、state、embodiment_id），得到 `eagle_*` 张量。

---

## 3. 输入与输出

### 3.1 输入

| 类型 | 说明 |
|------|------|
| **图像** | 多路相机在 processor 中打包为 video (B, T, V, C, H, W)，经 Eagle 得到 `eagle_*` 输入。 |
| **语言** | `task` 等 complementary data，默认无则用 `"Perform the task."`；可配置 `formalize_language`。 |
| **状态** | `observation.state`，pad/截断到 `max_state_dim`（默认 64），min-max 归一化。 |
| **embodiment_id** | 由 `embodiment_tag` 映射为整数（如 `new_embodiment`→31）。 |

### 3.2 输出

- **形状**：推理时 `get_action` 返回 `action_pred`，为 (B, action_horizon, max_action_dim)，**action_horizon ≤ 16**。
- **后处理**：取最后一帧、截断到 `env_action_dim`，再做 min-max 反归一化。

!!! warning "action_horizon 限制"
    预训练 GR00T 的 action head **最大支持 16 步**；processor 中 `action_horizon = min(config.chunk_size, 16)`。若需要更长 chunk，需在数据或任务侧做对齐（如重复最后一帧或滑动窗口）。

---

## 4. 关键配置参数

### 4.1 基本策略参数

| 参数 | 说明 | 典型值 |
|------|------|--------|
| `n_obs_steps` | 观测历史步数 | 1 |
| `chunk_size` | 动作块长度（实际取 min(chunk_size, 16)） | 50 |
| `n_action_steps` | 每步执行动作数 | 50 |
| `max_state_dim` / `max_action_dim` | 状态/动作最大维度 | 64 / 32 |
| `image_size` | 图像尺寸（Eagle 侧） | (224, 224) |

### 4.2 模型与预训练

| 参数 | 说明 | 典型值 |
|------|------|--------|
| `base_model_path` | 预训练模型路径或 HuggingFace ID | `nvidia/GR00T-N1.5-3B` |
| `tokenizer_assets_repo` | Eagle tokenizer/processor 资源 | `lerobot/eagle2hg-processor-groot-n1p5` |
| `embodiment_tag` | 当前 embodiment 标签 | `"new_embodiment"`、`"gr1"` 等 |

### 4.3 微调控制

| 参数 | 说明 | 典型值 |
|------|------|--------|
| `tune_llm` | 是否微调 LLM backbone | False |
| `tune_visual` | 是否微调视觉塔 | False |
| `tune_projector` | 是否微调投影层 | True |
| `tune_diffusion_model` | 是否微调动作头 | True |
| `lora_rank` | LoRA 秩（0 表示不用） | 0 |
| `use_bf16` | 是否使用 bfloat16 | True |

### 4.4 训练与数据

| 参数 | 说明 | 典型值 |
|------|------|--------|
| `optimizer_lr` | 学习率 | 1e-4 |
| `warmup_ratio` | 预热步数比例 | 0.05 |
| `batch_size` | 批大小 | 32 |
| `video_backend` | 视频解码后端 | `"decord"` |
| `balance_dataset_weights` / `balance_trajectory_weights` | 多数据集/轨迹采样权重 | True |

!!! note "归一化"
    VISUAL 为 IDENTITY；STATE 与 ACTION 在 processor 中使用 **MIN_MAX**，需在 `dataset_stats` 中提供 min/max。

---

## 5. 预处理与后处理

**预处理**：RenameObservations → AddBatchDimension → **GrootPackInputsStep**（video、state/action min-max 归一化与 pad、language、embodiment_id）→ **GrootEagleEncodeStep**（Eagle 处理 video+language）→ **GrootEagleCollateStep**（eagle_content → `eagle_*` 张量）→ Device。

**后处理**：**GrootActionUnpackUnnormalizeStep**（取最后一帧、截断到 env_action_dim、min-max 反归一化）→ Device(cpu)。

---

## 6. 训练与推理

### 6.1 模型加载

通过 `GR00TN15.from_pretrained(pretrained_model_name_or_path, tune_llm=..., tune_visual=..., ...)` 加载；会拉取 GR00T-N1.5 权重与 Eagle 资源，并确保 `eagle2hg-processor-groot-n1p5` 等已就绪（`ensure_eagle_cache_ready`）。

### 6.2 训练与推理接口

- **训练**：`GrootPolicy.forward(batch)` 过滤出 `state`、`state_mask`、`action`、`action_mask`、`embodiment_id` 及 `eagle_*`，调用 `_groot_model.forward`，在 bf16 autocast 下计算 loss。
- **推理**：`predict_action_chunk(batch)` 不传 action/action_mask，只传 state、state_mask、embodiment_id 与 `eagle_*`，调用 `_groot_model.get_action` 得到 `action_pred`，再按 `output_features["action"]` 截断返回。

---

## 7. 与 Kuavo 项目的关系

!!! info "当前状态"
    本仓库 **kuavo_train** 与 **kuavo_deploy** 主要对接 ACT、Diffusion、PI0、PI05；Groot 在 LeRobot 侧已有实现，但**尚未在训练入口与仿真/真机部署中注册**。

若要在 Kuavo 数据上使用 Groot，需：

1. 在 `configs/policy/` 下增加 `groot_config.yaml`（base_model_path、embodiment_tag、max_state_dim/max_action_dim 等）；
2. 在 `kuavo_train` 中支持 `policy_name=groot` 并构建 Groot 的 pre/post processor（需提供 dataset_stats 的 min/max）；
3. 在 `kuavo_deploy` 的 `setup_policy` 中增加对 `groot` 的加载与 Eagle 编码管线。

可参考 [PI0](pi0.md)、[配置说明](../training/configs.md) 的集成方式。

---

## 8. 相关文档与资源

| 类型 | 链接 |
|------|------|
| HuggingFace 模型 | [nvidia/GR00T-N1.5-3B](https://huggingface.co/nvidia/GR00T-N1.5-3B) |
| 源码 | `third_party/lerobot/src/lerobot/policies/groot/` |
| 策略概览 | [策略概览](../concepts/policy_overview.md) |
| 训练流水线 | [训练流水线](../training/pipeline.md) |
