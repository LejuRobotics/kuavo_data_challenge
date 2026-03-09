# 训练流水线

本节描述从 **LeRobot 数据集** 到 **策略训练** 的完整数据流与执行顺序，包括入口脚本、数据加载、策略与处理器构建、训练循环以及 Kuavo 相关定制。

---

## 1. 流水线总览

整体流程可概括为：

```
LeRobot 数据集 (meta/info.json, stats.json, data/, videos/)
        ↓
  make_dataset(cfg)  →  LeRobotDataset
        ↓
  make_policy(cfg, ds_meta)  →  Policy（含 Kuavo wrapper）
        ↓
  make_pre_post_processors(policy_cfg, dataset_stats, ...)  →  Preprocessor, Postprocessor
        ↓
  DataLoader  →  每步: batch → [Kuavo: 删除 depth 键] → preprocessor(batch) → update_policy(policy, batch)
        ↓
  定期保存 checkpoint（policy + preprocessor + postprocessor）
```

- **数据**：由 `dataset.repo_id`、`dataset.root` 指定，需已具备 `meta/info.json`、`meta/stats.json`（PI0/PI05 还需 quantile 统计）。
- **策略**：由 `policy.type` 选择（如 `act`、`diffusion`、`pi0`、`pi05`），通过 LeRobot 的 policy factory 及本项目的 `kuavo_train/wrapper/policy/*` 封装创建。
- **处理器**：根据策略类型与 `dataset.meta.stats` 构建预处理/后处理流水线（归一化、tokenizer 等），训练与部署共用同一套配置。

---

## 2. 入口脚本与配置

### 2.1 统一入口脚本（kuavo_train）

- 所有策略（**ACT / Diffusion / PI0 / PI05**）在本项目中都通过 `kuavo_train/train_policy.py` 进行训练；
- 通过 `policy_name=act|diffusion|pi0|pi05` 与 `--config-name=*_config.yaml` 选择具体策略与配置；
- 多卡训练使用 `kuavo_train/train_policy_with_accelerate.py` + `accelerate`；
- 具体超参数（`task`、`method`、`root`、`training.*`、`policy.*` 等）集中在 `configs/policy/*.yaml`，并由 `kuavo_train/wrapper/policy/*` 负责与 Lerobot policy 对接。

### 2.2 常用命令行参数示例

**Diffusion（使用 `train_policy.py`）**：

```bash
python kuavo_train/train_policy.py   --config-path=../configs/policy/   --config-name=diffusion_config.yaml   task=sim_task2   method=diffusion_rgb   root=/path/to/sim_task2_lerobot/lerobot   training.batch_size=32   policy_name=diffusion
```

**ACT（使用 `train_policy.py`）**：

```bash
python kuavo_train/train_policy.py   --config-path=../configs/policy/   --config-name=act_config.yaml   task=sim_task2   method=act_rgb   root=/path/to/sim_task2_lerobot/lerobot   training.batch_size=32   policy_name=act
```

**PI0（使用 `train_policy.py`）**：

```bash
python kuavo_train/train_policy.py   --config-path=../configs/policy/   --config-name=pi0_rgb_config.yaml   task=sim_task2   method=pi0_rgb   root=/path/to/sim_task2_lerobot/lerobot   training.batch_size=16   policy_name=pi0
```

**PI05（使用 `train_policy.py`）**：

```bash
python kuavo_train/train_policy.py   --config-path=../configs/policy/   --config-name=pi05_config.yaml   task=sim_task2   method=pi05_rgb   root=/path/to/sim_task2_lerobot/lerobot   training.batch_size=16   policy_name=pi05
```

- **数据集**：`task`、`root` 需与已有 LeRobot 数据集一致（`root` 指向 `.../lerobot` 目录，`task` 与 `dataset.repo_id` 对齐）。
- **策略选择**：通过 `policy_name`（act/diffusion/pi0/pi05）与对应的 `*_config.yaml` 选择策略；PI0 / PI05 需要在 config 中设置 `pretrained_path`（如 `lerobot/pi0_base`、`lerobot/pi05_base` 或本地路径）。
- **输出**：`training.output_directory` 下会生成 `run_<timestamp>/`，其下包含 `epoch*/`、`config.json`、`policy_preprocessor.json` 等。
- **GPU**：通过 `CUDA_VISIBLE_DEVICES` 指定卡号；多卡时可结合 `accelerate launch` 使用 `train_policy_with_accelerate.py`（见 [多卡训练](multi_gpu.md)）。
- **参数详解**：task、method、root、batch_size、policy_name 等说明及配置文件位置见 [配置说明](configs.md)。

---

## 3. 数据加载（make_dataset）

- **调用**：`make_dataset(cfg)`，定义在 `lerobot/datasets/factory.py`。
- **作用**：
  - 根据 `cfg.dataset` 创建 `LeRobotDataset`（或流式数据集），读取 `meta/info.json`、`meta/stats.json`、`meta/episodes/` 及 `data/`、`videos/` 路径模板；
  - 解析特征定义（观测、动作、视频键），供后续 `make_policy` 与 processor 使用。
- **依赖**：数据集需已存在且 `stats.json` 中具备当前策略所需的统计量（如 mean/std；PI0/PI05 需 quantile，见 [LeRobot v4.3 数据集](../dataset/lerobot_v30.md) 中的 quantile 增广脚本）。

---

## 4. 策略构建（make_policy）

- **调用**：`make_policy(cfg=cfg.policy, ds_meta=dataset.meta, rename_map=cfg.rename_map)`，定义在 `lerobot/policies/factory.py`。
- **作用**：
  - 根据 `policy.type` 选择策略类（如从 `lerobot.policies` 或通过第三方插件加载）；
  - Kuavo 项目在 `kuavo_train/wrapper/policy/` 下注册了 **ACT、Diffusion、PI0、PI05** 的 Custom 封装，使策略能正确读取 Kuavo 的 `input_features`（如 `observation.images.head_cam_h`、`observation.state` 等）；
  - 使用 `ds_meta` 中的特征信息与 `policy.config` 的 `input_features`、`output_features` 对齐。
- **结果**：得到可训练的 `PreTrainedPolicy` 实例，其 `config` 中定义了输入输出键与归一化方式（如 `normalization_mapping`）。

---

## 5. 预处理与后处理（make_pre_post_processors）

- **调用**：`make_pre_post_processors(policy_cfg=cfg.policy, pretrained_path=..., dataset_stats=dataset.meta.stats, ...)`，定义在 `lerobot/policies/factory.py`。
- **作用**：
  - 按策略类型组装 **PolicyProcessorPipeline**，常见步骤包括：
    - **device_processor**：将 batch 放到指定设备；
    - **normalizer_processor**：根据 `dataset.meta.stats` 与 `policy.config.normalization_mapping` 对状态/动作做归一化（mean/std 或 quantile）；
    - **rename_observations_processor**：可选键名重映射；
    - 策略特有步骤：如 PI0/PI05 的 **tokenizer_processor**、**observation_processor** 等。
  - 若有 `pretrained_path`，会从已保存的 `policy_preprocessor.json` 等恢复流水线，并用当前 `dataset_stats` 覆盖归一化参数，保证与当前数据集一致。
- **输出**：`preprocessor` 用于训练时对每个 batch 做预处理；`postprocessor` 用于部署时对策略输出做反归一化等，训练循环中不直接使用。

---

## 6. 训练循环

### 6.1 DataLoader 与迭代

- 使用 `dataset` 构建 `torch.utils.data.DataLoader`（可选 `EpisodeAwareSampler` 等）；
- 通过 `accelerator.prepare()` 包装 policy、optimizer、dataloader、lr_scheduler，以支持分布式与混合精度；
- 使用 `cycle(dataloader)` 无限迭代，按 `cfg.steps` 执行固定步数。

### 6.2 单步流程

每一训练步大致为：

1. **取 batch**：`batch = next(dl_iter)`，键与数据集 `info.json` 中定义一致（含 `observation.images.*`、`observation.state`、`action` 及可选的 `observation.depth_*` 等）。
2. **Kuavo 定制（RGB-only）**：在 `lerobot_train.py` 中，若需 **仅用 RGB 训练**，会在此时从 `batch` 中删除 `observation.depth_h`、`observation.depth_l`、`observation.depth_r`，再送入 preprocessor。
3. **预处理**：`batch = preprocessor(batch)`，得到策略 forward 所需格式（已归一化、已转设备、含 token 等）。
4. **前向与反传**：`update_policy(...)` 内调用 `policy.forward(batch)` 计算 loss，`accelerator.backward(loss)`，梯度裁剪，`optimizer.step()`，`lr_scheduler.step()`。
5. **日志与 checkpoint**：按 `log_freq` 打印/WandB 记录；按 `save_freq` 或最后一步保存 checkpoint（policy 权重 + config + preprocessor/postprocessor 配置）。

### 6.3 保存内容

- **checkpoint 目录**（如 `output_dir/run_xxx/epochN/`）：`model.safetensors`、策略相关权重与配置；
- **run 根目录**：`config.json`、`policy_preprocessor.json`（及可选 `policy_postprocessor.json`）等，部署时与 `epochN/` 一起使用，保证预处理与策略一致。

---

## 7. Kuavo 相关约定

### 7.1 RGB-only 训练（当前默认）

- **实现位置**：`lerobot_train.py` 主循环内，在 `preprocessor(batch)` 前删除 `observation.depth_*` 三个键。
- **目的**：与当前部署侧一致（部署时仅输入 RGB + state），避免 depth 通道与 RGB backbone 不匹配等问题。
- **若需恢复 RGB+Depth**：需去掉或条件化这段删除逻辑，并在策略配置与部署中保留 depth 输入；策略与数据格式见 [RGB-深度融合](../advanced/rgb_depth_fusion.md)、[深度支持说明](../advanced/depth_support.md)。

### 7.2 归一化与 quantile

- **ACT / Diffusion**：通常使用 `stats.json` 中的 **mean/std** 做高斯归一化。
- **PI0 / PI05**：使用 **quantile 归一化**（如 q01、q10、q50、q90、q99），需在数据准备阶段运行 `augment_dataset_quantile_stats.py`，使 `stats.json` 中包含相应分位数；否则需在策略配置中指定使用 mean/std 等替代方式（若支持）。

### 7.3 Wrapper 层

- 所有 Kuavo 策略均通过 `kuavo_train/wrapper/policy/<act|diffusion|pi0|pi05>/` 下的封装接入 LeRobot：
  - 保持与 LeRobot 的 `from_pretrained`、`save_pretrained`、`forward`、`select_action` 等接口一致；
  - 在 config 与 forward 中处理多相机、可选 depth、state 键名等，与 [架构说明](../concepts/architecture.md) 中描述一致。

---

## 8. 输出目录与部署衔接

- **典型结构**：`outputs/train/<task>/<method>/run_<timestamp>/`  
  - `epoch1/`、`epoch2/`、…、`epochbest/`：各 checkpoint；  
  - `config.json`、`policy_preprocessor.json` 等：与 run 绑定的配置与处理器。
- **部署**：在 `configs/deploy/kuavo_env.yaml` 中通过 `inference.task`、`inference.method`、`inference.timestamp`、`inference.epoch` 指定上述 run 与 epoch，或直接使用 `inference.checkpoint_run_dir` + `epoch`；加载策略时会同时加载对应 preprocessor，保证观测/动作处理与训练一致。

---

## 9. 相关文档

- [快速开始](../getting_started/quick_start.md)：从数据到训练、仿真评估的最短路径。
- [LeRobot v4.3 数据集](../dataset/lerobot_v30.md)：数据集结构、stats、quantile 增广。
- [配置说明](configs.md)：配置系统与推荐用法。
- [多卡训练](multi_gpu.md)：多卡与 Accelerate。
