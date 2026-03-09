# LeRobot v4.3 数据集说明

本项目使用 **LeRobot v4.3** 作为统一的训练数据格式。所有从 ROS / 仿真采集的原始数据在参与训练前，需转换为该格式并生成相应元数据与统计信息。

---

## 1. 目录结构

数据集根目录（例如 `sim_task2_lerobot/lerobot`）下典型结构为：

```
lerobot/
├── meta/
│   ├── info.json          # 数据集元信息：版本、机器人类型、特征定义、路径模板等
│   ├── stats.json         # 统计信息：mean / std / quantiles，用于归一化
│   ├── episodes/          # 按 chunk 存放的 episode 元数据（parquet）
│   │   ├── chunk-000/
│   │   │   ├── file-000.parquet
│   │   │   └── ...
│   │   └── ...
│   └── tasks.parquet      # 任务描述（可选）
├── data/                  # 状态、动作等非视频数据（按 info.json 中的 data_path 模板）
│   ├── chunk-000/
│   │   ├── file-000.parquet
│   │   └── ...
│   └── ...
└── videos/                # 视频流（按 info.json 中的 video_path 模板）
    ├── observation.images.head_cam_h/
    │   ├── chunk-000/
    │   │   ├── file-000.mp4
    │   │   └── ...
    │   └── ...
    ├── observation.images.wrist_cam_l/
    ├── observation.images.wrist_cam_r/
    ├── observation.depth_h/
    ├── observation.depth_l/
    └── observation.depth_r/
```

- **meta/**：所有元数据与索引，训练脚本和策略依赖此处读取特征定义、归一化参数等。
- **data/**：每帧的状态、动作、时间戳等表格数据，以 Parquet 分 chunk 存储。
- **videos/**：按特征键分目录存储的视频（如多视角 RGB、深度），便于流式读取。

---

## 2. meta/info.json

`info.json` 描述数据集的版本、规模、特征与存储路径模板。

### 2.1 顶层字段

| 字段 | 含义 |
|------|------|
| `codebase_version` | 数据集格式版本（如 `"v3.0"`），需与 LeRobot 库兼容 |
| `robot_type` | 机器人类型（如 `"kuavo4pro"`） |
| `total_episodes` | 总 episode 数 |
| `total_frames` | 总帧数 |
| `total_tasks` | 任务数（多任务时使用） |
| `fps` | 采集帧率（如 10） |
| `chunks_size` | 每个 chunk 的 episode 数量 |
| `data_path` | 数据文件路径模板，如 `"data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"` |
| `video_path` | 视频文件路径模板，如 `"videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"` |
| `splits` | 训练/验证划分，如 `{"train": "0:300"}` |
| `features` | 各特征的名称、类型、shape、命名等，见下 |

### 2.2 features 定义

每个 key 对应一个特征（观测或动作），常见包括：

- **observation.state**：机器人状态向量，`dtype: float32`，`shape: [state_dim]`（如 16 维关节+夹爪）；可含 `names.state_names` 等。
- **action**：动作向量，`dtype: float32`，`shape: [action_dim]`，可含 `names.action_names`。
- **observation.images.\***：RGB 图像，`dtype: "video"`，`shape: [C, H, W]`（如 `[3, 480, 640]`），`info` 中包含 `video.height`、`video.width`、`video.codec`、`video.is_depth_map` 等。
- **observation.depth_***：深度图（若存在），同样为 `dtype: "video"`，与 RGB 结构类似。
- **timestamp**、**frame_index**、**episode_index**、**index**、**task_index**：标量或单维，用于索引与对齐。

Kuavo 典型配置会包含：`observation.state`、`action`、三路 RGB（如 `head_cam_h`、`wrist_cam_l`、`wrist_cam_r`）、以及可选的三路深度（`depth_h`、`depth_l`、`depth_r`）。

---

## 3. meta/stats.json

`stats.json` 存储各特征的统计量，供训练时的归一化/反归一化使用。

### 3.1 内容结构

- 以**特征名为 key**，每个特征下为统计字典，例如：
  - **mean**、**std**：用于高斯归一化（ACT、Diffusion 等常用）。
  - **q01、q10、q50、q90、q99**：分位数，用于 **quantile 归一化**（PI0、PI05 需要）。

示例（片段）：

```json
{
  "observation.state": { "mean": [...], "std": [...] },
  "action": { "mean": [...], "std": [...], "q01": [...], "q10": [...], "q50": [...], "q90": [...], "q99": [...] }
}
```

### 3.2 Quantile 统计与 PI0 / PI05

- 若使用 **PI0 或 PI05**，训练前需确保数据集中含有 **quantile 统计**（q01、q10、q50、q90、q99）。
- 若现有数据集仅有 mean/std、没有 quantiles，可运行 LeRobot 提供的增广脚本：

```bash
python third_party/lerobot/src/lerobot/datasets/v30/augment_dataset_quantile_stats.py \
  --repo-id <your_repo_id> \
  --root /path/to/your/lerobot/data/lerobot
```

- 脚本会检查 `stats.json`，若已存在 quantile 则跳过；否则会计算并写回 `meta/stats.json`。详见 [快速开始](../getting_started/quick_start.md) 与 [训练流水线](../training/pipeline.md)。

---

## 4. 数据与视频存储

- **data/** 下的 Parquet 文件按 `info.json` 的 `data_path` 模板组织；每帧对应一行，包含观测、动作及索引列。
- **videos/** 下按 `video_path` 模板组织：每个视频特征一个子目录，其下再按 chunk/file 存放 MP4（或配置的编解码格式）。  
- 训练时通过 LeRobot 的 `LeRobotDataset` 按需加载帧与视频片段，无需一次性读入全部数据。

---

## 5. 从 ROS / 仿真生成数据集

- 将 ROS bag 或仿真录制数据转换为 LeRobot 格式的入口在 **kuavo_data** 模块，例如：
  - `kuavo_data/CvtRosbag2Lerobot.py`：将 Kuavo 的 rosbag 转为 LeRobot 数据集，生成 `meta/info.json`、`meta/stats.json` 及 `data/`、`videos/` 内容。
- 转换完成后，请确认：
  - `meta/info.json` 中 `codebase_version` 与当前 LeRobot 兼容；
  - `meta/stats.json` 存在且包含所需 mean/std（以及 PI0/PI05 所需的 quantiles，若未包含则运行上述 quantile 增广脚本）。

---

## 6. 训练时的使用

- 训练脚本（如 `lerobot_train.py`）通过 `--dataset.repo_id` 与 `--dataset.root` 指定数据集；会读取 `meta/info.json` 与 `meta/stats.json`，构建 DataLoader 与 Processor Pipeline。
- 不同策略对输入键的约定不同（如仅 RGB、或 RGB+Depth、或 state+语言）：在策略配置的 `input_features` 中指定，数据集只需提供相应键存在且格式符合 `info.json` 即可。

更多细节请参考：

- [训练流水线](../training/pipeline.md)
- [快速开始](../getting_started/quick_start.md)
- [配置说明](../training/configs.md)（训练配置与参数说明）
