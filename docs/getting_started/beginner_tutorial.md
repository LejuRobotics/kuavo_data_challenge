# 零基础完整学习教程

本教程面向**完全零基础**的读者：不了解模仿学习、VLA，不会下载开源模型，也不会配置 Python / ROS / Docker 等复杂环境。按顺序阅读并操作，即可完成从「什么都不懂」到「能训练并部署一个策略」的全流程。

!!! abstract "阅读说明"
    建议**按顺序阅读**。第一部分建立概念，第二至四部分完成环境与依赖，第五至七部分完成数据、训练与部署。若你已有 LeRobot 格式数据且只做训练，可适当跳过 ROS 相关步骤。

---

## 你将学到什么

完成本教程后，你将能够：

| 目标 | 内容 |
|------|------|
| **理解概念** | 模仿学习、VLA 是什么；ACT / Diffusion / PI0 / PI05 的区别与选型 |
| **准备环境** | 安装 Python、Conda、可选 GPU/CUDA；按需安装 ROS 或使用 Docker |
| **获取资源** | 克隆仓库、下载赛事数据、按需下载 HuggingFace 预训练模型（PI0/PI05） |
| **安装依赖** | 创建 Conda 环境、安装项目依赖并验证 |
| **准备数据** | 理解 LeRobot 格式；rosbag 转 LeRobot；PI0/PI05 的 quantile 统计 |
| **训练策略** | 用一条命令训练 ACT、Diffusion 或 PI0/PI05，并找到输出目录 |
| **部署测试** | 在仿真中加载模型并自动测试；了解真机部署入口 |

---

# 第一部分：概念入门（零基础必读）

在动手装环境、跑代码之前，先建立一点直觉，避免「不知道自己在跑什么」。

---

## 1.1 模仿学习（Imitation Learning）是什么？

**通俗理解**：让机器人「照着人类（或专家）做过的动作学」。我们事先录好很多次「人类操作机器人完成某任务」的数据（例如：相机画面 + 当时机器人的关节角度、夹爪开合等），然后训练一个模型：**输入**是当前的相机画面和机器人状态，**输出**是机器人接下来该怎么动。模型学会的是「在类似画面下，做出和专家类似的动作」，而不是自己瞎探索，所以叫「模仿」学习。

**在本项目中：**

- **输入**：多路相机图像（RGB）、机器人状态（关节角、夹爪等），有的策略还支持深度图、语言指令。
- **输出**：一段动作序列（例如接下来 50 步的关节目标），控制机器人执行。
- **数据**：来自 Kuavo 机器人的演示数据（rosbag 或已转换好的 LeRobot 格式）。

---

## 1.2 VLA（Vision-Language-Action）是什么？

**通俗理解**：一种「看 + 说 + 动」的模型。既能看图像（Vision），又能理解或生成语言（Language），还能输出机器人动作（Action）。例如：给一张场景图 + 一句「把红色积木放到盒子里」，模型输出机器人该如何动作。

**在本项目中：**

- **PI0、PI05** 属于 VLA 类策略：底层用视觉-语言大模型（如 PaliGemma）理解图像（和可选的语言指令），再通过一个「专家头」把表示转换成具体动作。
- **ACT、Diffusion** 则主要依赖视觉 + 状态，不强制需要语言；可以理解为「纯视觉-动作」策略。

!!! tip "选型建议"
    若你暂时不碰语言指令，只做「看画面 → 出动作」，用 **ACT** 或 **Diffusion** 即可；想用大模型、或以后加语言条件，再选 **PI0 / PI05**。

---

## 1.3 本项目的几种策略（用谁、何时要下载模型）

| 策略 | 通俗理解 | 是否需要额外下载预训练模型 | 适合人群 |
|------|----------|----------------------------|----------|
| **ACT** | 用 Transformer 看多路图像+状态，直接预测一段动作序列。 | **不需要** | 零基础首选，环境简单 |
| **Diffusion** | 用扩散模型「去噪」生成动作序列，适合复杂、多峰的动作分布。 | **不需要** | 零基础可选，效果往往更好 |
| **PI0** | 大视觉-语言模型（PaliGemma）+ 专家头，可加语言条件。 | **需要**：PaliGemma + pi0_base | 有 GPU、会下载 HuggingFace 模型时用 |
| **PI05** | 与 PI0 类似，接口为 OpenPI 风格。 | **需要**：PaliGemma + pi05_base | 同上 |

!!! success "零基础建议"
    完全零基础时，**先选 ACT 或 Diffusion**，不用下载大模型，只要准备好数据和环境即可训练。熟悉流程后，再按需尝试 PI0/PI05 并学习如何下载预训练模型。

---

## 1.4 整体流程一览

```
数据（rosbag 或已有 LeRobot 数据集）
    → 如需则转换格式 / 生成 quantile
    → 选择策略（ACT / Diffusion / PI0 / PI05）
    → 训练（若有 PI0/PI05 需先下载对应预训练模型）
    → 得到 checkpoint（模型权重 + 配置）
    → 在仿真或真机中加载该 checkpoint 做推理与测试
```

下面从环境开始，一步步做。

---

# 第二部分：环境准备（从零安装）

你需要准备：**操作系统**、**Python 环境**、**（可选）GPU/CUDA**、以及**（仅仿真/真机或做数据转换时）ROS 或 Docker**。

---

## 2.1 系统要求

- **推荐**：Ubuntu 20.04 或 22.04（64 位）。
- 若使用 22.04/24.04 且需要仿真或与真机一致的环境，建议用 **Docker** 跑 ROS 与仿真器（见后文）。
- **内存**：建议 16GB 及以上；训练时 8GB 显存的 GPU 可跑小 batch，更大显存更顺畅。

---

## 2.2 安装 Miniconda（用 Conda 管理 Python）

Conda 用来创建「独立环境」，避免和系统自带的 Python 或其他项目冲突。

**步骤 1：下载 Miniconda 安装脚本**

打开终端，执行（以 Linux x86_64 为例）：

```bash
cd ~
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
```

!!! note "国内网络"
    若下载较慢，可使用清华镜像：  
    `wget https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/Miniconda3-latest-Linux-x86_64.sh`

**步骤 2：安装**

```bash
bash Miniconda3-latest-Linux-x86_64.sh
```

按提示操作（一路回车可接受默认）。若询问「Do you wish to initialize Miniconda3?」选 **yes**，这样每次开终端会自动进入 base 环境。

**步骤 3：生效并验证**

安装完成后执行一次 `source ~/.bashrc` 或新开终端，然后：

```bash
conda --version
```

应能看到版本号。

---

## 2.3 创建本项目的 Python 环境

在项目根目录下（或任意你放代码的目录）执行：

```bash
conda create -n kdc python=3.10 -y
conda activate kdc
```

- `kdc` 是环境名称，可改成别的（如 `kuavo`）。
- 之后每次要训练或运行本项目，先执行：`conda activate kdc`。

验证 Python 版本：

```bash
python --version
# 应显示 Python 3.10.x
```

---

## 2.4 （可选）NVIDIA 驱动与 CUDA——仅训练/推理需要 GPU 时

若你**没有 NVIDIA 显卡**或暂时不用 GPU，可以仅用 CPU 训练（非常慢，仅适合验证流程）。若有显卡并希望加速，需要：

1. **安装 NVIDIA 驱动**（如 535 等，以你显卡和系统为准）。
2. **安装 CUDA Toolkit**（本项目建议 CUDA 11.7 及以上）。

安装驱动示例（Ubuntu，仅供参考，请以官方文档为准）：

```bash
ubuntu-drivers devices
sudo apt install nvidia-driver-535
sudo reboot
```

重启后验证：`nvidia-smi`，能看到显卡信息即表示驱动正常。CUDA 的安装可参考 [NVIDIA 官方文档](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/)。若使用 Conda，也可用 `conda install cuda -c nvidia` 按需选择。

---

## 2.5 ROS 是什么？什么时候需要？

**ROS**（Robot Operating System）是一套机器人开发框架，本项目中：

| 场景 | 是否需要 ROS |
|------|----------------|
| **数据转换** | 乐聚原始数据是 **rosbag**，转成 LeRobot 才能训练；转换脚本会用到 ROS 相关包，因此**做 rosbag → LeRobot 转换时需要本机已安装 ROS**（或使用带 ROS 的 Docker 环境）。 |
| **仅做训练** | 若你拿到的是**已经转换好的 LeRobot 数据集**，可以直接用 `train_policy.py` 训练，此时**可以不装 ROS**，只要 Python 和项目依赖即可。 |
| **仿真测试** | 需要**启动仿真器**，仿真器通常依赖 ROS；本机需装 ROS Noetic 或在 Docker 里用带 ROS 的镜像。 |
| **真机部署** | 真机环境一般是 Ubuntu 20.04 + ROS Noetic，本机若也要和真机联调，需要 ROS。 |

!!! tip "结论"
    若你**只有 rosbag、需要自己转换**，则要先装 ROS（或用 Docker）；若**已有转换好的 LeRobot 数据**，可先不装 ROS，直接训练。到「仿真/真机部署」时再按 [安装指南](installation.md) 或赛事文档安装 ROS 或使用 Docker。

---

## 2.6 Docker 是什么？什么时候需要？

**Docker** 可以把一整块环境（操作系统、ROS、Python、CUDA 等）打包成「镜像」，在容器里运行，避免和本机环境冲突。

- **仅训练**：不强制需要 Docker，用 Conda 即可。
- **本机是 Ubuntu 22.04/24.04 又要跑 ROS 仿真**：往往用 **Docker + ROS Noetic 镜像** 更省事。
- **赛事/评测方提供 Docker 镜像**：按赛事文档在 Docker 里跑即可。

Docker 的安装可参考 [Docker 官方文档](https://docs.docker.com/engine/install/ubuntu/)。若你当前只做「克隆代码 → 装依赖 → 训练」，可**先不装 Docker**。

---

## 2.7 小结：你至少需要什么

| 目标 | 至少需要 |
|------|----------|
| **只训练（已有 LeRobot 数据）** | Ubuntu + Conda + Python 3.10 + 本项目依赖；GPU 建议有；**不需 ROS** |
| **从 rosbag 转换再训练** | 同上 + **ROS**（转换脚本依赖 ROS 包） |
| **训练 PI0/PI05** | 同上 + 能访问 HuggingFace 并下载 PaliGemma、pi0_base/pi05_base |
| **仿真测试** | 同上 + ROS 或 Docker（仿真器）+ 仿真器已启动 |
| **真机部署** | 真机环境（一般为 Ubuntu 20.04 + ROS Noetic）+ 本机或边侧机上的部署脚本与配置 |

下面假设你已经有了 **Ubuntu + Conda + 已创建并激活的 kdc 环境**，继续「获取代码与数据」。

---

# 第三部分：获取代码与数据

---

## 3.1 克隆本仓库并初始化子模块

打开终端，进入你打算放代码的目录，执行：

```bash
git clone https://github.com/LejuRobotics/kuavo_data_challenge.git
cd kuavo_data_challenge
```

若赛事或开发使用 **dev 分支**，请切换并更新子模块：

```bash
git fetch origin
git checkout dev
git submodule init
git submodule update --recursive --progress
```

这样会拉取 `third_party/lerobot` 等子模块，训练脚本依赖它们。

---

## 3.2 赛事数据从哪里下载？

- 数据与赛题说明以**赛事主办方**为准。
- **常见入口**：
  - **天池大赛**：[乐聚第一届具身智能操作任务挑战赛](https://tianchi.aliyun.com/competition/entrance/532415) 页面中的「数据集」或说明。
  - **赛事官方文档**：[https://kdc-doc.netlify.app/tianchi/cn/](https://kdc-doc.netlify.app/tianchi/cn/) 中会有数据下载、格式说明等。

!!! note "数据格式"
    下载后，若数据是 **rosbag**，需要转换成 LeRobot 格式（见第五部分）；若主办方已提供 **LeRobot 格式**（包含 `meta/info.json`、`meta/stats.json`、`data/`、`videos/` 等），可直接用于训练。

---

## 3.3 预训练模型是什么？何时需要下载？

- **ACT、Diffusion**：从零训练，**不需要**额外下载预训练模型。
- **PI0、PI05**：基于大模型 + 专家头，需要先有：
  - **PaliGemma**（如 `google/paligemma-3b-pt-224`）：视觉-语言底座，含 tokenizer；
  - **pi0_base** 或 **pi05_base**（LeRobot 在 HuggingFace 上的预训练策略权重）。

下面说明如何下载这些开源模型。

---

## 3.4 如何下载 HuggingFace 上的开源模型（PI0/PI05 用）

### 步骤 1：注册 Hugging Face 账号并同意条款

1. 打开 [https://huggingface.co](https://huggingface.co) 注册账号。
2. 使用 **PaliGemma** 前，需在模型页同意 Google 使用条款：  
   [google/paligemma-3b-pt-224](https://huggingface.co/google/paligemma-3b-pt-224) → 登录后点击 「Agree and access repository」。

### 步骤 2：创建 Access Token

1. 登录后点击头像 → **Settings** → **Access Tokens**。
2. 新建一个 Token，权限选 **Read** 即可。
3. 复制该 Token（只显示一次，请保存好）。

### 步骤 3：在本机登录 Hugging Face

在已激活的 Conda 环境中执行：

```bash
pip install huggingface_hub
huggingface-cli login
```

按提示粘贴刚才的 Token，回车。成功后，后续 `huggingface-cli download` 或代码中的 `snapshot_download` 会使用该账号。

### 步骤 4：下载 PaliGemma（PI0/PI05 的 tokenizer 与视觉编码器）

```bash
# 指定一个目录存放，例如项目下的 models
mkdir -p /path/to/kuavo_data_challenge/models
huggingface-cli download google/paligemma-3b-pt-224 \
  --local-dir /path/to/kuavo_data_challenge/models/paligemma-3b-pt-224 \
  --local-dir-use-symlinks False
```

请把 `/path/to/kuavo_data_challenge` 换成你的实际项目路径。部署时若 tokenizer 路径与训练时不一致，可设置环境变量：

```bash
export PALIGEMMA_TOKENIZER_PATH=/path/to/kuavo_data_challenge/models/paligemma-3b-pt-224
```

### 步骤 5：下载 pi0_base 或 pi05_base

**方式 A：训练时直接写 Hub ID（推荐，首次会自动下载）**

在训练命令或配置里写：

- PI0：`pretrained_path=lerobot/pi0_base`
- PI05：`pretrained_path=lerobot/pi05_base`

**方式 B：预先下载到本地再指定路径**

```bash
# PI0
huggingface-cli download lerobot/pi0_base \
  --local-dir /path/to/kuavo_data_challenge/models/pi0_base \
  --local-dir-use-symlinks False

# PI05
huggingface-cli download lerobot/pi05_base \
  --local-dir /path/to/kuavo_data_challenge/models/pi05_base \
  --local-dir-use-symlinks False
```

训练时把 `pretrained_path` 设为上述本地路径即可。

!!! warning "国内网络 / 离线"
    **国内**：可配置 Hugging Face 镜像或代理；或先在能访问外网的机器下载好，再把 `models/` 目录拷到本机，并设置 `HF_HUB_OFFLINE=1` 使用本地。  
    **完全离线**：在能联网的机器下载完整模型目录后，拷贝到离线环境，配置中 `pretrained_path` 指向本地路径，并设置 `HF_HUB_OFFLINE=1`。

---

# 第四部分：安装项目依赖

在项目根目录下，确保已激活 Conda 环境（`conda activate kdc`），然后执行下面步骤。

---

## 4.1 配置 pip 源（可选，国内建议）

加速安装：

```bash
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
```

---

## 4.2 安装依赖（二选一）

**仅做训练、且已有 LeRobot 数据**（不做 rosbag 转换、不做仿真/真机时，可不装 ROS）：

```bash
pip install -r requirements_ilcode.txt
```

**需要数据转换、仿真或真机部署**（依赖 ROS 等）：

```bash
# 若本机已装 ROS Noetic，可先执行
source /opt/ros/noetic/setup.bash

pip install -r requirements_total.txt
```

!!! note "如何选"
    若系统没有 ROS，且你已有 LeRobot 格式数据、不需要做 rosbag 转换，仅训练时可只装 `requirements_ilcode.txt`；若需要做 rosbag → LeRobot 转换，需先安装 ROS 再装 `requirements_total.txt`。

---

## 4.3 以「可编辑」方式安装本项目

这样可以直接改代码并生效，且命令行能用到本项目包：

```bash
cd /path/to/kuavo_data_challenge
pip install -e .
```

---

## 4.4 验证安装

```bash
python -c "import torch; print(torch.__version__)"
python -c "import lerobot; print(lerobot.__version__)"
```

若无报错并能看到版本号，说明 PyTorch 和 LeRobot 已装好。若有 GPU，可再测：

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

输出 `True` 表示当前环境能用 CUDA。

---

# 第五部分：数据准备

---

## 5.1 数据格式简要说明

- **rosbag**：ROS 的录包格式，里面是相机、关节、夹爪等话题的原始数据。
- **LeRobot 格式**：本项目训练使用的统一格式，包含：
  - `meta/info.json`：数据集元信息、特征定义；
  - `meta/stats.json`：均值/方差或分位数，用于归一化；
  - `data/`、`videos/`：按 chunk 存的状态、动作和视频。

若你拿到的**已经是 LeRobot 格式**（例如赛事提供的已转换数据），只需确认路径中有 `meta/info.json`、`meta/stats.json` 以及对应的 `data/`、`videos/`，即可在训练时用 `root=/path/to/lerobot` 指向该目录（指向包含 `meta` 的那一层，即 `.../lerobot`）。

---

## 5.2 若数据是 rosbag：转换为 LeRobot

若主办方给的是 rosbag，需要用本项目提供的脚本转成 LeRobot。示例：

```bash
cd /path/to/kuavo_data_challenge

python kuavo_data/CvtRosbag2Lerobot.py \
  --config-path=../configs/data/ \
  --config-name=KuavoRosbag2Lerobot.yaml \
  rosbag.rosbag_dir=/path/to/your/rosbag \
  rosbag.lerobot_dir=/path/to/output/lerobot_data
```

- `rosbag.rosbag_dir`：放 rosbag 的目录。
- `rosbag.lerobot_dir`：转换结果输出目录（通常下面会生成 `lerobot` 子目录，训练时 `root` 指到该 `lerobot` 目录）。

具体配置（启用哪些相机、是否用深度等）见 `configs/data/KuavoRosbag2Lerobot.yaml`。

---

## 5.3 使用 PI0 或 PI05 时：生成 Quantile 统计

PI0/PI05 若使用 **quantile 归一化**，需先在数据集上跑一遍统计脚本，把分位数写入 `meta/stats.json`。例如：

```bash
python third_party/lerobot/src/lerobot/datasets/v30/augment_dataset_quantile_stats.py \
  --repo-id sim_task2_lerobot \
  --root /path/to/sim_task2_lerobot/lerobot
```

`--repo-id` 需与数据集 `meta/info.json` 中的命名一致，`--root` 指向包含 `meta` 的 `lerobot` 目录。若你用的数据集名称不是 `sim_task2_lerobot`，请改成实际名称。

---

# 第六部分：训练你的第一个策略

这里以 **ACT** 和 **Diffusion** 为例（不需预训练模型）；PI0/PI05 只需多一步「指定 pretrained_path」即可。

---

## 6.1 准备路径与参数

- 假设你的 LeRobot 数据在：`/data/sim_task2_lerobot/lerobot`（即该目录下有 `meta/`、`data/`、`videos/`）。
- 训练输出会写在项目下的 `outputs/train/` 中，例如 `outputs/train/sim_task2/act_rgb/run_20260101_120000/`。

下面命令中的 `root`、`task`、`method` 需和你的数据一致：`task` 一般与数据集名称或 `repo_id` 对应，`method` 可自定义，用于区分同一任务下不同实验。

---

## 6.2 训练 ACT

单卡（例如 0 号 GPU）：

```bash
cd /path/to/kuavo_data_challenge
conda activate kdc

CUDA_VISIBLE_DEVICES=0 python kuavo_train/train_policy.py \
  --config-path=../configs/policy/ \
  --config-name=act_config.yaml \
  task=sim_task2 \
  method=act_rgb \
  root=/data/sim_task2_lerobot/lerobot \
  training.batch_size=32 \
  policy_name=act
```

把 `root` 换成你的数据集路径；`task`/`method` 按需改。训练结束后，模型与配置在：

`outputs/train/sim_task2/act_rgb/run_<时间戳>/`

其下有 `epoch1/`、`epoch2/`、…、`epochbest/` 以及 `config.json`、`policy_preprocessor.json` 等。

---

## 6.3 训练 Diffusion

```bash
CUDA_VISIBLE_DEVICES=0 python kuavo_train/train_policy.py \
  --config-path=../configs/policy/ \
  --config-name=diffusion_config.yaml \
  task=sim_task2 \
  method=diffusion_rgb \
  root=/data/sim_task2_lerobot/lerobot \
  training.batch_size=32 \
  policy_name=diffusion
```

同样，修改 `root`、`task`、`method` 以匹配你的数据。输出目录结构同 ACT。

---

## 6.4 训练 PI0（需先有 pi0_base 或 PaliGemma + pi0_base）

若已下载 pi0_base 到本地 `/path/to/models/pi0_base`，或打算直接用 Hub ID 自动下载：

```bash
CUDA_VISIBLE_DEVICES=0 python kuavo_train/train_policy.py \
  --config-path=../configs/policy/ \
  --config-name=pi0_rgb_config.yaml \
  task=sim_task2 \
  method=pi0_rgb \
  root=/data/sim_task2_lerobot/lerobot \
  training.batch_size=16 \
  policy_name=pi0
```

若用本地路径，需在对应 config 中设置 `policy.pretrained_path=/path/to/models/pi0_base`，或在命令行覆盖（具体见 [训练流水线](../training/pipeline.md) 和 [PI0 策略说明](../policy/pi0.md)）。PI05 同理，使用 `pi05_config.yaml` 和 `policy_name=pi05`，并设置 `pretrained_path` 为 `lerobot/pi05_base` 或本地路径。

---

## 6.5 如何判断训练是否正常

- 终端有 loss 打印，且 loss 随 step 总体下降或波动下降。
- `outputs/train/<task>/<method>/run_<时间戳>/` 下出现 `epoch1/`、`epoch2/` 等，且内有 `model.safetensors` 等文件。
- 若报错：先看报错信息（缺少文件、显存不足、路径错误等），再对照 [常见问题](../faq.md) 与 [训练流水线](../training/pipeline.md)。

---

# 第七部分：仿真与真机部署

训练完成后，得到的是「run 目录 + 某个 epoch」，例如：

`outputs/train/sim_task2/act_rgb/run_20260101_120000/epochbest/`

要在仿真里测试，需要：**仿真器已启动**（通常依赖 ROS），且本机能与仿真器通信（同一 ROS master）。

---

## 7.1 修改部署配置指向你的训练结果

编辑 `configs/deploy/kuavo_env.yaml`，设置推理相关字段，例如：

```yaml
inference:
  policy_type: act   # 或 diffusion / pi0 / pi05
  task: sim_task2
  method: act_rgb
  timestamp: run_20260101_120000   # 你的 run 目录名
  epoch: best
  eval_episodes: 10
  device: cuda
```

若 run 不在默认的 `outputs/train/` 下，可用 `checkpoint_run_dir` 直接指定 run 的绝对路径（详见 [仿真自动测试](../deployment/sim_auto_test.md)）。

---

## 7.2 启动仿真器（按赛事/仓库说明）

仿真器一般由赛事文档或 **kuavo-ros-opensource** 等仓库提供，需先单独启动（如 MuJoCo 仿真 + ROS）。确保：

- 本机与仿真器使用同一 `ROS_MASTER_URI`；
- 仿真器已提供 `/simulator/reset`、`/simulator/start`、`/simulator/success` 等服务/话题。

---

## 7.3 运行自动测试脚本

在项目根目录、已激活 Conda 环境且 ROS 已 source 的情况下：

```bash
python kuavo_deploy/src/scripts/script_auto_test.py \
  --task auto_test \
  --config configs/deploy/kuavo_env.yaml
```

脚本会加载你配置的 checkpoint、与仿真器通信、跑若干 episode 并统计成功率。更多参数与故障排查见 [仿真自动测试](../deployment/sim_auto_test.md)。

---

## 7.4 真机部署

真机一般需在机器人本机或边侧机配置环境（如 Ubuntu 20.04 + ROS Noetic），并修改 `kuavo_env.yaml` 中 `env_name` 等为真机配置。完整步骤见 [真机评测](../deployment/real_eval.md) 与赛事/仓库的 README。

---

# 第八部分：常见问题与求助

!!! info "报错时先看"
    终端完整报错信息；是否路径写错、显存不足（可减小 `batch_size`）、缺依赖（对照 [安装指南](installation.md)）、数据缺 `meta/stats.json` 或 quantile（PI0/PI05）。

**文档入口：**

| 文档 | 说明 |
|------|------|
| [安装指南](installation.md) | 环境要求与 Conda 创建 |
| [快速开始](quick_start.md) | 从数据到训练与仿真的最短路径 |
| [训练流水线](../training/pipeline.md) | 数据流、入口脚本与配置 |
| [仿真自动测试](../deployment/sim_auto_test.md) | 仿真评估与配置说明 |
| [真机评测](../deployment/real_eval.md) | 真机部署步骤 |
| [策略概览](../concepts/policy_overview.md) | 各策略对比与选型 |
| [常见问题](../faq.md) | 常见报错与排查 |

**赛事与社区：** 报名与规则以 [天池赛事页](https://tianchi.aliyun.com/competition/entrance/532415) 与 [赛事文档](https://kdc-doc.netlify.app/tianchi/cn/) 为准；技术讨论可加赛事 QQ 群或 [OpenLET 社区](https://openlet.openatom.tech/)。

---

按本教程顺序做完一遍，你就完成了从零到「能训练并部署一个策略」的完整路径；后续再根据需要深入数据格式、各策略参数和真机细节即可。
