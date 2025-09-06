
# 🚀 **Kuavo Data Challenge**

> 具身智能操作任务挑战赛 | 乐聚机器人·北京通用人工智能研究院 | [2025/09 2026/03]

![项目徽章](https://img.shields.io/badge/比赛-天池竞赛-blue) 
![构建状态](https://img.shields.io/badge/build-passing-brightgreen)

---

## 🌟 项目简介
本仓库基于 [Lerobot](https://github.com/huggingface/lerobot) 开发，结合乐聚 Kuavo（夸父）机器人，提供 **数据格式转换**（rosbag → parquet）、**模仿学习（IL）训练**、**仿真器测试**以及**真机部署验证**的完整示例代码。

**关键词**：具身智能 · 工业制造 · 阿里云天池竞赛

---

## 🎯 比赛目标
  
- 使用本仓库代码熟悉 Kuavo 机器人数据格式，完成模仿学习模型的训练与测试。 
- 围绕主办方设定的机器人操作任务，开发具备感知与决策能力的模型。 
- 最终目标及评价标准以赛事官方说明文档为准。  

---

## ✨ 核心功能
- 数据格式转换模块（rosbag → Lerobot parquet）  
- IL 模型训练框架 (diffusion policy, ACT)
- Mujoco 模拟器支持  
- 真机验证与部署  

⚠️ 注意：本示例代码尚未支持末端控制，目前只支持关节角控制！

---

## ♻️ 环境要求
- **系统**：推荐 Ubuntu 20.04（22.04 / 24.04 建议使用 Docker 容器运行）  
- **Python**：推荐 Python 3.10  
- **ROS**：ROS Noetic + Kuavo Robot ROS 补丁（支持 Docker 内安装）  
- **依赖**：Docker、NVIDIA CUDA Toolkit（如需 GPU 加速）  

---

## 📦 安装指南

### 1. 操作系统环境配置
推荐 **Ubuntu 20.04 + NVIDIA CUDA Toolkit + Docker**。  
<details>
<summary>详细步骤（展开查看），仅供参考</summary>

#### a. 安装操作系统与 NVIDIA 驱动
```bash
sudo apt update
sudo apt upgrade -y
ubuntu-drivers devices
# 测试通过版本为 535，可尝试更新版本（请勿使用 server 分支）
sudo apt install nvidia-driver-535
# 重启计算机
sudo reboot
# 验证驱动
nvidia-smi
```

#### b. 安装 NVIDIA Container Toolkit

```bash
sudo apt install curl
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg \
  --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
  && curl -s -L \
  https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb \
   [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update
export NVIDIA_CONTAINER_TOOLKIT_VERSION=1.17.8-1
sudo apt-get install -y \
    nvidia-container-toolkit=${NVIDIA_CONTAINER_TOOLKIT_VERSION} \
    nvidia-container-toolkit-base=${NVIDIA_CONTAINER_TOOLKIT_VERSION} \
    libnvidia-container-tools=${NVIDIA_CONTAINER_TOOLKIT_VERSION} \
    libnvidia-container1=${NVIDIA_CONTAINER_TOOLKIT_VERSION}
```

#### c. 安装 Docker

```bash
sudo apt update
sudo apt install git
sudo apt install docker.io
# 配置 NVIDIA Runtime
nvidia-ctk
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
sudo docker info | grep -i runtime
# 输出中应包含 "nvidia" Runtime
```

</details>

---

### 2. ROS 环境配置

kuavo mujoco 仿真与真机运行均基于 **ROS Noetic**环境，由于真机kuavo机器人是ubuntu20.04 + ROS Noetic（非docker），因此推荐直接安装 ROS Noetic，若因ubuntu版本较高无法安装 ROS Noetic，可使用docker。

<details>
<summary>a. 系统直接安装 ROS Noetic（推荐）</summary>

* 官方指南：[ROS Noetic 安装](http://wiki.ros.org/noetic/Installation/Ubuntu)
* 国内加速源推荐：[小鱼ROS](https://fishros.org.cn/forum/topic/20/)

安装示例：

```bash
wget http://fishros.com/install -O fishros && . fishros
# 菜单选择：5 配置系统源 → 2 更换源并清理第三方源 → 1 添加ROS源
wget http://fishros.com/install -O fishros && . fishros
# 菜单选择：1 一键安装 → 2 不更换源安装 → 选择 ROS1 Noetic 桌面版
```

测试 ROS 安装：

```bash
roscore  # 新建终端
rosrun turtlesim turtlesim_node  # 新建终端
rosrun turtlesim turtle_teleop_key  # 新建终端
```

</details>

<details>
<summary>b. 使用 Docker 安装 ROS Noetic</summary>

有两种方法可选：

**方法一：kuavo仿真器文档（推荐）**
阅读 [readme for simulator](https://github.com/LejuRobotics/kuavo-ros-opensource/blob/opensource/kuavo-data-challenge/readme.md)，包含镜像构建与mujoco仿真配置的完整说明。

**方法二：直接下载并导入镜像**

```bash
wget https://kuavo.lejurobot.com/docker_images/kuavo_opensource_mpc_wbc_img_v0.6.1.tar.gz
sudo docker load -i kuavo_opensource_mpc_wbc_img_v0.6.1.tar.gz
```
</details>

<br>
⚠️ 警告：如果上述中ROS使用的是docker环境，下方后续的代码可能需要在容器里面运行，如有问题，请核对当前是否在容器内！

---

### 3. 克隆代码

```bash
# SSH
git clone --depth=1 git@github.com:LejuRobotics/kuavo-data-challenge.git

# HTTPS
git clone --depth=1 https://github.com/LejuRobotics/kuavo-data-challenge.git
```

更新third_party下的lerobot子模块：

```bash
cd kuavo-data-challenge
git submodule init
git submodule update --recursive
```

---

### 4. Python 环境配置

使用 conda （推荐）或 python venv 创建虚拟环境（推荐 python 3.10）：

```bash
conda create -n kdc python=3.10
conda activate kdc
```

或：

```bash
python -m venv kdc
source kdc/bin/activate
```

安装依赖：

```bash
pip install -r requirements_ilcode.txt   # 推荐
# 或
pip install -r requirements_total.txt    # 需确保 ROS Noetic 已安装
```

如果运行时报ffmpeg或torchcodec的错：

```bash
conda install ffmpeg==6.1.1

# 或

pip uninstall torchcodec
```

---

## 📨 使用方法

### 1. 数据格式转换

将 Kuavo 原生 rosbag 数据转换为 Lerobot 框架可用的 parquet 格式：

```bash
python kuavo_data/CvtRosbag2Lerobot.py \
  --config-path=../configs/data/ \
  --config-name=KuavoRosbag2Lerobot.yaml \
  rosbag.rosbag_dir=/path/to/rosbag \
  rosbag.lerobot_dir=/path/to/lerobot_data
```

说明：

* `rosbag.rosbag_dir`：原始 rosbag 数据路径
* `rosbag.lerobot_dir`：转换后的lerobot-parquet 数据保存路径，通常会在此目录下创建一个名为lerobot的子文件夹
* `configs/data/KuavoRosbag2Lerobot.yaml`：请查看并根据需要选择启用的相机及是否使用深度图像等

---

### 2. 模仿学习训练

使用转换好的数据进行模仿学习训练：

```bash
python kuavo_train/train_policy.py \
  --config-path=../configs/policy/ \
  --config-name=diffusion_config.yaml \
  task=your_task_name \
  method=your_method_name \
  root=/path/to/lerobot_data/lerobot \
  training.batch_size=128 \
  policy_name=diffusion
```

说明：

* `task`：自定义，任务名称（最好与数转中的task定义对应），如`pick and place`
* `method`：自定义，方法名，用于区分不同的训练，如`diffusion_bs128_usedepth_nofuse`等
* `root`：训练数据的本地路径，注意加上lerobot，与1中的数转保存路径需要对应，为：`/path/to/lerobot_data/lerobot`
* `training.batch_size`：批大小，可根据 GPU 显存调整
* `policy_name`：使用的策略，用于策略实例化的，目前支持`diffusion`和`act`
* 其他参数可详见yaml文件说明，推荐直接修改yaml文件，避免命令行输入错误

---

### 3. 仿真器测试

完成训练后可启动mujoco仿真器并调用部署代码并进行评估：

a. 启动mujoco仿真器：详情请见[readme for simulator](https://github.com/LejuRobotics/kuavo-ros-opensource/blob/opensource/kuavo-data-challenge/readme.md)

b. 调用部署代码

- 配置文件位于 `./configs/deploy/`：
  * `kuavo_sim_env.yaml`：仿真器运行配置
  * `kuavo_real_env.yaml`：真机运行配置


- 请查看yaml文件，并修改下面的`# inference configs`相关的参数（模型加载）等。

- 启动自动化推理部署：
  ```bash
  bash kuavo_deploy/eval_kuavo.sh
  ```
- 按照指引操作，一般最后请选择`"8. 仿真中自动测试模型，执行eval_episodes次:`，这步操作详见[kuavo deploy](kuavo_deploy/readme/inference.md)
---



### 4. 真机测试

步骤同3中a部分，更换指定配置文件为 `kuavo_real_env.yaml`，即可在真机上部署测试。

---

## 📡 ROS 话题说明

**仿真环境：**

| 话题名                                           | 功能说明          |
| --------------------------------------------- | ------------- |
| `/cam_h/color/image_raw/compressed`           | 上方相机 RGB 彩色图像 |
| `/cam_h/depth/image_raw/compressedDepth`      | 上方相机深度图       |
| `/cam_l/color/image_raw/compressed`           | 左侧相机 RGB 彩色图像 |
| `/cam_l/depth/image_rect_raw/compressedDepth` | 左侧相机深度图       |
| `/cam_r/color/image_raw/compressed`           | 右侧相机 RGB 彩色图像 |
| `/cam_r/depth/image_rect_raw/compressedDepth` | 右侧相机深度图       |
| `/gripper/command`                            | 仿真rq2f85夹爪控制命令    |
| `/gripper/state`                              | 仿真rq2f85夹爪当前状态   |
| `/joint_cmd`                                  | 所有关节的控制指令，包含腿部  |
| `/kuavo_arm_traj`                             | 机器人机械臂轨迹控制 |
| `/sensors_data_raw`                           | 所有传感器原始数据 |

**真机环境：**

| 话题名                                           | 功能说明          |
| --------------------------------------------- | ------------- |
| `/cam_h/color/image_raw/compressed`           | 上方相机 RGB 彩色图像 |
| `/cam_h/depth/image_raw/compressedDepth`      | 上方相机深度图，realsense  |
| `/cam_l/color/image_raw/compressed`           | 左侧相机 RGB 彩色图像 |
| `/cam_l/depth/image_rect_raw/compressedDepth` | 左侧相机深度图，realsense       |
| `/cam_r/color/image_raw/compressed`           | 右侧相机 RGB 彩色图像 |
| `/cam_r/depth/image_rect_raw/compressedDepth` | 右侧相机深度图，realsense       |
| `/control_robot_hand_position`                | 灵巧手关节角控制指令      |
| `/dexhand/state`                              | 灵巧手当前关节角状态        |
| `/leju_claw_state`                            | 乐聚夹爪当前关节角状态     |
| `/leju_claw_command`                          | 乐聚夹爪关节角控制指令     |
| `/joint_cmd`                                  | 所有关节的控制指令，包含腿部    |
| `/kuavo_arm_traj`                             | 机器人机械臂轨迹控制       |
| `/sensors_data_raw`                           | 所有传感器原始数据 |



---

## 📁 代码输出结构

```
outputs/
├── train/<task>/<method>/run_<timestamp>/   # 训练模型与参数
├── eval/<task>/<method>/run_<timestamp>/    # 测试日志与视频
```

---

## 📂 核心代码结构

```
KUAVO-DATA-CHALLENGE/
├── configs/                # 配置文件
├── kuavo_data/             # 数据处理转换模块
├── kuavo_deploy/           # 部署脚本（模拟器/真机）
├── kuavo_train/            # 模仿学习训练代码
├── lerobot_patches/        # Lerobot 运行补丁
├── outputs/                # 模型与结果
├── third_party/            # Lerobot 依赖
└── requirements_xxx.txt    # 依赖列表
└── README.md               # 说明文档
```

---

## 🐒 关于 `lerobot_patches`

该目录包含对 **Lerobot** 的兼容性补丁，主要功能包括：

* 扩展 `FeatureType`，支持 RGB 与 Depth 图像
* 定制 `compute_episode_stats` 与 `create_stats_buffers`，便于图像与深度数据统计
* 修改 `dataset_to_policy_features`，确保 Kuavo RGB+深度数据正确映射

使用时需在入口脚本开头引入：

```python
import lerobot_patches.custom_patches  # Ensure custom patches are applied, DON'T REMOVE THIS LINE!
```

---

## 🙏 致谢

本项目基于 [**Lerobot**](https://github.com/huggingface/lerobot) 扩展而成。
感谢 HuggingFace 团队开发的开源机器人学习框架，为本项目提供了重要基础。


