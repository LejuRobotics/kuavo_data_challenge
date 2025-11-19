
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
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update
export NVIDIA_CONTAINER_TOOLKIT_VERSION=1.17.8-1
sudo apt-get install -y nvidia-container-toolkit=${NVIDIA_CONTAINER_TOOLKIT_VERSION} nvidia-container-toolkit-base=${NVIDIA_CONTAINER_TOOLKIT_VERSION} libnvidia-container-tools=${NVIDIA_CONTAINER_TOOLKIT_VERSION} libnvidia-container1=${NVIDIA_CONTAINER_TOOLKIT_VERSION}
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
<summary>a. 系统直接安装 ROS Noetic（<b>推荐</b>）（展开查看），仅供参考</summary>

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
<summary>b. 使用 Docker 安装 ROS Noetic（展开查看），仅供参考</summary>

- 首先最好是换个源：

```bash
sudo vim /etc/docker/daemon.json
```

- 然后在这个json文件中写入一些镜像源：

```json
{
    "registry-mirrors": [
        "https://docker.m.daocloud.io",
        "https://docker.imgdb.de",
        "https://docker-0.unsee.tech",
        "https://docker.hlmirror.com",
        "https://docker.1ms.run",
        "https://func.ink",
        "https://lispy.org",
        "https://docker.xiaogenban1993.com"
    ]
}
```

- 然后保存文件并退出后，重启docker服务：

```shell
sudo systemctl daemon-reload && sudo systemctl restart docker
```

- 现在开始创建镜像，首先建立Dockerfile：
```shell
mkdir /path/to/save/docker/ros/image
cd /path/to/save/docker/ros/image
vim Dockerfile
```
然后在Dockerfile文件中写入如下内容：

```Dockerfile
FROM ubuntu:20.04

ENV DEBIAN_FRONTEND=noninteractive

RUN sed -i 's|http://archive.ubuntu.com/ubuntu/|http://mirrors.tuna.tsinghua.edu.cn/ubuntu/|g' /etc/apt/sources.list && \
    sed -i 's|http://security.ubuntu.com/ubuntu/|http://mirrors.tuna.tsinghua.edu.cn/ubuntu/|g' /etc/apt/sources.list

RUN apt-get update && apt-get install -y locales tzdata gnupg lsb-release
RUN locale-gen en_US.UTF-8
ENV LANG=en_US.UTF-8 LANGUAGE=en_US:en LC_ALL=en_US.UTF-8

# 设置ROS的debian源
RUN sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'

# 添加ROS的Keys
RUN apt-key adv --keyserver 'hkp://keyserver.ubuntu.com:80' --recv-key C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654

# 安装ROS Noetic
# 设置键盘布局为 Chinese
RUN apt-get update && \
    apt-get install -y keyboard-configuration apt-utils && \
    echo 'keyboard-configuration keyboard-configuration/layoutcode string cn' | debconf-set-selections && \
    echo 'keyboard-configuration keyboard-configuration/modelcode string pc105' | debconf-set-selections && \
    echo 'keyboard-configuration keyboard-configuration/variant string ' | debconf-set-selections && \
    apt-get install -y ros-noetic-desktop-full && \
    apt-get install -y python3-rosdep python3-rosinstall python3-rosinstall-generator python3-wstool build-essential && \
    rm -rf /var/lib/apt/lists/*

# 初始化rosdep
RUN rosdep init
```
写入完毕后保存退出。

- 执行ubuntu20.04 + ROS Noetic镜像的构建：

```shell
sudo docker build -t ubt2004_ros_noetic .
```

- 构建完成后进入镜像即可，初次启动容器加载镜像：

```shell
sudo docker run -it --name ubuntu_ros_container ubt2004_ros_noetic /bin/bash
# 或 GPU 启动（推荐）
sudo docker run -it --gpus all --runtime nvidia --name ubuntu_ros_container ubt2004_ros_noetic /bin/bash
# 可选，挂载本地目录路径等
# sudo docker run -it --gpus all --runtime nvidia --name ubuntu_ros_container -v /path/to/your/code:/root/code ubt2004_ros_noetic /bin/bash
```

之后每次加载：
```shell
sudo docker start ubuntu_ros_container
sudo docker exec -it ubuntu_ros_container /bin/bash
```

- 或者自定义启动加载文件，launch_docker.sh
```shell
#!/bin/bash

# Paths
CODE_DIR=/path/to/code
PYTHON_DIR=/path/to/python_env
DATA_DIR=/path/to/data
IMAGE=ros:noetic
CONTAINER=ros_noetic

# Create container if it doesn't exist
if ! docker ps | grep -q "$CONTAINER"; then
    echo "🛠  Creating container $CONTAINER ..."
    docker create --name=$CONTAINER $IMAGE
fi

# Run container with mounts and environment
echo "🚀 Starting container $CONTAINER ..."
docker run \
    -i -t \
    -v $CODE_DIR:/code \
    -v $DATA_DIR:/data \
    -v $PYTHON_DIR:$PYTHON_DIR \
    --env PATH=/path/to/python_venv/kdc/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
    $CONTAINER /bin/bash
```


- 进入镜像后，初始化ros环境变量，然后启动roscore

```shell
source /opt/ros/noetic/setup.bash
roscore
```

无误的话，ubuntu20.04 + ros noetic的docker配置方式就结束了。

</details>

<br>
⚠️ 警告：如果上述中ROS使用的是docker环境，下方后续的代码可能需要在容器里面运行，如有问题，请核对当前是否在容器内！

---

### 3. 克隆代码

```bash
# SSH
git clone git@github.com:LejuRobotics/kuavo_data_challenge.git
# 或者
# HTTPS
git clone https://github.com/LejuRobotics/kuavo_data_challenge.git
```

更新third_party下的lerobot子模块：

```bash
cd kuavo-data-challenge
git submodule init
git submodule update --recursive --progress
```

---

### 4. Python 环境配置

使用 conda （推荐）或 python venv 创建虚拟环境（推荐 python 3.10）：

- ananconda配置：

```bash
conda create -n kdc python=3.10
conda activate kdc
```

- 或，源码安装Python3.10.18，再用venv创建虚拟环境

注意：```ppa:deadsnakes``` 在2025年6月后不能在ubuntu20.04上提供了

```bash
sudo apt update
sudo apt install -y software-properties-common
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update
sudo apt install -y python3.10 python3.10-venv python3.10-dev
```
可以尝试下，不行请使用源码安装：
```bash
sudo apt update
sudo apt install -y build-essential libssl-dev zlib1g-dev libncurses5-dev libncursesw5-devlibreadline-dev libsqlite3-dev libgdbm-dev libdb5.3-dev libbz2-dev libexpat1-dev liblzma-dev tk-dev libffi-dev uuid-dev wget

wget https://www.python.org/ftp/python/3.10.18/Python-3.10.18.tgz
tar -xzf Python-3.10.18.tgz
cd Python-3.10.18
./configure --prefix=$HOME/python3.10 --enable-optimizations
make -j$(nproc)
sudo make install

python3.10 -m venv kdc
source kdc/bin/activate
```

- 查看和确保安装正确：
```shell
python  # 查看python版本，看到确认输出为3.10.xxx（通常是3.10.18）
# 输出示例：
# Python 3.10.18 (main, Jun  5 2025, 13:14:17) [GCC 11.2.0] on linux
# Type "help", "copyright", "credits" or "license" for more information.
# >>> 

pip --version # 查看pip对应的版本，看到确认输出为3.10的pip
# 输出示例：pip 25.1 from /path/to/your/env/python3.10/site-packages/pip (python 3.10)
```


安装依赖：

```bash
source /opt/ros/noetic/setup.bash  # 进入python环境先source好ros自带的python库，建议这行写入~/.bashrc

pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple  # 建议首先换源，能加快下载安装速度

pip install -r requirements_ilcode.txt   # 无需ROS Noetic，但只能使用kuavo_train模仿学习训练代码，kuavo_data（数转）及 kuavo_deploy（部署代码）均依赖ROS
# 或
pip install -r requirements_total.txt    # 需确保 ROS Noetic 已安装 (推荐)
```


如果pip安装完毕但运行训练代码时报ffmpeg或torchcodec的错：

```bash
conda install ffmpeg==6.1.1

# 或

# pip uninstall torchcodec
```

如果想使用torchcodec，又没有conda，环境是用python venv创建的：
- 源码构建：参考[ffmpeg官方库](https://trac.ffmpeg.org/wiki/CompilationGuide/Ubuntu#GettheDependencies)

&nbsp;&nbsp;&nbsp;&nbsp; (a). 提前把osm那些包装好，仿照官方文档

&nbsp;&nbsp;&nbsp;&nbsp; (b). openh264:
```bash
cd ~/python-pkg/ffmpeg_source
git clone https://github.com/cisco/openh264.git
cd openh264
git checkout v2.4.1   # 对应 FFmpeg 官方支持版本
make -j$(nproc)
sudo make install PREFIX=$HOME/ffmpeg_build
```

&nbsp;&nbsp;&nbsp;&nbsp; (c). 编译安装ffmpeg，这种安装和conda安装一模一样的功能，验证不会有问题
```bash
PATH="$HOME/bin:$PATH" PKG_CONFIG_PATH="$HOME/ffmpeg_build/lib/pkgconfig:/usr/lib/x86_64-linux-gnu/pkgconfig" ./configure   --prefix=/usr/local   --disable-doc --extra-cflags=-I$HOME/ffmpeg_build/include --extra-ldflags=-L$HOME/ffmpeg_build/lib  --enable-swresample   --enable-swscale   --enable-openssl   --enable-libxml2   --enable-libtheora   --enable-demuxer=dash   --enable-postproc   --enable-hardcoded-tables   --enable-libfreetype   --enable-libharfbuzz   --enable-libfontconfig   --enable-libdav1d   --enable-zlib   --enable-libaom   --enable-pic   --enable-shared   --disable-static   --disable-gpl   --enable-version3   --disable-sdl2   --enable-libopenh264   --enable-libopus   --enable-libmp3lame   --enable-libopenjpeg   --enable-libvorbis   --enable-pthreads   --enable-libtesseract   --enable-libvpx
sudo make -j$(nproc)
sudo make install
sudo ldconfig
# ffmpeg -version验证
```


- 关于 kuavo_humanoid_sdk：

有时会出现版本不匹配的问题，上述是通过pip install在pypi.org上找包安装的，若出现相关问题，可以手动至kuavo-ros-control或kuavo-ros-opensource源码安装，例如，激活Python环境后：
```bash
cd /your/path/to/kuavo-ros-control/src/kuavo_humanoid_sdk
# 或
# cd /your/path/to/kuavo-ros-opensource/src/kuavo_humanoid_sdk

./install.sh
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

### 2.1 模仿学习训练：单机多卡模式

安装accelerate库： pip install accelerate

```bash
accelerate launch --config_file ./configs/policy/accelerate_config.yaml \ 
  ./kuavo_train/train_policy_with_accelerate.py  --  \ 
  --config-path ./configs/policy \ 
  --config-name diffusion_config.yaml
```

说明：

* diffusion_config.yaml文件中配置参数设置参考上面《2.0 模仿学习训练》详细参数说明 

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
* 定制 `compute_episode_stats` 与 `create_stats_buffers`，用于图像与深度数据的统计量统计，min，max，mean，std等
* 修改 `dataset_to_policy_features`，确保 Kuavo RGB + Depth的FeatureType正确映射

需要使用基于lerobot的定制设计如深度数据、新的FeatureType、归一化方式等，可自行添加，并在使用时在入口脚本（如kuavo_train/train_policy.py等训练文件代码）的最开头一行引入：

```python
import lerobot_patches.custom_patches  # Ensure custom patches are applied, DON'T REMOVE THIS LINE!
```

---

## 🙏 致谢

本项目基于 [**Lerobot**](https://github.com/huggingface/lerobot) 扩展而成。
感谢 HuggingFace 团队开发的开源机器人学习框架，为本项目提供了重要基础。


