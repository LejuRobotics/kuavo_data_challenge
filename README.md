# 🚀 **Kuavo Data Challenge**
<p align="right">
  <a href="README.md"><b>English</b></a> |
  <a href="README_ZH.md">简体中文</a>
</p>

[![leju](https://img.shields.io/badge/LEJUROBOT-blue)](https://www.lejurobot.com/zh)
[![tong](https://img.shields.io/badge/BIGAI-red)](https://www.bigai.ai/)


## ✨ New features!

### This is a branch under continuous development! It now supports:

- Depth imaging under ACT / Diffusion policy, each with its own fusion method for RGB vs depth imaging. For more details, see [ACT](kuavo_train/wrapper/policy/act/ACTModelWrapper.py) and [Diffusion](kuavo_train/wrapper/policy/diffusion/DiffusionModelWrapper.py)
- Multi-GPU acceleration provided by Accelerate! See [Multi-GPU acceleration](#multigpu) for details.
- Latest Lerobot version 0.4.2 support! [lerobot](https://github.com/huggingface/lerobot)
- [Frame alignment](kuavo_deploy/utils/obs_buffer.py)!
- Complete restructuring of the directories.
- ···

### More to come:
- End-effector delta control support
- Extra Imitation Learning based algorithms!

---
## 🌟 Overview
This repository is developed based on [Lerobot](https://github.com/huggingface/lerobot), combined with Leju Kuavo robot, providing complete example code for **data format conversion** (rosbag → parquet), **Imitation Learning (IL) training**, **simulator testing**, and **real robot deployment verification**.

---

## ✨ Features
- Data format conversion module (rosbag → Lerobot parquet)  
- IL model training framework (diffusion policy, ACT)
- Mujoco simulation support  
- Real robot verification and deployment  

⚠️ Note: This repository does not yet support **end-effector** control; currently only **joint angle control** is available!

---

## ♻️ Requirements
- **System**: **Ubuntu 20.04** recommended (if you are running 22.04 / 24.04 it's suggested to use Docker containers)  
- **Python**: **Python 3.10** recommended  
- **ROS**: **ROS Noetic + Kuavo Robot ROS patches** (it's OK if installed in Docker container)  
- **Dependencies**: **Docker, NVIDIA CUDA Toolkit** (if GPU acceleration is needed)  

---

## 📦 Installation

### 1. OS Configuration
**Ubuntu 20.04 + NVIDIA CUDA Toolkit + Docker** is recommended.  
<details>
<summary>Detailed steps (expand to view), for reference only</summary>

#### a. Install OS and NVIDIA Drivers
```bash
sudo apt update
sudo apt upgrade -y
ubuntu-drivers devices
# Tested verfied version is 535, you can try newer versions (do not use server branch)
sudo apt install nvidia-driver-535
# Reboot the computer
sudo reboot
# Verify driver installation
nvidia-smi
```

#### b. Install NVIDIA Container Toolkit

When using nvidia-smi acceleration in Docker images, it is necessary to load the nvidia runtime library, therefore NVIDIA Container Toolkit needs to be installed.

```bash
sudo apt install curl
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update
export NVIDIA_CONTAINER_TOOLKIT_VERSION=1.17.8-1
export NVIDIA_CONTAINER_TOOLKIT_VERSION=1.17.8-1 && sudo apt-get install -y nvidia-container-toolkit=${NVIDIA_CONTAINER_TOOLKIT_VERSION} nvidia-container-toolkit-base=${NVIDIA_CONTAINER_TOOLKIT_VERSION} libnvidia-container-tools=${NVIDIA_CONTAINER_TOOLKIT_VERSION} libnvidia-container1=${NVIDIA_CONTAINER_TOOLKIT_VERSION}
```


#### c. Install Docker

```bash
sudo apt update
sudo apt install git
sudo apt install docker.io
# Configure NVIDIA Runtime in Docker
nvidia-ctk
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
sudo docker info | grep -i runtime
# The output should include "nvidia" Runtime
```

</details>

---

### 2. ROS Environment Configuration

Both Kuavo Mujoco simulation and real robot operation are based on the **ROS Noetic** environment. Since the real Kuavo robot uses Ubuntu 20.04 + ROS Noetic (non-docker), it is recommended to directly install ROS Noetic. If ROS Noetic cannot be installed due to a higher Ubuntu version, Docker can be used.

<details>
<summary>a. Direct System Installation of ROS Noetic (<b>Recommended</b>)</summary>

* Official Guide: [ROS Noetic Installation](http://wiki.ros.org/noetic/Installation/Ubuntu)
* Recommended Chinese mirror source: [小鱼ROS](https://fishros.org.cn/forum/topic/20/)

Installation example:

```bash
wget http://fishros.com/install -O fishros && . fishros
# Menu selection: 5 Configure system sources → 2 Change sources and clean third-party sources → 1 Add ROS sources
wget http://fishros.com/install -O fishros && . fishros
# Menu selection: 1 One-click installation → 2 Install without changing sources → Select ROS1 Noetic Desktop
```

Test ROS installation:

```bash
roscore  # Open a new terminal
rosrun turtlesim turtlesim_node  # Open a new terminal
rosrun turtlesim turtle_teleop_key  # Open a new terminal
```

</details>

<details>
<summary>b. Install ROS Noetic Using Docker</summary>

- First, it's best to change the mirror source:

```bash
sudo vim /etc/docker/daemon.json
```

- Then write some mirror sources in this json file:

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

- Then save the file and exit, restart the Docker service:

```shell
sudo systemctl daemon-reload && sudo systemctl restart docker
```

- Now start creating the image, first create the Dockerfile:
```shell
mkdir /path/to/save/docker/ros/image
cd /path/to/save/docker/ros/image
vim Dockerfile
```
Then write the following content in the Dockerfile:

```Dockerfile
FROM ubuntu:20.04

ENV DEBIAN_FRONTEND=noninteractive

RUN sed -i 's|http://archive.ubuntu.com/ubuntu/|http://mirrors.tuna.tsinghua.edu.cn/ubuntu/|g' /etc/apt/sources.list && \
    sed -i 's|http://security.ubuntu.com/ubuntu/|http://mirrors.tuna.tsinghua.edu.cn/ubuntu/|g' /etc/apt/sources.list

RUN apt-get update && apt-get install -y locales tzdata gnupg lsb-release
RUN locale-gen en_US.UTF-8
ENV LANG=en_US.UTF-8 LANGUAGE=en_US:en LC_ALL=en_US.UTF-8

# Set ROS debian sources
RUN sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'

# Add ROS keys
RUN apt-key adv --keyserver 'hkp://keyserver.ubuntu.com:80' --recv-key C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654

# Install ROS Noetic
# Set keyboard layout to Chinese if necessary
RUN apt-get update && \
    apt-get install -y keyboard-configuration apt-utils && \
    echo 'keyboard-configuration keyboard-configuration/layoutcode string cn' | debconf-set-selections && \
    echo 'keyboard-configuration keyboard-configuration/modelcode string pc105' | debconf-set-selections && \
    echo 'keyboard-configuration keyboard-configuration/variant string ' | debconf-set-selections && \
    apt-get install -y ros-noetic-desktop-full && \
    apt-get install -y python3-rosdep python3-rosinstall python3-rosinstall-generator python3-wstool build-essential && \
    rm -rf /var/lib/apt/lists/*

# Initialize rosdep
RUN rosdep init
```
After writing, save and exit. Build the Ubuntu 20.04 + ROS Noetic image:

```shell
sudo docker build -t ubt2004_ros_noetic .
```

After the build is complete, enter the image. For the first time starting the container and loading the image:

```shell
sudo docker run -it --name ubuntu_ros_container ubt2004_ros_noetic /bin/bash
# Or GPU launch (recommended)
sudo docker run -it --gpus all --runtime nvidia --name ubuntu_ros_container ubt2004_ros_noetic /bin/bash
# Optional, mount local directory paths, etc.
# sudo docker run -it --gpus all --runtime nvidia --name ubuntu_ros_container -v /path/to/your/code:/root/code ubt2004_ros_noetic /bin/bash
```

For subsequent launches:
```shell
sudo docker start ubuntu_ros_container
sudo docker exec -it ubuntu_ros_container /bin/bash
```

After entering the image, initialize the ROS environment variables, then start roscore:

```shell
source /opt/ros/noetic/setup.bash
roscore
```

If everything is correct, the Docker configuration for Ubuntu 20.04 + ROS Noetic is complete.

</details>

<br>
⚠️ Warning: If ROS is using a Docker environment as mentioned above, the following code may need to be run inside the container. If you encounter issues, please check whether you are currently inside the container!

---

### 3. Clone Code

```bash
# SSH
git clone --depth=1 https://github.com/LejuRobotics/kuavo_data_challenge.git
# Or
# HTTPS
git clone --depth=1 https://github.com/LejuRobotics/kuavo_data_challenge.git
```

Update the lerobot submodule under third_party:

```bash
cd kuavo_data_challenge
git submodule init
git submodule update --recursive
```

---

### 4. Python Environment Configuration

#### Choose one of the following:
- **Use conda** (recommended):

```bash
conda create -n kdc_icra python=3.10
conda activate kdc_icra
```

- **Use venv**: 

Install python3.10 first, then use venv to create a virtual environment:

⚠️ Warning: ```ppa:deadsnakes``` no longer provide packages for ubuntu20.04 after June 2025, the following installation method may not work anymore:

```bash
sudo apt update
sudo apt install -y software-properties-common
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update
sudo apt install -y python3.10 python3.10-venv python3.10-dev
```
You may need to build from source as follows:
```bash
sudo apt update
sudo apt install -y build-essential libssl-dev zlib1g-dev libncurses5-dev libncursesw5-devlibreadline-dev libsqlite3-dev libgdbm-dev libdb5.3-dev libbz2-dev libexpat1-dev liblzma-dev tk-dev libffi-dev uuid-dev wget

wget https://www.python.org/ftp/python/3.10.18/Python-3.10.18.tgz
tar -xzf Python-3.10.18.tgz
cd Python-3.10.18
./configure --prefix=$HOME/python3.10 --enable-optimizations
make -j$(nproc)
sudo make install
```

Now create the venv environment:

```bash
python3.10 -m venv kdc_dev
source kdc_dev/bin/activate
```

Check and ensure correct installation:
```shell
python  # Check Python version, confirm output is 3.10.xxx (usually 3.10.18)
# Example output:
# Python 3.10.18 (main, Jun  5 2025, 13:14:17) [GCC 11.2.0] on linux
# Type "help", "copyright", "credits" or "license" for more information.
# >>> 

pip --version # Check pip version, confirm output shows pip for 3.10
# Example output: pip 25.1 from /path/to/your/env/python3.10/site-packages/pip (python 3.10)
```

#### Install dependencies:

```bash
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple  # If youre located within mainland China, it is recommended to change the source first to speed up download and installation
# There is no need for you to run this if otherwise!

pip install -r requirements_ilcode.txt   # No ROS Noetic required, but only guarantees functionality of kuavo_train imitation learning training code. kuavo_data (data conversion) and kuavo_deploy (deployment code) both depend on ROS
# Or
pip install -r requirements_total.txt    # Ensure ROS Noetic is installed first (recommended)
```

After installation, double-check the lerobot version: Should be Version 0.4.2 as of November 2025.
```bash
pip show | grep lerobot
```

If not, reset the lerobot repository:
```bash
cd third_party/lerobot
git fetch
git reset --hard origin/main
cd ../../
```

Retry pip install -r requirement_xx.txt to retry installation.

If you encounter ffmpeg or torchcodec errors when running:

```bash
conda install ffmpeg==6.1.1

# Or

# pip uninstall torchcodec
```

---

## 📨 Usage

### 1. Data Format Conversion

Convert Kuavo native rosbag data to parquet format usable by the Lerobot framework:

```bash
python kuavo_data/CvtRosbag2Lerobot.py \
  --config-path=../configs/data/ \
  --config-name=KuavoRosbag2Lerobot.yaml \
  rosbag.rosbag_dir=/path/to/rosbag \
  rosbag.lerobot_dir=/path/to/lerobot_data
```

Description:

* `rosbag.rosbag_dir`: Path to original rosbag data
* `rosbag.lerobot_dir`: Path to save converted lerobot-parquet data. A subfolder named lerobot is usually created in this directory
* `configs/data/KuavoRosbag2Lerobot.yaml`: Please review and select cameras to enable and whether to use depth images as needed

Or, you can set args in ```configs/data/KuavoRosbag2Lerobot.yaml```

---

### 2. Imitation Learning Training

Use the converted data for imitation learning training:

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

Description:

* `task`: Custom task name (preferably corresponding to the task definition in data conversion), e.g., `pick and place`
* `method`: Custom method name, used to distinguish different training runs, e.g., `diffusion_bs128_usedepth_nofuse`, etc.
* `root`: Local path to training data. Note to include lerobot. Should correspond to the data conversion save path in step 1: `/path/to/lerobot_data/lerobot`
* `training.batch_size`: Batch size, can be adjusted according to GPU memory
* `policy_name`: Policy to use, used for policy instantiation. Currently supports `diffusion` and `act`
* For other parameters, please refer to the yaml file documentation. It is recommended to directly modify the yaml file to avoid command-line input errors

Or, you can set args in ```configs/policy/act_config/diffusion_config.yaml```

---

<a id="multigpu"></a>
### 2.1 IL Training with Multi-GPU Support

Double-check installation of Accelerate: pip install accelerate (Usually automatically installed with Lerobot)

```bash
# Configure the accelerate yaml according to your machines specs
vim configs/accelerate/accelerate_config.yaml
# After configuration, try this example:
accelerate launch --config_file configs/accelerate/accelerate_config.yaml kuavo_train/train_policy_with_accelerate.py  --config-path=../configs/policy --config-name=diffusion_config.yaml
```

---

### 3. Simulator Testing

After training is complete, you can start the Mujoco simulator and call the deployment code for evaluation:

a. Start Mujoco simulator: For details, see [readme for simulator](https://github.com/LejuRobotics/kuavo-ros-opensource/tree/opensource/kuavo-data-challenge-icra)

b. Call deployment code

- Configuration files are located in `./configs/deploy/`:
  * `kuavo_env.yaml`: Kuavo robot execution environment configuration, with `env_name` as `Kuavo-Sim`. Change other parameters such as `obs_key_map` as needed.


- Please review the yaml file and modify the `# inference configs` related parameters (model loading), etc.

- Start automated inference deployment:
  ```bash
  python kuavo_deploy/eval_kuavo.py
  ```
- Follow the instructions. Generally, Select `3` first, then provide the `kuavo_env.yaml` path (`configs/deploy/kuavo_env.yaml`). Finally, select `"8. Auto-test model in simulation, execute eval_episodes times:"`. For details on this operation, see [kuavo deploy](kuavo_deploy/readme/inference.md)
---



### 4. Real Robot Testing

Same steps as part a in step 3, change the configuration file `kuavo_env.yaml`'s `env_name` as `Kuavo-Real`.

- PC deployment steps to be updated; For Orin deployment, please check: [README_AGX_ORIN.md](README_AGX_ORIN.md)

- The log during testing is located at log/kuavo_deploy/kuavo_deploy.log, please check thoroughly.

---

## 📡 ROS Topic Description

**Simulation Environment:**

| Topic Name                                      | Description              |
| --------------------------------------------- | ----------------------- |
| `/cam_h/color/image_raw/compressed`           | Top camera RGB color image |
| `/cam_h/depth/image_raw/compressedDepth`      | Top camera depth image      |
| `/cam_l/color/image_raw/compressed`           | Left camera RGB color image |
| `/cam_l/depth/image_rect_raw/compressedDepth` | Left camera depth image      |
| `/cam_r/color/image_raw/compressed`           | Right camera RGB color image |
| `/cam_r/depth/image_rect_raw/compressedDepth` | Right camera depth image      |
| `/gripper/command`                            | Simulated rq2f85 gripper control command    |
| `/gripper/state`                              | Simulated rq2f85 gripper current state   |
| `/joint_cmd`                                  | Control commands for all joints, including legs  |
| `/kuavo_arm_traj`                             | Robot arm trajectory control |
| `/sensors_data_raw`                           | Raw data from all sensors |

**Real Robot Environment:**

| Topic Name                                      | Description              |
| --------------------------------------------- | ----------------------- |
| `/cam_h/color/image_raw/compressed`           | Top camera RGB color image |
| `/cam_h/depth/image_raw/compressedDepth`      | Top camera depth image, realsense  |
| `/cam_l/color/image_raw/compressed`           | Left camera RGB color image |
| `/cam_l/depth/image_rect_raw/compressedDepth` | Left camera depth image, realsense       |
| `/cam_r/color/image_raw/compressed`           | Right camera RGB color image |
| `/cam_r/depth/image_rect_raw/compressedDepth` | Right camera depth image, realsense       |
| `/control_robot_hand_position`                | Dexterous hand joint angle control command      |
| `/dexhand/state`                              | Dexterous hand current joint angle state        |
| `/leju_claw_state`                            | Leju claw current joint angle state     |
| `/leju_claw_command`                          | Leju claw joint angle control command     |
| `/joint_cmd`                                  | Control commands for all joints, including legs    |
| `/kuavo_arm_traj`                             | Robot arm trajectory control       |
| `/sensors_data_raw`                           | Raw data from all sensors |



---

## 📁 Output Structure

```
outputs/
├── train/<task>/<method>/run_<timestamp>/   # Training models and parameters
├── eval/<task>/<method>/run_<timestamp>/    # Test logs and videos
```

---

## 📂 Code Structure

```
KUAVO_DATA_CHALLENGE/
├── configs/                # Configuration files
├── kuavo_data/             # Data processing and conversion module
├── kuavo_deploy/           # Deployment scripts (simulator/real robot)
├── kuavo_train/            # Imitation learning training code
├── lerobot_patches/        # Lerobot runtime patches
├── outputs/                # Models and results
├── third_party/            # Lerobot dependencies
└── requirements_xxx.txt    # Dependency lists
└── README.md               # Documentation
```

---

## 🐒 About `lerobot_patches`

This directory contains compatibility patches for **Lerobot**, with main features including:

* Extend `FeatureType` to support RGB and Depth images
* Customize `compute_episode_stats` and `create_stats_buffers` for statistical calculations of images and depth data, min, max, mean, std, etc.
* Modify `dataset_to_policy_features` to ensure correct mapping of Kuavo RGB + Depth FeatureType

If you need to use Lerobot-based custom designs such as depth data, new FeatureTypes, normalization methods, etc., you can add them yourself. When using, import at the very beginning of the entry script (such as kuavo_train/train_policy.py and other training file code):

```python
import lerobot_patches.custom_patches  # Ensure custom patches are applied, DON'T REMOVE THIS LINE!
```

---

## 🙏 Acknowledgments

This project is extended based on [**Lerobot**](https://github.com/huggingface/lerobot).
Thanks to the HuggingFace team for developing the open-source robot learning framework, which provides an important foundation for this project.


