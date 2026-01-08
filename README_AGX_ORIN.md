# 夸父机器人上位机推理指南

## 1. 连接机器人上位机

可以选择两种方式连接上位机：

(a) 使用与机器人处于同一局域网的电脑，通过 SSH 连接  
```bash
ssh leju_kuavo@xxx.xxx.xxx.xxx   # 自行确认上位机 IP
# 密码：leju_kuavo
````

(b) 使用键鼠与显示器直接连接机器人上位机（后续步骤相同）

---

## 2. 创建工作目录并准备环境

创建工作目录：

```bash
cd ~
mkdir kdc_ws
cd kdc_ws
```

克隆代码仓库：

```bash
# 使用 https
git clone https://github.com/LejuRobotics/kuavo_data_challenge.git

# 或使用 ssh
# git clone git@github.com:LejuRobotics/kuavo_data_challenge.git
```

初始化分支与子模块：

```bash
cd kuavo_data_challenge
git checkout origin/dev
git submodule init
git submodule update --recursive --progress

# 如果这一步骤由于网络原因下载失败或很慢：请
# cd third_party
# git clone https://githubproxy.cc/https://github.com/huggingface/lerobot.git
# cd ../ # 回到上一级目录
```

---

## 3. 创建 Python 环境并安装依赖

通常上位机已预装 Python 3.10，如未安装请按照附录参考先完成python3.10的安装。

```bash
python3.10 -m venv ~/kdc_ws/kdc_env
source ~/kdc_ws/kdc_env/bin/activate

which pip
pip list

# 若需要 ROS 依赖
source /opt/ros/noetic/setup.bash

pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
pip install -r requirements_agxorin.txt
# 如果遇到ROS相关的库的问题：
pip install -r requirements_agxorin_old.bk

# 如果因pip版本问题导致依赖冲突：
pip install -r requirements_agxorin.txt --use-deprecated=legacy-resolver
```

---

## 4. 放置训练好的权重

将训练产出的完整目录复制到如下路径：

```
~/kdc_ws/kuavo_data_challenge/outputs/train/<task>/<method>/<timestamp>/epoch<epoch>
```

示例：

```bash
mkdir -p outputs/train/your_task_name/your_method/your_timestamp
cp -R <your_epoch_dir> outputs/train/your_task_name/your_method/your_timestamp
```

目录结构应如下：

```
outputs
 └── train
     └── your_task_name
         └── your_method
             └── timestamp
                 ├── epochxxx
                 │   ├── config.json
                 │   └── model.safetensors
                 ├── policy_postprocessor_step_0_unnormalizer_processor.safetensors
                 ├── policy_postprocessor.json
                 ├── policy_preprocessor_step_3_normalizer_processor.safetensors
                 └── policy_preprocessor.json
```

---

## 5. 配置与运行推理

编辑部署配置：

```bash
vim configs/deploy/kuavo_env.yaml
```

请务必**逐项确认配置正确**，否则可能无法正常推理。
（vim：`ESC` → `:wq!` 保存退出；`:q!` 放弃修改）

开始推理：

```bash
python kuavo_deploy/eval_kuavo.py
```

根据提示依次输入：

(a) 输入 **3**
(b) 输入 `configs/deploy/kuavo_env.yaml`
(c) 输入 **2** 回放 rosbag（机器人将开始动作，请注意安全）

回放结束后：

(d) 输入 **3** 开始推理
(e) 推理过程中可随时按 **s** 停止（推荐），或 `Ctrl+C` 结束

---

## 附录：

### python3.10安装：

⚠️ 注意：```ppa:deadsnakes``` 在2025年6月后不能在ubuntu20.04上提供了，下述安装方式不一定成功：

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
```

---
### 关于 kuavo_humanoid_sdk：

⚠️ 有时会出现版本不匹配的问题，无法通信什么的，会报错：机械臂初始化失败！解决方案，若出现相关问题：

（a）进入机器人下位机，

```bash
  ssh lab@192.168.26.1 # 密码三个空格
  cd ~/kuavo-ros-opensource
  git describe --tag # 查看opensource版本
  # 显示xxx
```
  - 返回边侧机，或上位机，
```bash
# 进入环境
conda activate kdc_dev
# 或
source kdc_dev/bin/activate
pip install kuavo-humanoid-sdk==xxx #安装对应版本的sdk
```


（b）（时间较久，较复杂，不推荐）可以拷贝机器人下位机的kuavo-ros-opensource的内容安装，[kuavo-ros-opensource](https://github.com/LejuRobotics/kuavo-ros-opensource)，例如，

```bash
scp -r lab@192.168.26.1:~/kuavo-ros-opensource /your/path/
cd /your/path/kuavo-ros-opensource/src/kuavo_humanoid_sdk
# 或
# cd /your/path/to/kuavo-ros-opensource/src/kuavo_humanoid_sdk
# 进入环境
conda activate kdc_dev
# 或
source kdc_dev/bin/activate

./install.sh
```
