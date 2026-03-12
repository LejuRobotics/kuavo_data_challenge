# Kuavo Robot Host Computer Evaluation Guide

## 1. Connect to the Robot's Host Computer

There are two ways to communicate with the host computer:

(a) Use a computer located on the same local network as the robot, and then use SSH
```bash
ssh leju_kuavo@xxx.xxx.xxx.xxx   # Manually verify the HC's IP
# Password: leju_kuavo
```

(b) Hook up a display, keyboard and mouse directly to the host computer, then log-in using the credentials from above.

---

## 2. Create your Workspace and Prepare its Environment

Create your workspace:

```bash
cd ~
mkdir kdc_ws
cd kdc_ws
```

Clone our repository:

```bash
# https
git clone -b dev --depth=1 https://github.com/LejuRobotics/kuavo_data_challenge.git

# ssh
# git clone git@github.com:LejuRobotics/kuavo_data_challenge.git
```

Initialise the branch and all its submodules:

```bash
cd kuavo_data_challenge
git checkout origin/dev
git submodule init
git submodule update --recursive --progress
```

---

## 3. Create your Python Environment and Install Dependencies

Typically, Python 3.10 is preinstalled in the host computer. If not, install Python 3.10 first according to the appendix.

```bash
python3.10 -m venv ~/kdc_ws/kdc_env
source ~/kdc_ws/kdc_env/bin/activate

which pip
pip list

# If ROS dependencies are needed:
# source /opt/ros/noetic/setup.bash

pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
pip install -r requirements_agxorin.txt

# If a newer pip version causes dependency conflicts：
pip install -r requirements_agxorin.txt --use-deprecated=legacy-resolver
```

---

## 4. Copy Over Your Pretrained Weights

Copy the entire directory generated from your training to the following path:

```
~/kdc_ws/kuavo_data_challenge/outputs/train/<task>/<method>/<timestamp>/epoch<epoch>
```

For example:

```bash
mkdir -p outputs/train/your_task_name/your_method/your_timestamp
cp -R <your_epoch_dir> outputs/train/your_task_name/your_method/your_timestamp
```

Its directory structure should be as follows:

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

## 5. Configuration & Deploy

Edit the deployment configuration:

```bash
vim configs/deploy/kuavo_env.yaml
```

You **must** ensure that every entry of the configuration is set correctly. Otherwise, the deployment may fail.
(vim: `ESC` → `:wq!` save and exit; `:q!` abort changes)

Deploy:

```bash
python kuavo_deploy/eval_kuavo.py
```

Enter as prompted:

(a) Enter **3**
(b) Enter `configs/deploy/kuavo_env.yaml`
(c) Enter **2** to play a rosbag first (The robot should start moving about now, please watch for safety)

After the playback:

(d) Press **3** to start the deployment
(e) Press **s** at any time to stop the deployment (recommended). Alternatively, `Ctrl+C` can also kill the deployment.

---

## Appendix:

### Python 3.10 Installation:

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

---
### About kuavo_humanoid_sdk:

⚠️ Sometimes, the versions may become mismatched, which prevents communication and errors with: 机械臂初始化失败 (Arm initialisation failed)! Here's how to fix that:

（a）Enter the lower computer as follows:

```bash
  ssh lab@192.168.26.1 # password is three spaces
  cd ~/kuavo-ros-opensource
  git describe --tag # Check the kuavo-ros-opensource version
  # Displays xxx
```
  - Now, go back to the other computer:
```bash
# Enter your conda environment
conda activate kdc_dev
# or venv
source kdc_dev/bin/activate
pip install kuavo-humanoid-sdk==xxx #Install the corresponding version of sdk
```


（b）(Time-consuming and error prone, not recommended) Copy the kuavo-ros-opensource folder from the lower computer [kuavo-ros-opensource](https://github.com/LejuRobotics/kuavo-ros-opensource), such as:

```bash
scp -r lab@192.168.26.1:~/kuavo-ros-opensource /your/path/
cd /your/path/kuavo-ros-opensource/src/kuavo_humanoid_sdk
# or
# cd /your/path/to/kuavo-ros-opensource/src/kuavo_humanoid_sdk
# Enter your conda environment
conda activate kdc_dev
# or
source kdc_dev/bin/activate

./install.sh
```
