# 环境配置方案

## 安装ROS Noetic（ROS 1），不是ROS 2！
<!-- https://blog.csdn.net/m0_73745340/article/details/135281023 -->
### 配置安装源
- 打开终端，键入：
```bash
wget http://fishros.com/install -O fishros && . fishros
```
- 接着会出现一个菜单界面；选择5 一键配置系统源，测试一下哪个系统源最可靠，选择哪个

- 测试完后，输入2 来更换系统源并清理第三方源

- 随后选择1 添加ROS/ROS2源

### 鱼香ROS一键安装
- 再次跑跟以上相同的命令再次打开鱼香ROS的菜单：
```bash
wget http://fishros.com/install -O fishros && . fishros
```
- 这次选择1 一键安装，随后2 不更换源安装
- 过一会儿后会问要安装的ROS版本，这里选**ROS1 Noetic**，别搞错！随后选桌面版
随后自动安装过程比较长，注意观察一下有没有报错，以及打开系统资源监视器看看有没有持续的网络、CPU、或其它资源用量。如果长时间终端和系统资源无动静可能需要Ctrl+C取消然后重来
- 如果运行完成没有报错，ROS Noetic安装成功

### 测试ROS安装
安装完成后可以用ROS自带的Turtlesim测试一下ROS是否可正确运行
- 开Turtlesim之前先启动ros核心：
```bash
roscore
```
- 再打开两个终端，分别运行：
```bash
rosrun turtlesim turtlesim_node
rosrun turtlesim turtle_teleop_key
```
这个时候有个窗口里面有个小乌龟，另一个窗口可以通过键盘控制该乌龟，这个时候安装就没有问题了！

---
- 把Miniconda的安装脚本下载下来，暂时下载到项目目录里面：
```bash  
#From: https://web.archive.org/web/20231129185127/https://mediawiki.middlebury.edu/CS/Useful_Tools
wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
```

- 预先下载好的先安装Miniconda应该就在这里，现在执行安装：
```bash  
bash Miniconda3-latest-Linux-x86_64.sh
```
- 首先会有个使用条款，按回车打开，继续按使它翻页到最底部，打yes和回车
- 下面会显示默认目录，默认目录就可以
- 后面如果再有个Python Path的警告，再次打yes和回车
- 在以下ab两个方案，创建python环境：
## a. 重新创建 Conda 环境

* **Create a conda environment (Python 3.10 recommended)**
    ```bash
    # Set "kdc" to your conda env name
    conda create -n kdc python=3.10 #Keep on typing 'a' and enter to accept ToS
    conda activate kdc
    ```

* **安装用于通信的kuavo_humanoid_sdk包，参考链接：https://gitee.com/leju-robot/kuavo-ros-opensource/tree/master/src/kuavo_humanoid_sdk**


* **For full system (data transformation, simulator, deployment on real robot, etc.):**

  ```bash
  pip install -r requirements_total.txt
  ```

* **For imitation learning training only:**

  ```bash
  pip install -r requirements_ilcode.txt
  ```

## b. 使用打包好的环境
```bash
# 还需要kdc_v0.tar.gz的下载链接
# 或者conda unpack打包好的环境，注意将setup_env.sh脚本和环境压缩包文件kdc_v0.tar.gz放在同一目录下
./setup_env.sh
source ./kdc_v0/bin/activate
```
## 继续安装依赖
```bash
# Install dependencies. This will take a while...
进入 third_party/lerobot 目录
pip install -e ".[aloha, pusht]"

# Uninstall torchcodec 
pip uninstall torchcodec

进入 kuavo-data-challenge 项目的根目录
pip install -e .

# 安装用于通信的kuavo_humanoid_sdk包，参考链接：
# https://gitee.com/leju-robot/kuavo-ros-opensource/tree/master/src/kuavo_humanoid_sdk

```