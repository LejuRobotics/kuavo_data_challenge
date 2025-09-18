# 项目 Docker 打包指南

本指南介绍如何构建包含 **ROS Noetic + Miniforge + 项目 + editable 第三方包** 的 Docker 镜像。

---

## 1️⃣ 设置 Docker 镜像加速（可选）

国内访问 Docker Hub 速度慢，可以使用阿里云镜像加速器。

1. 编辑 Docker 配置文件：

```bash
sudo vim /etc/docker/daemon.json
```

2. 替换为以下内容：

```json
{
    "default-runtime": "nvidia",
    "runtimes": {
        "nvidia": {
            "path": "nvidia-container-runtime",
            "args": []
        }
    },
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

3. 保存后重启 Docker：

```bash
sudo systemctl daemon-reload
sudo systemctl restart docker
sudo systemctl status docker
```

---

## 2️⃣ 使用 Conda Pack 打包环境

1. 假设你已有 Conda 环境 `myenv`：

```bash
conda activate myenv
```

2. 打包环境：

```bash
conda pack -n myenv -o myenv.tar.gz
```

⚠️ 注意：
- 如果环境中有 editable 安装包（`pip install -e`），可以先忽略，稍后在 Dockerfile 中再安装。
- 示例：

```bash
conda pack -n myenv --ignore-missing-files -o myenv.tar.gz
```

3. 将打包好的环境压缩包myenv.tar.gz放在项目根目录下

---

## 3️⃣ Dockerfile 构建项目镜像

### 下面提供一个 **Dockerfile的可用示例**：

[Dockerfile可用示例](../Dockerfile)

### 该Dockerfile的主要功能如下所示：

#### 1. 基础镜像
- 使用官方 ROS Noetic Ubuntu 20.04 镜像 `ros:noetic-ros-base-focal`。

#### 2. 国内加速
- APT：使用清华源加速 Ubuntu 软件包下载。
- Conda：配置 Conda channels 为清华镜像。
- Pip：配置 PyPI 国内镜像。

#### 3. 系统工具和 ROS 包
- 安装常用工具：`curl`, `wget`, `sudo`, `build-essential`, `bzip2` 等。
- 安装 ROS 包：`ros-noetic-ros-base`, `ros-noetic-cv-bridge`, `ros-noetic-apriltag-ros`（如果你需要别的ros包，可以自行添加）。

#### 4. Miniforge
- 安装 Miniforge3 并配置环境变量。

#### 5. 项目和 Conda 环境
- 设置工作目录 `/root/kuavo_data_challenge`。
- 复制项目代码。
- 解压 Conda Pack 打包的环境 `myenv.tar.gz`。
- 使用 `conda-unpack` 修复路径。
- 安装项目和第三方包（editable 模式）。
- 删除测试目录和冗余缓存，减小镜像体积。

#### 6. 容器环境优化
- 自动激活 Conda 环境（写入 `.bashrc`）。
- 设置默认命令为 `bash`。
- 多阶段构建：只将最终环境和源码 COPY 到 runtime，避免 builder 中的临时文件占用空间，实现镜像压缩。

---

## 4️⃣ 构建 Docker 镜像并导出为tar文件

将Dockerfile放置于项目根目录下，运行指令：
```bash
docker build -t kdc_v0 .
```

导出：
```bash
docker save -o kdc_v0.tar kdc_v0:latest
```

需将kdc_v0替换成你的镜像名称

---

## 5️⃣ 运行 Docker 容器

### 下面提供一个运行 Docker 容器的 **shell 脚本可用示例**：

[shell脚本可用示例](run_with_gpu.sh)

### 该脚本用于启动或创建 Docker 容器：

- **导入镜像**：
- **检查容器是否存在**：
  - 存在 → 启动并附加 (`docker start -ai`)
  - 不存在 → 创建新容器并启动 (`docker run -it --gpus all --net=host ...`)
- **设置环境变量**：
  - ROS 网络配置 (`ROS_MASTER_URI`、`ROS_IP`)
- **支持 GPU 容器**  

---

## 6️⃣ 注意事项

比赛测试需上传压缩包，压缩包内包含两个文件，一个是kdc_v0.tar(名字可以自行更改)，是docker镜像的压缩包，另一个是运行脚本run_with_gpu.sh(此名不要更改)

必须确保打包好的docker镜像通过以下代码即可运行仿真测试：

```bash
# 启动docker
sh run_with_gpu.sh

# 启动仿真自动化测试
python kuavo_deploy/examples/scripts/script_auto_test.py --task auto_test --config configs/deploy/kuavo_sim_env.yaml

```
