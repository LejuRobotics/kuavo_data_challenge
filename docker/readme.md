# Docker Packaging Guide for Your Project

This guide explains how to build a Docker image that includes ROS Noetic + Miniforge + your project code + editable third-party packages.

---

## 1️⃣ Configure Docker Registry Mirrors (Optional)

Accessing Docker Hub from within China can be slow. You may use registry mirrors such as those provided by DaoCloud or other public accelerators.

1. Edit the Docker configuration file:

```bash
sudo vim /etc/docker/daemon.json
```

2. Replace its contents with the following:

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

3. Save the file and restart Docker:

```bash
sudo systemctl daemon-reload
sudo systemctl restart docker
sudo systemctl status docker
```

---

## 2️⃣ Package Your Conda Environment Using conda-pack

1. Install  conda-pack

```bash
conda install -c conda-forge conda-pack
```

2. Assume you already have a Conda environment named `kdc`：

```bash
conda activate kdc
```

3. Pack the environment:

```bash
conda pack -n kdc -o myenv.tar.gz
```

⚠️ Important Notes:
- If your environment contains packages installed in editable mode (`pip install -e`), you can choose to ignore them during packing and reinstall them later inside the Dockerfile.
- Example:

```bash
conda pack -n kdc --ignore-editable-packages -o myenv.tar.gz
```

4. Place the resulting myenv.tar.gz in your project root directory.

---

## 3️⃣ Build the Project Image with Dockerfile

⚠️ Important:
- Ensure your outputs/ folder contains only one set of model files and their corresponding config files intended for submission. Including unnecessary files will significantly increase the Docker image size.

### Below is a working **Dockerfile **：

[Dockerfile example](../Dockerfile)

### Key Features of This Dockerfile:

#### 1. Base Image
- Uses the official ROS Noetic image on Ubuntu 20.04: `ros:noetic-ros-base-focal`.

#### 2. Domestic Acceleration
- APT: Configured to use Tsinghua University mirrors.
- Conda: Channels set to Tsinghua mirrors.
- Pip: Uses Alibaba Cloud PyPI mirror.

#### 3. System Tools & ROS Packages
- Installs common utilities: `curl`, `wget`, `sudo`, `build-essential`, `bzip2`, etc。
- Installs ROS packages: `ros-noetic-ros-base`, `ros-noetic-cv-bridge`, `ros-noetic-apriltag-ros`(You may add other ROS dependencies as needed).

#### 4. Miniforge
- Installs Miniforge3 and sets up environment variables.

#### 5. Project Code & Conda Environment
- Sets working directory to `/root/kuavo_data_challenge`.
- Copies entire project source code.
- Extracts the packed Conda environment `myenv.tar.gz`.
- Runs `conda-unpack` to fix hardcoded paths.
- Reinstalls the project and third-party packages in editable mode.
- Cleans up test directories and cache to reduce final image size.

#### 6. Container Optimization
- Automatically activates the Conda environment by appending to `.bashrc`.
- Sets default command to `bash`.
- Uses multi-stage build: only the final runtime environment and source code are copied into the final image, excluding builder-stage temporary files—significantly reducing image size.

---

## 4️⃣ Build and Export the Docker Image as a TAR File

Place the Dockerfile in your project root, then run:
```bash
docker build -t kdc_v0 .
```

Export the image:
```bash
docker save -o kdc_v0.tar kdc_v0:latest
```

Replace kdc_v0 with your actual image name if different.

---

## 5️⃣ Run the Docker Container

### Below is a **sample shell script** for running the container:

[shell script example](run_with_gpu.sh)

### This script is used to start or create a Docker container:

- **Import Image**：
- **Check if the container exists**：
  - Exists → Start and Attach (`docker start -ai`)
  - Does not exist → Create a new container and start it (`docker run -it --gpus all --net=host ...`)
- **Set environment variable**：
  - ROS Network Configuration (`ROS_MASTER_URI`、`ROS_IP`)
- **Supports GPU containers**  

---

## 6️⃣ Precautions

For the competition test, you need to upload a compressed file, which should contain two files: one is kdc_v0.tar (the name can be changed), which is the compressed Docker image, and the other is the execution script run_with_gpu.sh (do not change this name).

You must ensure that the packaged Docker image can run the simulation test with the following code:

```bash
# Start docker
sh run_with_gpu.sh

# Start simulation automated testing
python kuavo_deploy/src/scripts/script_auto_test.py --task auto_test --config configs/deploy/kuavo_env.yaml

```
