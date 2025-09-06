# 使用ROS Noetic镜像
FROM ros:noetic-ros-base-focal

# 设置环境变量，避免交互式安装
ARG DEBIAN_FRONTEND=noninteractive

# 设置国内APT镜像源（清华源）
RUN sed -i 's/archive.ubuntu.com/mirrors.tuna.tsinghua.edu.cn/g' /etc/apt/sources.list && \
    sed -i 's/security.ubuntu.com/mirrors.tuna.tsinghua.edu.cn/g' /etc/apt/sources.list

# 安装必要的工具
RUN apt-get update && apt-get install -y \
    curl \
    wget \
    gnupg2 \
    lsb-release \
    sudo \
    ca-certificates \
    build-essential \
    bzip2 \
    # ROS 工具和依赖
    ros-noetic-ros-base \
    ros-noetic-cv-bridge \
    ros-noetic-apriltag-ros \
    && rm -rf /var/lib/apt/lists/*

# 设置miniforge国内镜像（清华源）
ENV MINIFORGE_URL="https://mirrors.tuna.tsinghua.edu.cn/github-release/conda-forge/miniforge/Release%2025.3.1-0/Miniforge3-25.3.1-0-Linux-x86_64.sh"

# 安装 Miniforge3（使用国内镜像）
RUN curl -L ${MINIFORGE_URL} -o /tmp/miniforge.sh \
    && chmod +x /tmp/miniforge.sh \
    && bash /tmp/miniforge.sh -b -p /opt/conda \
    && rm /tmp/miniforge.sh

# 设置环境变量
ENV PATH="/opt/conda/bin:${PATH}"
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

# 配置conda国内镜像
RUN conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main \
    && conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/r \
    && conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/msys2 \
    && conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge/ \
    && conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/bioconda/ \
    && conda config --set show_channel_urls yes

# 设置pip国内镜像
RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple \
    && pip config set global.trusted-host pypi.tuna.tsinghua.edu.cn

# 安装mamba（使用conda-forge国内镜像）
RUN conda install -y mamba -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge/

# 或者使用micromamba（备选方案）
# RUN curl -L https://mirrors.tuna.tsinghua.edu.cn/github-release/mamba-org/micromamba/LatestRelease/micromamba-linux-64 -o /opt/conda/bin/micromamba \
#     && chmod +x /opt/conda/bin/micromamba

# 验证安装
RUN conda --version && mamba --version && python --version

# 设置工作目录
WORKDIR /root/kuavo-data-challenge

# 复制项目代码
COPY . .

# 解压 Conda 环境
RUN if [ -f "myenv.tar.gz" ]; then mkdir -p ./myenv; tar -xzf myenv.tar.gz -C ./myenv; fi

# 激活虚拟环境并运行 conda-unpack
# 激活虚拟环境并安装项目依赖
# 激活虚拟环境并安装第三方包
RUN /bin/bash -c "\
    source ./myenv/bin/activate && \
    conda-unpack && \
    pip install -e . && \
    cd ./third_party/lerobot && pip install -e . \
"

# 如果使用conda环境文件，可以用mamba安装（更快）
# COPY environment.yml .
# RUN mamba env update -f environment.yml

# 添加环境变量
RUN echo "source /root/kuavo-data-challenge/myenv/bin/activate" >> /root/.bashrc

# 设置默认命令
CMD ["bash"]
