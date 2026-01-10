# =========================
# Stage 1: Builder
# =========================
FROM ros:noetic-ros-core-focal AS builder

ARG DEBIAN_FRONTEND=noninteractive

# 国内APT源
RUN sed -i 's/archive.ubuntu.com/mirrors.tuna.tsinghua.edu.cn/g' /etc/apt/sources.list && \
    sed -i 's/security.ubuntu.com/mirrors.tuna.tsinghua.edu.cn/g' /etc/apt/sources.list

# 安装必要工具和ROS依赖
RUN apt-get update && apt-get install -y \
    curl wget gnupg2 lsb-release sudo ca-certificates build-essential bzip2 \
    ros-noetic-cv-bridge \
    ros-noetic-apriltag-ros \
    && rm -rf /var/lib/apt/lists/*

# 安装 Miniforge
ENV MINIFORGE_URL="https://mirrors.zju.edu.cn/miniforge/Miniforge3-Linux-x86_64.sh"
RUN curl -L ${MINIFORGE_URL} -o /tmp/miniforge.sh \
    && bash /tmp/miniforge.sh -b -p /opt/conda \
    && rm /tmp/miniforge.sh

ENV PATH="/opt/conda/bin:${PATH}"
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

# 配置国内镜像并安装 mamba
RUN conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge/ \
    && conda config --set show_channel_urls yes \
    && conda install -y mamba -c conda-forge

# 工作目录
WORKDIR /root/kuavo_data_challenge
COPY . .

# 解压 Conda 环境并安装项目
RUN if [ -f "myenv.tar.gz" ]; then \
        mkdir -p ./myenv && tar -xzf myenv.tar.gz -C ./myenv && rm myenv.tar.gz; \
    fi && \
    /bin/bash -c "\
        source ./myenv/bin/activate && \
        conda-unpack && \
        pip install -e . && \
        cd ./third_party/lerobot && pip install -e . && \
        conda clean -afy && \
        rm -rf ./myenv/lib/python*/site-packages/*/tests ./myenv/lib/python*/site-packages/*/test ./myenv/pkgs/* \
    "

# =========================
# Stage 2: Final
# =========================
FROM ros:noetic-ros-core-focal

# 设置工作目录
WORKDIR /root/kuavo_data_challenge

# 复制 Conda 环境和项目代码
COPY --from=builder /opt/conda /opt/conda
COPY --from=builder /root/kuavo_data_challenge /root/kuavo_data_challenge

# 环境变量
ENV PATH="/opt/conda/bin:${PATH}"
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

RUN apt-get update && apt-get install -y \
    ros-noetic-cv-bridge \
    ros-noetic-apriltag-ros \
    && rm -rf /var/lib/apt/lists/*

# 保留 ROS 环境变量
# 激活 Conda 环境
RUN echo "source /opt/ros/noetic/setup.bash" >> /root/.bashrc && \
    echo "source /root/kuavo_data_challenge/myenv/bin/activate" >> /root/.bashrc

# 默认命令
CMD ["bash"]
