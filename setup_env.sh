#!/bin/bash

# 设置遇到错误立即停止执行
set -e
echo "==================================================="
echo "🚀 欢迎使用 KDC 项目环境配置脚本! 先进行pip换源"
echo "==================================================="
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple  # 建议首先换源，能加快下载安装速度


echo "==================================================="
echo "👉 第 1 步：检查并安装 ROS 环境依赖 (requirements_ros_env.txt)"
echo "==================================================="

# 检查文件是否存在
if [ -f "requirements_ros_env.txt" ]; then
    # pip 会自动检查，如果已安装则跳过，未安装则下载
    pip install -r requirements_ros_env.txt
    echo "✅ ROS 环境依赖检查/安装完成！"
else
    echo "❌ 错误：未找到 requirements_ros_env.txt 文件，请确认它与此脚本在同一目录下！"
    exit 1
fi

echo ""
echo "==================================================="
echo "👉 第 2 步：安装主项目依赖 (requirements.txt)"
echo "==================================================="

if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
    echo "✅ 主项目依赖安装完成！"
else
    echo "❌ 错误：未找到 requirements.txt 文件，请确认它与此脚本在同一目录下！"
    exit 1
fi

echo ""
echo "==================================================="
echo "👉 第 3 步：运行全局依赖冲突检查"
echo "==================================================="
# pip check 会检查当前环境中安装的所有包是否存在版本不兼容的问题
if pip check; then
    echo "🎉 恭喜！所有依赖均已安装且没有检测到版本冲突！"
else
    echo "⚠️ 注意：pip check 检测到了一些版本冲突，请根据上面的提示核对。"
fi


echo ""
echo "==================================================="
echo "👉 第 4 步：安装特定版本的 ffmpeg 和 pyarrow 以及 pyaudio"
echo "==================================================="
conda install ffmpeg=6.1.1 -y
pip uninstall pyarrow -y
conda install pyarrow -y
conda install pyaudio -y

echo ""
echo "==================================================="
echo "👉 第 5 步: 安装VLA 所需要的flash-attn,请先确认nvcc -V cuda版本大于11.7, 如需升级请访问https://developer.nvidia.com/cuda-12-2-0-download-archive?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=20.04&target_type=deb_loca"
echo "==================================================="

if [ ! -d "flash_attn-2.8.3" ]; then
    echo "未检测到 flash_attn-2.8.3 文件夹，开始下载并解压..."
    wget https://files.pythonhosted.org/packages/3b/b2/8d76c41ad7974ee264754709c22963447f7f8134613fd9ce80984ed0dab7/flash_attn-2.8.3.tar.gz
    tar -zxvf flash_attn-2.8.3.tar.gz
else
    echo "文件夹 flash_attn-2.8.3 已存在，跳过下载和解压。"
fi

cd flash_attn-2.8.3/
# 使用 MAX_JOBS=2 限制编译核心数，防止内存溢出 (OOM)
MAX_JOBS=2 python setup.py install
cd ../


echo "==================================================="
echo "👉 第 6 步：检查并配置 Hugging Face 镜像源"
echo "==================================================="
BASHRC_FILE="$HOME/.bashrc"

# 检查 ~/.bashrc 文件是否存在，不存在则创建（兜底防护）
if [ ! -f "$BASHRC_FILE" ]; then
    touch "$BASHRC_FILE"
fi

# 检查是否已经存在该配置
if grep -q "HF_ENDPOINT=https://hf-mirror.com" "$BASHRC_FILE"; then
    echo "✅ Hugging Face 镜像源已配置在 ~/.bashrc 中，无需重复添加。"
else
    echo "⚠️ 未检测到 Hugging Face 镜像源配置，正在添加到 ~/.bashrc..."
    # 写入配置到 bashrc 末尾
    echo "" >> "$BASHRC_FILE"
    echo "# Hugging Face Mirror Endpoint" >> "$BASHRC_FILE"
    echo "export HF_ENDPOINT=https://hf-mirror.com" >> "$BASHRC_FILE"
    
    echo "✅ 镜像源已成功添加至 ~/.bashrc！"
fi
source "$BASHRC_FILE"  # 立即生效配置