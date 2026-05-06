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

# 提示用户输入，并将输入结果存入变量 INSTALL_ROS
read -p "是否需要检查 ROS 环境依赖？[Y/n] (默认: Y): " CHECK_ROS

# 如果用户直接按回车，输入为空，则默认赋值为 "Y"
CHECK_ROS=${CHECK_ROS:-Y}

# 判断用户输入是否为 Y, y 或者 yes
if [[ "$CHECK_ROS" == "Y" || "$CHECK_ROS" == "y" || "$CHECK_ROS" == "yes" || "$CHECK_ROS" == "Yes" ]]; then
    # 检查文件是否存在（虽然肯定存在，但保留检查是个好习惯，防患于未然）
    if [ -f "requirements_ros_env.txt" ]; then
        echo "⏳ 正在检查并安装 ROS 环境依赖..."
        
        # 将 pip install 放在 if 中，如果成功返回 0，失败返回非 0
        if pip install -r requirements_ros_env.txt; then
            echo "✅ ROS 环境依赖检查/安装完成！"
        else
            # pip 报错时会进入这里
            echo "❌ 错误：ros依赖库不全，请仔细核对ROS是否安装好"
            # 退出脚本，防止在缺少依赖的情况下继续执行后续代码
            exit 1
        fi
    else
        echo "❌ 错误：未找到 requirements_ros_env.txt 文件，请确认它与此脚本在同一目录下！"
        exit 1
    fi
else
    # 如果用户输入 n、N 或其他字符
    echo "⏭️  已跳过 ROS 环境依赖的检查与安装。"
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
# pip uninstall pyarrow -y
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

# 尝试在 Python 中导入 flash_attn，并将输出和错误信息丢弃 (&> /dev/null)
if python -c "import flash_attn" &> /dev/null; then
    echo "检测到 flash_attn 已安装，跳过编译。"
else
    echo "未检测到 flash_attn，准备开始编译安装..."
    
    # 进入目录，如果目录不存在则报错并退出
    cd flash_attn-2.8.3/ || { echo "错误: 找不到 flash_attn-2.8.3/ 目录"; exit 1; }
    
    # 使用 MAX_JOBS=2 限制编译核心数，防止内存溢出 (OOM)
    echo "正在使用 MAX_JOBS=2 编译安装 flash-attn，这可能需要一些时间..."
    MAX_JOBS=2 python setup.py install
    
    # 返回上级目录
    cd ../
    
    echo "flash_attn 安装流程执行完毕。"
fi


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