#!/bin/bash

ENV_NAME="kdc_v0"
ENV_DIR="./${ENV_NAME}"

# 创建目录并解压环境包
mkdir -p "${ENV_DIR}"
tar -xzf "${ENV_NAME}.tar.gz" -C "${ENV_DIR}"

echo "环境解压完成，目录：${ENV_DIR}"

# 这里演示直接用解压后的python运行
echo "使用解压环境的 python 运行测试："
"${ENV_DIR}/bin/python" --version

# 激活环境
echo "激活环境："
source "${ENV_DIR}/bin/activate"

# 进入激活环境后，运行python
echo "环境激活，运行 python："
python --version

# 运行 conda-unpack 以清理前缀
echo "运行 conda-unpack 清理前缀："
conda-unpack

echo "脚本执行完毕。"

