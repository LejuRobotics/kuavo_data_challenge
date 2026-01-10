#!/bin/bash

IMAGE_NAME="kdc_v0"
CONTAINER_NAME="kdc_v0"
IMAGE_TAR="${IMAGE_NAME}.tar"   # 镜像文件路径

# 如果容器存在，先删除
if [ "$(docker ps -aq -f name=${CONTAINER_NAME})" ]; then
    echo "Container exists. Removing..."
    docker rm -f ${CONTAINER_NAME}
fi

# 检查镜像是否存在，如果存在则删除
EXISTING_IMAGE=$(docker images -q $IMAGE_NAME)
if [ "$EXISTING_IMAGE" ]; then
    echo "Image $IMAGE_NAME already exists. Removing..."
    docker rmi -f $IMAGE_NAME
fi

# 直接加载镜像
if [ -f "$IMAGE_TAR" ]; then
    echo "Loading image from $IMAGE_TAR..."
    docker load -i "$IMAGE_TAR"
else
    echo "Error: $IMAGE_TAR not found!"
    exit 1
fi

# 创建并启动新的容器
docker run --gpus all -it \
    --net=host \
    -e ROS_MASTER_URI=http://127.0.0.1:11311 \
    -e ROS_IP=127.0.0.1 \
    --name ${CONTAINER_NAME} \
    ${IMAGE_NAME} bash


