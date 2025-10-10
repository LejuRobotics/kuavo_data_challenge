#!/bin/bash

# 桥接接口名称
BRIDGE=br0

# 物理网卡名称（替换成你的接口名）
IFACE1=enx00e04c684355
IFACE2=enxc8a362b260f5

# 检查桥接接口是否存在
if ip link show "$BRIDGE" &>/dev/null; then
    echo "错误：桥接接口 $BRIDGE 已存在，请先删除或使用其它名称。"
    exit 1
fi

# 检查物理网卡是否存在
if ! ip link show "$IFACE1" &>/dev/null; then
    echo "错误：物理网卡 $IFACE1 不存在，退出脚本。"
    exit 1
fi

if ! ip link show "$IFACE2" &>/dev/null; then
    echo "错误：物理网卡 $IFACE2 不存在，退出脚本。"
    exit 1
fi

# 桥接IP地址
BRIDGE_IP=192.168.26.1/24

echo "=== 停用接口 ==="
sudo ip link set dev $IFACE1 down
sudo ip link set dev $IFACE2 down

echo "=== 清除接口IP地址 ==="
sudo ip addr flush dev $IFACE1
sudo ip addr flush dev $IFACE2

echo "=== 创建桥接接口 ==="
sudo ip link add name $BRIDGE type bridge

echo "=== 将物理接口加入桥接 ==="
sudo ip link set dev $IFACE1 master $BRIDGE
sudo ip link set dev $IFACE2 master $BRIDGE

echo "=== 启动物理接口和桥接接口 ==="
sudo ip link set dev $IFACE1 up
sudo ip link set dev $IFACE2 up
sudo ip link set dev $BRIDGE up

echo "=== 给桥接接口分配IP地址 ==="
sudo ip addr add $BRIDGE_IP dev $BRIDGE

echo "=== 显示接口状态 ==="
ip addr show $BRIDGE
ip addr show $IFACE1
ip addr show $IFACE2

echo "临时关闭桥接流量经过 iptables 过滤"

sudo sysctl -w net.bridge.bridge-nf-call-iptables=0

echo "=== 桥接配置完成 ==="
