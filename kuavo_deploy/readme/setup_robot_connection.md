# 边侧机通信配置方案教程（基于下位机桥接边侧机和上位机两台机器）

---

## 1. 用网线+usb转网口连接下位机、边侧机，查看下位机有线网口信息

- 查看有线网络接口：

```bash
nmcli connection show
```

得到类似如下的有线网口信息

```bash
NAME                   UUID                                  TYPE      DEVICE              
enxc8a362b260f5        871ce4e7-3633-47ab-a88b-c0613c6ca67a  ethernet  enxc8a362b260f5 
Wired connection 2     65e0a36e-e998-38cd-ad8c-a73ad59e10c0  ethernet  enx00e04c684355 
```

---

## 2. 创建并配置桥接接口

假设下位机网口：

- `enx00e04c684355`（连边侧机）  
- `enxc8a362b260f5`（连上位机）

### 新建桥接配置脚本 `setup_bridge.sh`, 注意桥接接口名不要与已存在的重复

```bash
#!/bin/bash

# 桥接接口名称
BRIDGE=br0

# 物理网卡名称（替换成你的接口名）
IFACE1=enx00e04c684355
IFACE2=enxc8a362b260f5

# 检查桥接接口是否存在
if ! ip link show "$BRIDGE" &>/dev/null; then
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

```

执行：

```bash
sudo chmod +x setup_bridge.sh
sudo bash setup_bridge.sh
```

---

## 3.  边侧机上分配静态IP的步骤（上位机和下位机有线连接的ip一般已经配好，一般为192.168.26.x网段，注意边侧机网段应在同一网段）

### 查看已有的网络连接配置

```bash
nmcli connection show
```

该命令列出当前所有网络连接及其名称、UUID、类型和设备。

### 修改指定有线连接为静态IP配置

假设要修改的连接名称是 `"有线连接 1"`，配置静态IP地址和掩码为 `192.168.26.10/24`，不设置网关，手动配置：

```bash
sudo nmcli connection modify "有线连接 1" ipv4.addresses 192.168.26.10/24 ipv4.gateway "" ipv4.method manual
```

说明：

- `ipv4.addresses` 设置静态IP地址及子网掩码  
- `ipv4.gateway` 留空表示无默认网关  
- `ipv4.method manual` 设置为手动静态IP配置  

### 激活修改后的网络连接

```bash
sudo nmcli connection up "有线连接 1"
```

此命令重新激活指定连接，使配置生效。

### 验证网络配置

查看当前接口IP：

```bash
ip addr show
```

或查看连接详情：

```bash
nmcli connection show "有线连接 1"
```

完成以上步骤后，边侧机的有线接口将使用静态IP `192.168.26.10`，在对应子网内通信。

---

## 4. 验证步骤

- 测试互相 ping：

```bash
# 示例
ping 192.168.26.12  # 边侧机 ping 上位机
ping 192.168.26.10  # 上位机 ping 边侧机
```

- ping通后在边侧机设置ROS_IP,ROS_MASTER_URI:

```bash
# 1. 在 ~/.bashrc 文件末尾添加注释，方便识别
echo "# ROS网络配置" >> ~/.bashrc

# 2. 添加 ROS_IP 环境变量
echo "export ROS_IP=192.168.26.10" >> ~/.bashrc

# 3. 添加 ROS_MASTER_URI 环境变量
echo "export ROS_MASTER_URI=http://192.168.26.1:11311" >> ~/.bashrc

# 4. 立即让修改生效
source ~/.bashrc

```

- 测试rostopic通信

```bash
# 2. Verify ros topic(有数据则没问题)
rostopic echo /sensors_data_raw
rostopic echo /cam_h/color/image_raw/compressed
rostopic echo /cam_r/color/image_raw/compressed
rostopic echo /cam_l/color/image_raw/compressed
rostopic echo /leju_claw_state # if you use leju_claw
rostopic echo /dexhand/state   # if you use qiangnao
```