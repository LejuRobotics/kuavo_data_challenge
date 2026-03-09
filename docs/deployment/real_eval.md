# 真机评测

本节描述在 **Kuavo 真实机器人** 上部署并评估已训练策略的流程。真机评估依赖 [kuavo-ros-opensource](https://github.com/LejuRobotics/kuavo-ros-opensource) 提供底层控制、MPC、WBC 与 ROS 接口；本项目的 `kuavo_deploy` 在此基础上加载策略、执行推理并通过 ROS 下发动作。

---

## 1. 概述

真机评估的核心流程为：

1. **启动机器人**：通过 kuavo-ros-opensource 的 launch 启动控制器、MPC、WBC；
2. **到达工作姿态**：使用 `go.bag` 回放预录轨迹，使机械臂移动到任务起始位置；
3. **策略推理**：加载已训练 checkpoint，读取相机与状态观测，推理得到动作并下发；
4. **安全控制**：支持暂停/停止信号，关节限幅由底层控制保障。

**入口脚本**：`kuavo_deploy/src/scripts/script.py`  
**核心逻辑**：`kuavo_deploy/src/eval/real_single_test.py` 中的 `kuavo_eval()`

---

## 2. 前置条件：kuavo-ros-opensource

真机评估前必须完成 [kuavo-ros-opensource](https://github.com/LejuRobotics/kuavo-ros-opensource) 的环境配置与机器人启动。以下摘录其 README 要点，详情请以官方仓库为准。

### 2.1 克隆与构建

```bash
# HTTPS
git clone https://gitee.com/leju-robot/kuavo-ros-opensource.git

# 或 GitHub
# git clone https://github.com/LejuRobotics/kuavo-ros-opensource.git
```

**Docker 环境**（无实机时可用于编译与仿真）：

- 构建镜像：`./docker/build.sh`
- 运行容器：`./docker/run.sh`
- 编译：`catkin config -DCMAKE_ASM_COMPILER=/usr/bin/as -DCMAKE_BUILD_TYPE=Release`，`catkin build humanoid_controllers`

**实机环境**：

- 旧镜像需安装依赖：`./docker/install_env_in_kuavoimg.sh`
- 编译：`source installed/setup.bash`，`catkin build humanoid_controllers`

### 2.2 机器人版本与质量

- **ROBOT_VERSION**：通过环境变量设置（如 `export ROBOT_VERSION=45` 表示 4.5 版本），需与实物匹配；
- **总质量**：写入 `~/.config/lejuconfig/TotalMassV${ROBOT_VERSION}`，模型编译时会读取；修改后会自动清除 OCS2 缓存并重新编译。

### 2.3 末端执行器配置

在 `src/kuavo_assets/config/kuavo_v$ROBOT_VERSION/kuavo.json` 中设置 `EndEffectorType`：

| 值 | 说明 |
|----|------|
| `none` | 无末端执行器或屏蔽 |
| `qiangnao` | 灵巧手（默认） |
| `lejuclaw` | 二指夹爪 |
| `qiangnao_touch` | 触觉灵巧手 |

需与 `kuavo_env.yaml` 中的 `eef_type`（如 `rq2f85`、`leju_claw`、`qiangnao`）一致。

### 2.4 实机启动与校准

- **首次开机**：`roslaunch humanoid_controllers load_kuavo_real.launch cali:=true`，在 cali 模式下完成零点标定；
- **日常运行**：`roslaunch humanoid_controllers load_kuavo_real.launch`，机器人进入待站立后按 `o` 启动；
- **仅上半身**：修改 `kuavo.json` 中 `only_half_up_body: true`，使用 `load_kuavo_real.launch` 或 `load_kuavo_real_half_up_body.launch`（i7 NUC 时）。

零点与校准流程详见官方 [readme](https://github.com/LejuRobotics/kuavo-ros-opensource)。

---

## 3. 真机评估任务（script.py）

`script.py` 提供多种任务模式，用于在真机上到达工作位置并执行策略推理。

| 任务 | 命令 | 说明 |
|------|------|------|
| `go` | `--task go` | 插值到 bag 第一帧，再回放 bag 前往工作位置 |
| `run` | `--task run` | 从当前位置直接运行模型推理 |
| `go_run` | `--task go_run` | 先 go 到达工作位置，再 run 执行推理 |
| `here_run` | `--task here_run` | 插值到 bag 最后一帧，然后运行模型 |
| `back_to_zero` | `--task back_to_zero` | 倒放 bag 回到零位（推理中断后使用） |

### 3.1 运行示例

```bash
# 到达工作位置并执行推理
python kuavo_deploy/src/scripts/script.py \
  --task go_run \
  --config configs/deploy/kuavo_env.yaml
```

```bash
# 仅从当前位置运行模型
python kuavo_deploy/src/scripts/script.py \
  --task run \
  --config configs/deploy/kuavo_env.yaml
```

### 3.2 go.bag 与 go_bag_path

真机推理前通常需要将机械臂移动到**任务起始姿态**。`go.bag` 是预录的 ROS bag，包含 `/kuavo_arm_traj`、`/leju_claw_command`（或 `/control_robot_hand_position`、`/gripper_command`）等话题，用于回放轨迹。

在 `configs/deploy/kuavo_env.yaml` 中配置：

```yaml
inference:
  go_bag_path: /path/to/your/go.bag  # 预录轨迹的完整路径
```

- **go**：插值到 bag 第一帧关节角，再按 100Hz 均匀发布 bag 内消息；
- **play_bag(reverse=True)**：倒序回放，用于 `back_to_zero` 回到零位。

bag 中的关节角度需与训练时使用的坐标系与单位一致（如度转弧度由脚本处理）。

---

## 4. 配置文件（真机与仿真差异）

| 字段 | 仿真 | 真机 |
|------|------|------|
| `env.env_name` | `Kuavo-Sim` | `Kuavo-Real` |
| `inference.go_bag_path` | 比赛仿真器暂不用 | **必填**，预录轨迹路径 |
| ROS 话题 | 仿真器发布 | 真机传感器与控制器 |

`obs_key_map`、`image_size`、`depth_range`、`eef_type` 等需与训练保持一致；真机时 `only_arm` 等根据任务选择。

---

## 5. kuavo_service：服务端 / 客户端架构（可选）

当策略推理在**远程机**（如带 GPU 服务器）上运行、机器人控制器在**本地**时，可使用 `kuavo_service` 解耦：

- **Policy Server**：加载策略，接收观测、返回动作（ZeroMQ 等）；
- **Policy Client**：在机器人端作为 `policy_type=client`，向 Server 请求推理，不本地加载模型。

配置中 `policy_type: client` 时，`script.py` / `real_single_test` 会使用 `PolicyClient`，通过网络获取动作。具体地址、端口与启动方式见 `kuavo_deploy/kuavo_service/` 下的实现。

---

## 6. 安全与控制信号

### 6.1 暂停与停止

- **暂停/恢复**：`kill -USR1 <pid>`，推理循环会阻塞直至恢复；
- **停止**：`kill -USR2 <pid>`，收到后退出当前任务。

脚本启动时会打印进程 pid 及上述命令。

### 6.2 关节限幅与急停

- 关节上下限由 `kuavo_env.yaml` 的 `env.limits` 定义；
- 底层控制器（MPC、WBC）负责限幅与安全保护；
- **急停**：物理急停按钮生效时，电机下电，策略推理应配合停止。

### 6.3 建议

- 首次真机测试建议在**悬挂或保护**下进行；
- 确认 `go.bag` 轨迹与当前工作空间、障碍物无冲突；
- 推理前确保机器人已站立/就绪，并有人值守。

---

## 7. PI0 / PI05 真机部署要点

与仿真类似，真机部署 PI0/PI05 时需：

1. **RGB-only**：移除 depth 观测键；若训练为 RGB-only，部署保持一致；
2. **无语言**：注入空 `task` 以满足 tokenizer；
3. **Tokenizer 路径**：设置 `PALIGEMMA_TOKENIZER_PATH`，或在配置中指定本地路径。

具体逻辑见 `real_single_test.py` 与 `KuavoRealEnv`；与仿真的差异主要为观测来源（真机相机、关节话题）和 `go.bag` 流程。

---

## 8. 输出与日志

- **输出目录**：`outputs/eval/<task>/<method>/<timestamp>/epoch<epoch>/`
- **evaluation.log**：评估时间戳、episode 数、各 episode 奖励、成功率
- 真机默认**不保存** rollout 视频（避免内存堆积），若需可取消 `real_single_test.py` 中相应注释。

---

## 9. 相关资源

- **kuavo-ros-opensource**：<https://github.com/LejuRobotics/kuavo-ros-opensource> — 机器人控制、MPC、WBC、launch 与校准流程
- [仿真自动测试](sim_auto_test.md)：仿真自动测试流程
- [快速开始](../getting_started/quick_start.md)：从数据到训练与评估
- [策略概览](../concepts/policy_overview.md)：策略类型与部署支持
