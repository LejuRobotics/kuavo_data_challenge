# 🤖 Kuavo Deploy - 机器人部署模块

> 基于ROS的Kuavo机器人部署模块，支持真实机器人和仿真环境的模型推理、轨迹回放和机械臂控制等功能。

## 📁 模块结构

```
kuavo_deploy/
├── kuavo_env/                    # 机器人环境模块
│   ├── kuavo_real_env/           # 真实机器人环境
│   │   └── KuavoRealEnv.py       # 真实机器人环境实现
│   ├── kuavo_sim_env/            # 仿真环境
│   │   └── KuavoSimEnv.py        # 仿真环境实现
│   └── KuavoBaseRosEnv.py        # ROS环境基类
├── examples/                     # 示例代码与评估
│   ├── eval/                     # 评估脚本
│   │   ├── eval_kuavo.py         # Kuavo环境评估脚本
│   │   └── auto_test/            # 自动化测试
│   │       ├── eval_kuavo.py     # Kuavo环境自动化评估脚本
│   │       └── eval_kuavo_autotest.py  # 自动化测试脚本
│   └── scripts/                  # 控制脚本
│       ├── script.py             # 主要控制脚本
│       ├── controller.py         # 机械臂控制器
│       └── script_auto_test.py   # 自动化控制脚本
├── utils/                        # 工具模块
│   └── logging_utils.py          # 日志工具
└── eval_others.py                # 其他环境如pusht、aloha等评估脚本

configs/
└── deploy/                # 部署配置（新位置）
    ├── config_inference.py       # 推理配置加载器
    ├── config_kuavo_env.py       # 环境配置加载器
    ├── kuavo_real_env.yaml       # kuavo真实环境参数
    ├── kuavo_sim_env.yaml        # kuavo仿真环境参数
    └── others_env.yaml           # 其他评估环境参数

outputs/
└── eval/
    └── {task}/                   # 任务名称（如 pick_place, push 等）
        └── {method}/             # 方法名称（如 diffusion, act 等）
            └── {timestamp}/      # 运行时间戳
                └── epoch{epoch}/ # 评测所用的模型权重轮次
                    ├── evaluation.log           # 手动评估日志
                    ├── evaluation_autotest.log  # 自动化评测日志
                    ├── evaluation_autotest.json # 自动化评测结果json
                    ├── rollout_0_observation.images.head_cam_h.mp4   # episode0 头部相机视频
                    ├── rollout_0_observation.images.wrist_cam_l.mp4  # episode0 左腕相机视频
                    ├── rollout_0_observation.images.wrist_cam_r.mp4  # episode0 右腕相机视频
                    ├── rollout_1_observation.images.head_cam_h.mp4   # episode1 头部相机视频
                    ├── rollout_1_observation.images.wrist_cam_l.mp4  # episode1 左腕相机视频
                    ├── rollout_1_observation.images.wrist_cam_r.mp4  # episode1 右腕相机视频
                    └── ...  # 以此类推（rollout_2、rollout_3、...）
```

## ✨ 主要特性

### 🎯 核心功能
- **真实机器人控制**: 支持Kuavo真实机器人的机械臂控制
- **仿真环境**: 提供仿真环境用于模型测试和验证
- **模型推理**: 支持Diffusion Policy等模型的实时推理
- **轨迹回放**: 支持ROS bag文件的轨迹回放功能
- **多模态输入**: 支持头部相机和手腕相机的图像输入
- **末端执行器**: 支持强脑灵巧手和夹爪两种末端执行器

### 🔧 技术特性
- **ROS集成**: 基于ROS的机器人控制架构
- **Gymnasium环境**: 标准化的强化学习环境接口
- **实时控制**: 支持10Hz的实时推理频率
- **安全机制**: 内置关节限制和安全检查
- **日志系统**: 完整的日志记录和调试功能

### 🎮 控制模式
- **关节控制**: 直接控制机械臂关节角度
- **轨迹插值**: 平滑的轨迹插值算法
- **紧急停止**: 支持中断和紧急停止功能

## 🚀 快速开始

### 1. 从github拉取代码仓库

```bash
git clone https://github.com/LejuRobotics/kuavo-data-challenge.git
cd kuavo-data-challenge
git submodule update --init --recursive
```

### 2. 配置环境（ros、conda、python环境）

参考 [环境配置指南](readme/setup_env.md)

### 3. 真机有线通讯配置

参考 [机器人连接配置](readme/setup_robot_connection.md)

### 4. 真机推理

参考 [推理指南](readme/inference.md)