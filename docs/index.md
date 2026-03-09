# Kuavo Data Challenge 文档

欢迎来到 **Kuavo Data Challenge** 官方文档站。本文档帮助你完成从数据准备、策略训练到仿真与真机部署的完整流程。

!!! abstract "文档说明"
    本站为**中文文档**，导航与链接文案统一为中文。本站提供**入门路径**、**项目能力概览**与**常用入口**。零基础读者建议从「零基础完整学习教程」开始；有环境基础的可直接看「快速开始」；想了解近期新增功能可查看「更新与动态」。

---

## 入门路径

根据你的情况选择一条路径即可：

| 适合人群 | 推荐入口 | 说明 |
|----------|----------|------|
| **零基础**（不懂模仿学习/VLA、不会装环境、不会下模型） | [零基础完整学习教程](getting_started/beginner_tutorial.md) | 从概念到环境、数据、训练与部署的完整步骤，含提示框与表格归纳 |
| **有环境基础**（已熟悉 Python/Conda，想快速跑通） | [快速开始](getting_started/quick_start.md) | 环境安装、数据转换、训练与仿真/真机的一条龙流程概览 |
| **需查阅目录与关键文件** | [项目文件构成](getting_started/file_structure.md) | 仓库目录结构、配置与训练/部署脚本对应关系 |

---

## 项目能力概览

| 模块 | 说明 |
|------|------|
| **数据** | 数据准备与 LeRobot v4.3 数据集格式 |
| **策略** | 多种策略（ACT / Diffusion / PI0 / PI05 / Groot / SmolVLA）的训练与部署 |
| **部署** | 仿真与真机部署流程（基于 ROS 和 Kuavo SDK） |
| **进阶** | RGB / RGB+Depth 融合方案与高级特性 |

---

## 快捷入口

!!! tip "了解近期更新"
    想快速了解相较初始项目**新增了哪些功能**？请查看 **[更新与动态](updates.md)**，包含深度图像支持、多卡并行、LeRobot 0.4.3、帧对齐、末端增量控制、更多 VLA 模型等说明。

!!! success "乐聚赛事与开源资源"
    想了解乐聚机器人相关**赛事**（如第一届具身智能操作任务挑战赛、ICRA REAL-I 等）和**开源项目**（Kuavo ROS 开源、LET 数据集、OpenLET 社区）？请查看 **[乐聚赛事与开源资源](leju_ecosystem.md)**。

---

## 常用文档索引

| 主题 | 文档 | 说明 |
|------|------|------|
| 安装与环境 | [安装指南](getting_started/installation.md) | 依赖与环境配置 |
| 快速开始 | [快速开始](getting_started/quick_start.md) | 一条龙上手流程 |
| 策略概览 | [策略概览](concepts/policy_overview.md) | 各策略对比与选型 |
| 训练流程 | [训练流水线](training/pipeline.md) | 统一训练入口与配置 |
| 仿真测试 | [仿真自动测试](deployment/sim_auto_test.md) | 仿真环境评测 |
| 真机评测 | [真机评测](deployment/real_eval.md) | 真机部署与 SDK |
| 常见问题 | [常见问题](faq.md) | 报错与常见问题汇总 |

---

## 更多资讯与社区

乐聚机器人在开放原子开源基金会下建有 **OpenLET 社区**，提供赛事公告、技术经验分享、开源数据集与算法框架等。可前往 [OpenLET 资讯](https://openlet.openatom.tech/explore/journalism) 浏览最新动态。
