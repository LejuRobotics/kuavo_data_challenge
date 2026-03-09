# 乐聚机器人赛事与开源资源

本文档汇总与 **乐聚机器人（Leju Robotics）** 相关的官方赛事、开源项目与社区资源，便于快速查找赛事简介与开源项目说明。

!!! abstract "使用说明"
    下表与下文按**赛事**、**开源项目**分类整理；文末提供**链接速查表**与**与本项目关系**说明。更多动态与技术文章可前往 [OpenLET 资讯](https://openlet.openatom.tech/explore/journalism) 浏览。

---

## 赛事一览

| 赛事名称 | 简介 | 官方链接 |
|----------|------|----------|
| **第一届具身智能操作任务挑战赛 & 创业启航营** | 乐聚机器人联合北京通用人工智能研究院、阿里云天池举办。以「真实任务 + 开源数据 + 真机评测」为核心，面向真实工业场景的具身智能落地，提供约 **50 万奖金** 与真机支持。支持仿真赛与真机赛，仿真阶段晋级队伍可进入真机评测。 | [天池赛事主页](https://tianchi.aliyun.com/competition/entrance/532415) · [赛事文档](https://kdc-doc.netlify.app/tianchi/cn/) |
| **ICRA · REAL-I 具身智能挑战赛** | IEEE 机器人与自动化顶会 ICRA 2026 官方赛事。提供约 **9 万美元奖池**、**3 万条多模态数据** 及真机评测，面向具身智能与机器人操作任务。 | 详见 [OpenLET 社区资讯](https://openlet.openatom.tech/explore/journalism) |
| **阿里云天池 · 具身智能挑战赛** | 与阿里云天池平台合作的具身智能相关赛事，可关注天池大赛与 OpenLET 社区获取最新赛题与时间。 | [天池大赛](https://tianchi.aliyun.com/) · [OpenLET 社区](https://openlet.openatom.tech/) |
| **CRAIC 人形机器人挑战赛** | 面向人形机器人的专项挑战赛，由 OpenLET 社区与合作伙伴共同推动。 | [OpenLET 社区](https://openlet.openatom.tech/) |

!!! tip "赛事文档（第一届具身智能操作任务挑战赛）"
    第一届具身智能操作任务挑战赛的**数据、基准代码、仿真与提交说明**等，请参阅官方赛事文档：  
    **[https://kdc-doc.netlify.app/tianchi/cn/](https://kdc-doc.netlify.app/tianchi/cn/)**  
    包含报名、数据集、安装与训练指南、提交方式等。本仓库（Kuavo Data Challenge）为该赛事的基准代码之一。

---

## 开源项目与资源

### 1. Kuavo ROS 开源（kuavo-ros-opensource）

| 项目 | 说明 |
|------|------|
| **简介** | 乐聚 **Kuavo 人形机器人** 的 ROS 开源仓库，涵盖运动控制、全身控制（MPC/WBC）、仿真与实物运行。 |
| **主要内容** | 运动控制接口与 Topic、全身控制器参数、MuJoCo / Gazebo / Isaac Sim 仿真、实物启动与校准（含只上半身、轮臂等配置）、手柄/键盘/Quest3 VR 控制、Docker 构建与运行等。 |
| **仓库** | [GitHub](https://github.com/LejuRobotics/kuavo-ros-opensource) · [Gitee](https://gitee.com/leju-robot/kuavo-ros-opensource) |

与**第一届具身智能操作任务挑战赛**配合使用时，仿真器基于该仓库的 MuJoCo 仿真与机器人模型；基准代码（本仓库）负责数据转换、策略训练与评测接口。

---

### 2. LET 数据集（乐聚开源数据集）

| 项目 | 说明 |
|------|------|
| **简介** | 乐聚在 **AtomGit AI 社区** 开放的数据集资源，面向具身智能与灵巧操作等场景，可用于研究与算法训练。 |
| **资源页** | [AtomGit - lejurobot/let_dataset](https://ai.gitcode.com/lejurobot/let_dataset) |

社区中还有「灵巧操作 LET 数据集」等细分资源，可在 OpenLET 与 AtomGit 中进一步查看。

---

### 3. 乐聚 OpenLET 社区

| 项目 | 说明 |
|------|------|
| **简介** | 乐聚机器人在 **开放原子开源基金会** 下建立的 **OpenLET 社区**，聚焦人形机器人领域的**真实数据开源与生态共建**，提供资讯、开源数据集、算法框架、应用案例、技术经验分享以及具身智能相关挑战赛信息。 |
| **社区首页** | [https://openlet.openatom.tech/](https://openlet.openatom.tech/) |
| **资讯与动态** | [OpenLET 资讯](https://openlet.openatom.tech/explore/journalism) |

可在社区中获取赛事公告、数据集更新、技术文章（如 Kuavo 配置、Roban 倒地起身、强化学习与 Mimic 等）以及各挑战赛的说明与入口。

---

## 链接速查

| 类型 | 名称 | 链接 |
|------|------|------|
| 赛事文档 | 第一届具身智能操作任务挑战赛 官方文档 | [kdc-doc.netlify.app/tianchi/cn/](https://kdc-doc.netlify.app/tianchi/cn/) |
| 赛事报名 | 天池 · 乐聚第一届具身智能操作任务挑战赛 | [tianchi.aliyun.com/competition/entrance/532415](https://tianchi.aliyun.com/competition/entrance/532415) |
| 开源仓库 | Kuavo ROS 开源（GitHub） | [github.com/LejuRobotics/kuavo-ros-opensource](https://github.com/LejuRobotics/kuavo-ros-opensource) |
| 开源仓库 | Kuavo ROS 开源（Gitee） | [gitee.com/leju-robot/kuavo-ros-opensource](https://gitee.com/leju-robot/kuavo-ros-opensource) |
| 数据集 | LET 数据集（AtomGit） | [ai.gitcode.com/lejurobot/let_dataset](https://ai.gitcode.com/lejurobot/let_dataset) |
| 社区 | 乐聚 OpenLET 社区 | [openlet.openatom.tech](https://openlet.openatom.tech/) |
| 资讯 | OpenLET 资讯与动态 | [openlet.openatom.tech/explore/journalism](https://openlet.openatom.tech/explore/journalism) |

---

!!! info "与本项目的关系"
    当前 **Kuavo Data Challenge** 仓库为「第一届具身智能操作任务挑战赛 & 创业启航营」的**基准代码**之一，与 **kuavo-ros-opensource** 仿真器配合使用。更多赛事规则、数据与提交方式请以赛事官网与上述官方文档为准。

---

## 推荐阅读

更多赛事动态、技术经验与开源资源更新，请前往 [OpenLET 资讯](https://openlet.openatom.tech/explore/journalism) 浏览。
