# UniLab 强化学习全栈教程

> 从 MuJoCo 语法到 RL 策略部署的完整开发指南

## 教程结构

| 章节 | 内容 | 预计时间 |
|------|------|----------|
| [01. MuJoCo 基础与后端](./01_mujoco_basics.md) | MJCF 语法、物理引擎、SimBackend 抽象 | 30 min |
| [02. 地形建模](./02_terrain_modeling.md) | 高度场地形、7 种地形类型、课程模式 | 25 min |
| [03. 机器人建模](./03_robot_modeling.md) | XML 模型结构、关节/执行器/传感器、Keyframe | 40 min |
| [04. RL 算法与训练](./04_rl_algorithms.md) | PPO/SAC 配置、Hydra 体系、训练流水线 | 35 min |
| [05. URDF 机器人移植](./05_urdf_import.md) | URDF→MJCF 转换、后处理、Keyframe 调试 | 45 min |
| [06. 奖励函数设计](./06_reward_function.md) | RewardContext、自定义奖励、Dispatch 机制 | 30 min |
| [07. 参数调优实战](./07_parameter_tuning.md) | 超参调整、地形平衡、课程学习、域随机化 | 35 min |

## 前置要求

- Python 3.10+, uv 包管理器
- MuJoCo 基础概念（关节、刚体、自由度）
- 强化学习基础（PPO、价值函数、策略梯度）
- 克隆仓库：`cd /home/robot/xiaoq/wheel_legged_RL_unilab`

## 项目快速导航

```
wheel_legged_RL_unilab/
├── conf/                           # Hydra 配置中心
│   └── ppo/task/xqrobotV2_walk_flat/mujoco.yaml  # ★ 训练入口
├── src/unilab/
│   ├── envs/locomotion/xqrobotV2/  # XqRobotV2 环境实现
│   ├── terrains/                   # 地形生成器
│   ├── assets/robots/xqrobotV2/    # 机器人 XML + 网格
│   ├── base/backend/mujoco/        # MuJoCo 后端
│   └── tools/import_robot.py       # URDF→MJCF 工具
├── shell/                          # 训练/评估脚本
├── assess/                         # 策略评估框架
└── tools/email/                    # 自循环邮件报告
```

## 关键设计原则

1. **机器人 XML 与场景 XML 分离**：`robot.xml` 定义物理结构，`scene_flat.xml` / `locomotion_task.xml` 定义 Keyframe 和场景
2. **配置驱动**：所有超参通过 Hydra YAML 配置，不硬编码
3. **Enforce contract**：环境通过 `NpEnvState` 契约与训练器交互，后端通过 `SimBackend` 抽象
4. **单一职责**：reward 函数是纯函数，env 层负责组装，backend 负责物理仿真

## 学习路径建议

- **想快速跑起来** → 01 → 04 → 07
- **想移植自己的机器人** → 01 → 03 → 05 → 06 → 07
- **想深入理解框架** → 按顺序 01→07

---

> 本教程基于 XqRobotV2 轮腿双足机器人作为主要示例。
> 项目架构详见仓库主 [README.md](../../README.md)，AI 开发规范见 [AGENTS.md](../../AGENTS.md)。
