# XqRobotV2 训练备份

## 目录结构

```
backup/
├── README.md                              # 本文件
└── XqRobotV2WalkFlat/                     # Flat Walk 任务
    └── flat_walk_ppo_v1/                  # PPO v1 基准版本
        ├── README.md                      # 版本说明
        ├── run_config.json                # Hydra 完整配置快照
        ├── run_summary.json               # 训练摘要
        ├── git_commit.txt                 # 代码版本号
        ├── uncommitted.diff               # 未提交改动
        ├── conf/                          # 训练配置
        │   └── ppo/task/xqrobotV2_walk_flat/
        │       ├── mujoco.yaml
        │       └── motrix.yaml
        ├── src/                           # 环境代码
        │   └── unilab/envs/locomotion/xqrobotV2/
        │       ├── __init__.py
        │       ├── base.py
        │       └── joystick.py
        ├── shell/                         # 启动脚本
        │   ├── train_ppo_flat.sh
        │   ├── eval_ppo_flat.sh
        │   └── tensorboard.sh
        └── scene_flat.xml                 # 机器人场景
```

## 设计原则

- **不自带策略模型**（太大，且在 `logs/` 中已有）
- **备份所有源码和配置**，可直接拷贝回项目目录启动训练
- 目录命名与 `logs/` 一致（`XqRobotV2WalkFlat` 等）
- 版本命名: `{task}_{algo}_v{version}`

## 版本列表

| 任务 | 版本 | 算法 | 奖励 | 日期 | 说明 |
|------|------|------|------|------|------|
| XqRobotV2WalkFlat | [flat_walk_ppo_v1](XqRobotV2WalkFlat/flat_walk_ppo_v1/README.md) | PPO | 54.1 | 2026-06-26 | 位置控制，50Hz |
| XqRobotV2WalkFlat | [flat_walk_ppo_v2](XqRobotV2WalkFlat/flat_walk_ppo_v2/README.md) | PPO | 训练中 | 2026-06-27 | **速度控制，100Hz，轮子对称** |

## 使用指南

### 完整恢复（启动训练）

```bash
# 拷贝备份到项目根目录
TASK=XqRobotV2WalkFlat
VER=flat_walk_ppo_v1
cp -r backup/${TASK}/${VER}/{conf,src,shell,scene_flat.xml} /path/to/wheel_legged_RL_unilab/

# 启动训练
cd /path/to/wheel_legged_RL_unilab
CUDA_VISIBLE_DEVICES=1 bash shell/train_ppo_flat.sh
```

### 部分恢复

```bash
# 只恢复配置
cp backup/XqRobotV2WalkFlat/flat_walk_ppo_v1/conf/ppo/task/xqrobotV2_walk_flat/mujoco.yaml conf/ppo/task/xqrobotV2_walk_flat/

# 只恢复环境代码
cp backup/XqRobotV2WalkFlat/flat_walk_ppo_v1/src/unilab/envs/locomotion/xqrobotV2/*.py src/unilab/envs/locomotion/xqrobotV2/
```

## 创建新备份

```bash
TASK=XqRobotV2WalkFlat
VER=flat_walk_ppo_v2
RUN=2026-06-26_23-39-00
PROJ=/path/to/wheel_legged_RL_unilab

mkdir -p backup/${TASK}/${VER}/{conf/ppo/task/xqrobotV2_walk_flat,src/unilab/envs/locomotion/xqrobotV2,shell}

# 配置
cp ${PROJ}/conf/ppo/task/xqrobotV2_walk_flat/mujoco.yaml backup/${TASK}/${VER}/conf/ppo/task/xqrobotV2_walk_flat/

# 环境代码
cp ${PROJ}/src/unilab/envs/locomotion/xqrobotV2/{__init__.py,base.py,joystick.py} backup/${TASK}/${VER}/src/unilab/envs/locomotion/xqrobotV2/

# 脚本
cp ${PROJ}/shell/{train_ppo_flat.sh,eval_ppo_flat.sh,tensorboard.sh} backup/${TASK}/${VER}/shell/

# 场景
cp ${PROJ}/src/unilab/assets/robots/xqrobotV2/scene_flat.xml backup/${TASK}/${VER}/

# 运行时快照
cp ${PROJ}/logs/rsl_rl_ppo/${TASK}/${RUN}_mujoco/{run_config.json,run_summary.json} backup/${TASK}/${VER}/

# 版本信息
git -C ${PROJ} log --oneline -5 > backup/${TASK}/${VER}/git_commit.txt
git -C ${PROJ} diff HEAD > backup/${TASK}/${VER}/uncommitted.diff

# 编写 README.md（参考已有版本）
```
