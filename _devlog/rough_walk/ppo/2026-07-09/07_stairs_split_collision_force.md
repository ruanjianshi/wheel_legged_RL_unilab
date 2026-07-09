# 07 rough_walk 地形去台阶 + 碰撞力检测 + 分离楼梯特化训练

**日期**: 2026-07-09
**来源**: 参考 Tita RL，将楼梯训练独立为 NP3O 配置，rough_walk 回归多类型粗糙地形
**关联**: [06_tita_action_smooth_terrain_curriculum](2026-07-08/06_tita_action_smooth_terrain_curriculum.md)

---

## 问题描述

v4-v6 rough_walk 训练中加入了 stairs 特化地形（60% 楼梯），导致训练目标混乱——同一套策略既要学爬楼梯又要学粗糙地形。楼梯训练应独立为一个专项任务。

## 解决方案

### 1. rough.py 地形回归

移除 `pyramid_stairs` / `pyramid_stairs_inv`，地形回到无楼梯的多类型混合：

| 地形 | 比例 |
|------|------|
| random_rough | 40% |
| wave_terrain | 40% |
| hf_pyramid_slope | 10% |
| hf_pyramid_slope_inv | 10% |

斜坡范围放宽到 (0.1, 0.35)，随机粗糙降低到 (0.005, 0.04)，使斜坡更陡、粗糙更平缓。

### 2. 碰撞检测升级

从关节位置极限代理改为真实力传感器检测（对齐 Tita）：

```
xqrobotV2.xml: 新增 4 个力传感器
  left_thigh_force, left_calf_force, right_thigh_force, right_calf_force

joystick.py: _update_leg_forces() 每步读取传感器
joystick.py: _reward_collision() 用接触力 > 1N 判定触地
```

### 3. 楼梯训练独立为 NP3O 配置

```
conf/np3o/                          # 新建算法配置目录
  ├── config.yaml                   # NP3O 基础配置
  └── task/xqrobotV2_stairs/
      └── mujoco.yaml               # 楼梯专项任务

src/unilab/envs/locomotion/xqrobotV2/
  └── stairs.py                     # StairsOnlyTerrainCfg (100% 楼梯)

src/unilab/algos/torch/
  └── np3o.py                       # NP3O 算法实现

scripts/training/
  └── train_np3o.py                 # NP3O 训练入口

shell/
  ├── train_np3o_stairs.sh          # 楼梯训练脚本
  └── eval_np3o_stairs.sh           # 楼梯评估脚本
```

## 修改文件

| 文件 | 改动 |
|------|------|
| `src/unilab/envs/locomotion/xqrobotV2/rough.py` | 移除楼梯地形，斜坡范围调整 |
| `src/unilab/assets/robots/xqrobotV2/xqrobotV2.xml` | 添加 4 个腿力传感器 |
| `src/unilab/envs/locomotion/xqrobotV2/joystick.py` | `_update_leg_forces` + `_reward_collision` 力传感器版 |
| `conf/ppo/task/xqrobotV2_walk_rough/mujoco.yaml` | 注释更新 |
| `src/unilab/envs/locomotion/xqrobotV2/stairs.py` | 新建 |
| `conf/np3o/*.yaml` | 新建 |
| `src/unilab/algos/torch/np3o.py` | 新建 |
| `scripts/training/train_np3o.py` | 新建 |
| `shell/train_np3o_stairs.sh` `eval_np3o_stairs.sh` | 新建 |
| `scripts/play/play_interactive.py` | 注册 np3o algo |
| `tools/mujoco/show_stairs.py` | 新建 |

## 验证方法

1. `uv run tools/mujoco/show_terrain.py` — 确认无楼梯
2. `uv run tools/mujoco/show_stairs.py` — 确认纯楼梯
3. NP3O 训练启动成功 (iter 89, 0.78s/iter)

## 后续计划

- [ ] NP3O stairs 训练到 10000 iter 后评估
- [ ] rough_walk PPO 重新训练验证碰撞力检测效果

---

*记录人: AI (opencode)*
