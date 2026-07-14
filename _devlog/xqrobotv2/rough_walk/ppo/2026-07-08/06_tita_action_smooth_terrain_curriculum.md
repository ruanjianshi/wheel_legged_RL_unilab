# 06 移植 Tita RL 关键机制：动作平滑 + 地形课程 + 观测归一化

**日期**: 2026-07-08
**来源**: 参考 `tita_rl` 项目，识别出 3 个高可行性改进
**关联**: [05_enable_dr_leg_adaptive](../2026-07-06/05_enable_dr_leg_adaptive.md)

---

## 问题描述

v3 训练 (DR + 20000 iter) Vx 方向已修正，但前向速度跟踪仍然偏弱。Tita RL 项目在竞品分析中发现几个可移植机制。

## 改进方案

### 改进 1: 动作平滑 (Low-Pass Filter)

**Tita RL 做法**: `filtered = 0.8 * new_action + 0.2 * prev_action`，防止策略输出高频振动，减少 PD 力矩抖动。

**实现**:
```
src/unilab/envs/locomotion/xqrobotV2/base.py:48        # XqRobotControlConfig 新增 action_smoothing 字段
src/unilab/envs/locomotion/xqrobotV2/joystick.py:341-346  # apply_action 中新增 EMA 滤波
conf/ppo/task/xqrobotV2_walk_rough/mujoco.yaml:50       # action_smoothing: 0.2
```

### 改进 2: 地形课程学习

**Tita RL 做法**: "游戏启发"式课程 — 机器人走得远 → 晋升到更难地形；走得近 → 降级到更简单地形。`promote_frac=0.5` (走超过半个地形长度升级)，`demote_frac=0.25` (走不到四分之一降级)。

**实现**:
```
conf/ppo/task/xqrobotV2_walk_rough/mujoco.yaml:59-63    # terrain_curriculum: enabled=true, promote/demote/cycle
src/unilab/envs/locomotion/xqrobotV2/joystick.py:206-208 # XqRobotDRProvider 新增 record_episode_start 调用
src/unilab/envs/locomotion/xqrobotV2/rough.py:180-186    # XqRobotV2WalkRoughEnv 新增 update_state 调用 update_on_done
src/unilab/envs/locomotion/xqrobotV2/rough.py:66-117    # 地形栅格 num_rows: 6→8 (8 个难度级别)
```

### 改进 3: 观测归一化

**Tita RL 做法**: RunningMeanStd (Welford 算法) 在线归一化观测，提升 DR 下泛化。

**实现**:
```
conf/ppo/task/xqrobotV2_walk_rough/mujoco.yaml:25       # empirical_normalization: false→true
```

### 改进 4: 地形组合优化 (Tita 风格)

**参考 Tita RL 思路**: 对轮腿机器人，楼梯是最高难度地形，其他类型权重降低。

| 地形 | 旧 | 新 |
|------|:--:|:--:|
| stairs | 15% | **35%** |
| stairs_inv | 15% | **25%** |
| random_rough | 30% | **20%** |
| wave | 30% | **10%** |
| slope | 5% | 5% |
| slope_inv | 5% | 5% |
| flat | 0% | 0% |

台阶高度范围从 [0.02, 0.08] 调整到 [0.03, 0.10] — 配合 8 级课程，高难度地形有更高台阶。

## 修改文件

| 文件 | 改动 |
|------|------|
| `src/unilab/envs/locomotion/xqrobotV2/base.py:L48` | 新增 `action_smoothing: float = 0.0` |
| `src/unilab/envs/locomotion/xqrobotV2/joystick.py:L341-346` | `apply_action` 新增 EMA 动作平滑 |
| `src/unilab/envs/locomotion/xqrobotV2/joystick.py:L206-208` | `build_reset_plan` 新增 `record_episode_start` |
| `src/unilab/envs/locomotion/xqrobotV2/rough.py:L180-186` | 新增 `update_state` override (terrain update_on_done) |
| `src/unilab/envs/locomotion/xqrobotV2/rough.py:L66-117` | 地形配置: stairs 权重比例 + num_rows=8 + 台阶高度提升 |
| `conf/ppo/task/xqrobotV2_walk_rough/mujoco.yaml` | action_smoothing=0.2, terrain_curriculum=enabled, empirical_normalization=true |

## 验证方法

1. Ruff format + lint 通过
2. 训练启动成功 (iter=16, 2.28s/iter, ETA~12.8h)
3. 预期: 动作更平滑、地形课程使策略渐进适应、观测归一化提升 DR 泛化

## 后续计划

- [x] 启动 v4 训练 @ 20000 iter
- [ ] iter=5000/10000/15000 阶段性评估
- [ ] 与 v3 对比: Vx 跟踪是否改善

---

*记录人: AI (opencode)*
