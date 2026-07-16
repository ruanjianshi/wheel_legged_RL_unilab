# 01 跳跃训练：phase-gated 奖励 + 多次迭代

**日期**: 2026-07-14
**来源**: jump 策略多次训练均"蹲而不跳"，需要系统化奖励设计
**关联**: [01_create_robot](../../../2026-07-12/01_create_robot.md), [01_fix_keyframe](../../../2026-07-13/01_fix_keyframe_and_termination.md)

---

## 问题描述

多轮训练（iter=9999）后策略表现为"只蹲不跳"：
- 蹲姿深度充足（base_z=0.33~0.47）
- 蹲住后不再延伸，从不跳起
- max_height 始终 ≤ 0.65m（初始站立高度）
- 左右腿不对称（R_hip 前倾 ~2× L_hip）

## 根因分析

### 1. 蹲姿奖励形成局部最优

```python
# 旧代码: 蹲姿奖励没有截止条件
crouch_prep: 只要 trigger>0.5 且蹲到位就一直给分
→ 策略学会"蹲住不动混分"，不冒险起跳
```

### 2. 蹬地奖励被窗口锁死

```python
# 旧代码: thrust 仅在 phase [15,45] 激活
vertical_thrust active ──但策略蹲到 phase 100+ 才开始尝试蹬地
→ 窗口已关闭，永远拿不到 thrust 奖励
```

### 3. 左右腿不对称

```
L_hip: +0.13 (前倾 13°)   R_hip: -0.25 (前倾 25°)
→ 右腿比左腿多 12° → 蹬地力不对称 → 旋转而非跳起
leg_mirror scale=-0.5 太小（罚分仅 0.07/步）
```

### 4. play_interactive.py J 键 bug

两处代码没排除 xqrobotwl jump 任务：
- `_init_commander`: line 892-893 无条件覆盖 col[4]=height_target
- 主循环: line 1273 只排除了 xqrobotV2_jump

## 解决方案

### 1. Phase-gated 奖励（核心架构）

```
trigger > 0.5 → jump_phase 计数器累加

Phase [1, 30]:  crouch 奖励 (蹲完 30 步后强制停)
Phase anytime:  thrust 奖励 (不锁窗, 随时蹬地给分)
Phase >= 1:     height/air 奖励 (全程开)
Phase >= 30:    landing 奖励
```

### 2. 奖励权重重新平衡

| 参数 | 旧 | 新 | 理由 |
|------|------|------|------|
| `crouch_prep` | 4→6→2 | 2 | 蹲着不值钱 |
| `crouch_depth` | 2→6→2 | 2 | 深蹲也不值钱 |
| `vertical_thrust` | 4→12 | **30** | 蹬地=蹲着 15 步 |
| `jump_height` | 4→12 | 12 | 高度鼓励不变 |
| `wheel_air_time` | 2→8→20 | 20 | 滞空高分 |
| `leg_mirror` | -0.5 | **-3.0** | 强制对称 |
| `action_scale` | 0.5→0.7 | **1.0** | 更大爆发 |
| `init_noise_std` | 1.0 | **0.5** | 精确控制 |
| `entropy_coef` | 0.01 | **0.005** | 减少噪声 |

### 3. 物理参数

| 参数 | 旧 | 新 |
|------|------|------|
| 腿关节 Kp | 30 | **60** |
| `crouch_height_target` | 0.40 | **0.35** |

### 4. play_interactive.py J 键修复

`_init_commander` 和主循环均添加 `xqrobotwl_jump` 排除条件。

## 修改文件

| 文件 | 改动 |
|------|------|
| `src/unilab/envs/locomotion/xqrobotwl/jump.py` | Phase-gated 全部 6 个跳跃奖励 + jump_phase 计数器 |
| `conf/ppo/task/xqrobotwl_jump_flat/mujoco.yaml` | 奖励权重/action_scale/crouch_target 全部调整 |
| `src/unilab/assets/robots/xqrobotwl/xqrobotwl.xml` | 腿关节 Kp 30→60 |
| `scripts/play/play_interactive.py:1273` | 排除 xqrobotwl_jump |
| `scripts/play/play_interactive.py:892-893` | 同样排除 |

## 训练迭代历史

| Run | iter | 主要改动 | wheel_air_time | 问题 |
|------|------|------|------|------|
| 2026-07-13_14-20-29 | 9999 | 初版 | 0.005 | 不会跳 |
| 2026-07-13_18-42-13 | 9999 | Kp=60,+thrust,+depth | 0.052 | 会蹬但不离地 |
| 2026-07-14_10-53-03 | 9999 | 奖励放大×3,+噪声↓ | 3.43 | **会跳了**（训练） |
| 2026-07-14_17-31-01 | 9999 | phase-gated + crouch↓+thrust↑ | 2.01 | 蹲而不跳（phase 窗口锁死） |

## 验证方法

- jump_height 应持续增长到 > 2.0
- wheel_air_time 应持续增长到 > 5.0
- max base_height 应超过 0.7m
- 左右腿对称：hip_L+R≈0, knee_L+R≈0

## 后续计划

- [ ] 重训 jump（当前改动已就绪）
- [ ] 评估起跳效果
- [ ] 如需：添加 SLIP/SRBM 参考轨迹（参考 Cassie 论文）
- [ ] 如需：增加 `jump_impulse` 奖励（瞬时加速度驱动爆发）

---

*记录人: AI (opencode)*
