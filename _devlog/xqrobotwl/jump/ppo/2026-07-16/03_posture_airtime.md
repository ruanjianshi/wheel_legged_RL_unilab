# 03 跳跃训练：姿态修正 + 腾空优化

**日期**: 2026-07-16
**来源**: 策略会跳但姿态有缺陷：叉腿、腾空蹬腿、键盘 J 键无效
**关联**: [02_kp_survival_posture](2026-07-15/02_kp_survival_posture.md)

---

## 问题描述

### 问题 1: 叉腿下蹲

```
无 trigger 时 R_roll=-0.29（外展 3×），靠叉腿降 base 混 crouch 分。
姿态不对称 → 跳不起来。
```

### 问题 2: 腾空后继续蹬腿

```
vertical_thrust 只在 phase≥1 时奖励 vz>0，
不区分地上/空中 → 策略在飞行中也拼命伸腿。
```

### 问题 3: 键盘 J 键无效

```
play_interactive.py 创建 env 后 curriculum 从 0 起步，
commands[:,4] *= 0 → J 键设的 1.0 每步被归零。
```

## 根因分析

| 问题 | 根因 | 影响 |
|------|------|------|
| 叉腿 | `crouch_prep`/`lean_forward` 不检查 hip_roll | 外展降 base 拿分 |
| 腾空蹬腿 | `vertical_thrust` 无轮触地门控 | 空中白白浪费力 |
| J 键无效 | `_total_env_steps=0` → curriculum=0 → 归零 jump_trigger | 键盘操控不跳 |

## 解决方案

### 1. hip_roll 约束

`lean_forward` 和 `crouch_prep` posture 增加 hip_roll 检查：

```python
roll_ok = (|L_roll - 0.1| < 0.2) & (|R_roll + 0.1| < 0.2)
# 限制 roll 在默认 ±0.2 内
```

### 2. vertical_thrust 地上门控

```python
on_ground = (max(wheel_contact, axis=1) > 0.5)
active = (phase >= 1) & (vz > 0) & on_ground
# 腾空 → on_ground=0 → thrust 停 → 空中不蹬腿
```

### 3. play 模式课程绕过

`_init_commander` 中加：

```python
if hasattr(env, "_jump_curriculum_end"):
    env._total_env_steps = env._jump_curriculum_end + 1
```

### 4. crouch_height_target 回调

`0.35 → 0.40`：不需要蹲那么深，减少叉腿动机。

## 修改文件

| 文件 | 改动 |
|------|------|
| `src/unilab/envs/locomotion/xqrobotwl/jump.py` | `lean_forward`: 加 hip_roll 罚；`crouch_prep`: posture 加 roll_ok；`vertical_thrust`: 加 on_ground 门控；`crouch_height_target`: 0.35→0.40 |
| `conf/ppo/task/xqrobotwl_jump_flat/mujoco.yaml` | `crouch_height_target`: 0.35→0.40 |
| `scripts/play/play_interactive.py` | `_init_commander`: 绕过 jump curriculum |

## 验证结果 (iter 9999)

| 指标 | 修复前 | 修复后 |
|------|------|------|
| `max_z` | 0.98m | **1.03m** |
| `jump_mean` | 0.78m | **0.81m** |
| `air_steps` | 293/500 | **360/500** |
| hip_roll 外展 | R=-0.29 ❌ | ±0.02 ✅ |
| 蹲姿高度 | 0.35m | 0.40m |

## 后续计划

- [ ] 重训验证空中不蹬腿
- [ ] 键盘 J 键验证跳跃

---

*记录人: AI (opencode)*
