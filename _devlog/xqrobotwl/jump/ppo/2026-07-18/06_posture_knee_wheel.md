# 06 跳跃姿态优化 + 站姿根因

**日期**: 2026-07-18
**来源**: 跳跃膝锁死+轮空转, 站立 z 偏低, 右腿偏置
**关联**: [05_stand_still_crouch_first](2026-07-17/05_stand_still_crouch_first.md)

---

## 问题描述

### 1. 跳跃膝锁死

```
step 30-140: knee = -0.89/+0.89 (铰在极限位), 
             jump_height=24 >> lean_forward=4.4 → 策略不在乎
```

### 2. 轮子空中狂转

```
R_wheel 从 0→21 rad/s (空中完全无用的加速)
```

### 3. 站立 z=0.46 偏低, 右腿不对称

## 根因分析

| 问题 | 根因 |
|------|------|
| 膝锁死 | `lean_forward` 膝罚线性(0.89) < jump_height(24)，策略无视 |
| 轮空转 | `wheel_air_time` 无轮速限制 |
| 站姿偏低 | DEFAULT_LEG_ANGLES 在 xqrobotwl 上非稳定平衡点——零动作下 robot 自然振荡 z: 0.37↔0.64 |
| 右腿偏置 | flat warm-start 继承——flat 策略学会"偏右蹲"保持稳定 |

## 解决方案

### 1. 膝罚平方化

```python
# linear → squared: 0.89 → 0.79
p_knee = clip(-knee_bend, 0, 1) ** 2
```

### 2. jump_height 膝门控

```python
knee_ok = (abs(knee_L) < 0.8) & (abs(knee_R) < 0.8)
# 膝锁死 = 高度奖励=0
```

### 3. wheel_air_time 轮转罚

```python
wheel_spin = sum(abs(wheel_vel)) * air
reward = (air * 0.5 - wheel_spin * 0.1)
```

### 4. 站姿增强

| 参数 | 旧 | 新 |
|------|------|------|
| `base_height` | -20 | **-60** |
| `leg_mirror` | -3 | **-12** |
| leg_mirror 实现 | 平方(L+R) | clip(asym-0.15,0,2) |

### 5. crouch_depth 补 roll_ok

之前只有 `crouch_prep` 检查 hip_roll，`crouch_depth` 漏了→已补。

## 修改文件

| 文件 | 改动 |
|------|------|
| `src/unilab/envs/locomotion/xqrobotwl/jump.py` | knee penalty→平方; jump_height 加 knee_ok; wheel_air_time 加轮转罚; crouch_depth 加 roll_ok; leg_mirror 改截断罚 |
| `conf/ppo/task/xqrobotwl_jump_flat/mujoco.yaml` | base_height 20→40→60; leg_mirror -3→-8→-12 |

## 评估结果(iter 9999)

| 指标 | 之前 | 现在 | 趋势 |
|------|------|------|------|
| 站立 z | 0.46 | 0.49 | ↑ |
| 站立存活 | ❌ 95步死 | ✅ 200步 |
| 右偏 | R_pitch=-0.48 | L_pitch=+0.38 | 方向纠正 |
| 跳跃 max_z | 1.44 | 1.40 | 持平 |
| 膝锁 | 30-140步 | 修复后待验证 | — |

## 后续计划

- [ ] 重训验证膝不锁、轮不转
- [ ] 站立 z 继续提升（DEFAULT_LEG_ANGLES 治本待做）

---

*记录人: AI (opencode)*
