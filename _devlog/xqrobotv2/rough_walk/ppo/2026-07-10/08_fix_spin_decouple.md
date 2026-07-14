# 08 修复转圈退化：碰撞回退 + 命令解耦

**日期**: 2026-07-10
**来源**: v5 rough_walk iter=4999 评估发现所有 Vx 指令 avg_vx=-0.13，恒向后走
**关联**: [07_stairs_split_collision_force](../2026-07-09/07_stairs_split_collision_force.md)

---

## 问题描述

v5 rough_walk (iter=4999) decoupling 评估：所有指令 avg_vx ≈ -0.13，Vx RMSE 0.44-0.74，base_h 仅 0.61。

## 根因分析

### 原因 1: 力传感器碰撞阈值 1N 太低

`xqrobotV2.xml` 的 `<force>` sensor 测量 site 处的所有作用力（关节反力 + 惯性力 + 接触力）。1N ≈ 0.1kg 力——正常腿加速运动即误判。

训练指标 `collision: -0.84`→ 84% 步数被判碰撞 → 策略学到"不动腿，纯轮子转圈"避罚。

### 原因 2: Vy=[0,0] 后解耦产生零速度

`vel_limit` 中 Vy 归零后，命令解耦代码 50% 选 axis=1 → Vx=0 → 零速度命令：

```python
axis = np.random.choice([0, 1])
if axis == 1: cmds[i, 0] = 0  # ❌ 零速度 → tracking 满分
```

策略收到零速度命令时 tracking_lin_vel = 3.0（满分），学会"不动最优"。

## 解决方案

### 碰撞检测回退关节极限代理

`_reward_collision`: 力传感器(1N) → 关节角度阈值：
```python
thigh_margin = max(0, 0.12 - dof_pos[1]) + max(0, 0.12 - dof_pos[4])
calf_margin  = max(0, |dof_pos[2]|-0.75) + max(0, |dof_pos[5]|-0.75)
```
v4 验证有效 (avg_vx=+0.18)。

### 命令解耦检测 Vy 范围

`_sample_commands` + `_update_commands`: `high[1]-low[1] < 1e-6` 时固定 axis=0。

## 修改文件

| 文件 | 修改内容 |
|------|---------|
| `src/unilab/envs/locomotion/xqrobotV2/joystick.py:L146-159` | `_reward_collision` 回退关节极限 |
| `src/unilab/envs/locomotion/xqrobotV2/joystick.py:L248-256` | `_sample_commands` Vy 范围检测 |
| `src/unilab/envs/locomotion/xqrobotV2/joystick.py:L621-628` | `_update_commands` Vy 范围检测 |
| `src/unilab/envs/locomotion/xqrobotV2/rough.py:L136-142` | `_sample_commands` Vy 范围检测 |

## 验证方法

v6 重训 + decoupling 评估：avg_vx > 0 且 RMSE < 0.3

## 后续计划

- [x] v6 重训启动 (iter ~900，运行中)
- [ ] 训练完成评估

---

*记录人: AI (opencode)*
