# 05 启用域随机化 + 腿长自适应

**日期**: 2026-07-06
**来源**: 评估 ckpt=4999 — Vx 方向已修但跟踪偏弱，Vx RMSE 改善 50% 但仍需更多训练
**关联**: [_devlog/rough_walk/ppo/2026-07-04/04_fix_forward_tracking.md](../2026-07-04/04_fix_forward_tracking.md)

---

## 问题描述

v2 训练（ckpt=4999）Vx 方向修正成功，但前向速度跟踪仍非常弱（命令 0.3 m/s 仅跟踪 ~0.03 m/s）。进一步代码审查发现，**XqRobotV2 的整个域随机化系统完全未生效**。

## 根因分析

### Bug 1: 域随机化完全被绕过

`XqRobotDRProvider.build_reset_plan()` 中硬编码 `randomization = None`（joystick.py:L181），跳过了 `build_common_reset_randomization()` 调用。这意味着所有 per-episode DR（base_mass, ground_friction, kp, kd, com）**完全不生效**。

父类 `LocomotionDRProvider.build_reset_plan()` 正确调用：
```python
randomization=build_common_reset_randomization(env, num_reset, ...)
```

但 XqRobotDRProvider 重写整个方法时没有带上这个调用。

### Bug 2: 腿长随机化从未实现

- `XqRobotDomainRandConfig.randomize_leg_length` 默认为 `False`
- 即使手动设为 `True`，Provider 中 `_LEG_GEOM_NAMES` 从未定义
- `joystick.py:L261` 的 `getattr(XqRobotDRProvider, "_LEG_GEOM_NAMES", [])` 永远返回 `[]`
- 腿长缩放逻辑完全为空壳

### 影响

XqRobotV2 环境没有任何域随机化 → 策略从未接触过质量/摩擦/PD增益/质心变化 → 在复杂地形上泛化能力弱 → Vx/Vy 跟踪弱。

## 解决方案

### 修改 1: joystick.py — 启用 per-episode DR

```
src/unilab/envs/locomotion/xqrobotV2/joystick.py:L170-210
```
- `build_reset_plan()`: `randomization = None` → `build_common_reset_randomization(env, num_reset)`
- 新增导入: `build_common_reset_randomization`, `GeomSizeOverride`, `InitRandomizationPlan`, `ModelVariantSpec`

### 修改 2: joystick.py — 实现腿长自适应

```
src/unilab/envs/locomotion/xqrobotV2/joystick.py:L178-L197
```
- 新增 `_LEG_GEOM_NAMES` 类属性，包含左右腿 12 个碰撞/视觉几何体名称
- 新增 `build_init_randomization_plan()` 方法，在 env 初始化时为全部 env 统一随机腿长缩放
- 缩放因子从 `leg_length_scale_range` 中均匀采样，应用到所有腿几何体上
- 每次训练启动时不同 run 获得不同腿长，形成跨 run 的归纳多样性

### 修改 3: mujoco.yaml — 启用所有 DR + 延长训练

```
conf/ppo/task/xqrobotV2_walk_rough/mujoco.yaml
```
- `max_iterations`: 5000 → **20000**（DR 增加训练难度，需 4x 迭代）
- `randomize_base_mass`: false → **true**（基座质量 +-1.5kg）
- `randomize_ground_friction`: false → **true**（地面摩擦 0.2-1.6x）
- `randomize_kp`: false → **true**（PD 增益 0.8-1.2x）
- `randomize_kd`: false → **true**（PD 阻尼 0.8-1.2x）
- `random_com`: false → **true**（质心偏移 +-3cm）
- `leg_length_scale_range`: [0.8, 1.2] → **[0.85, 1.15]**（略收窄首次 DR 训练的范围）

## 修改文件

| 文件 | 行号 | 改动 |
|------|------|------|
| `src/unilab/envs/locomotion/xqrobotV2/joystick.py` | L19-27 | 新增 DR 类型导入 |
| `src/unilab/envs/locomotion/xqrobotV2/joystick.py` | L171-177 | 新增 `_LEG_GEOM_NAMES` |
| `src/unilab/envs/locomotion/xqrobotV2/joystick.py` | L179-197 | 新增 `build_init_randomization_plan()` |
| `src/unilab/envs/locomotion/xqrobotV2/joystick.py` | L200-229 | 修复 `build_reset_plan()` 启用 DR |
| `conf/ppo/task/xqrobotV2_walk_rough/mujoco.yaml` | L24 | max_iterations=20000 |
| `conf/ppo/task/xqrobotV2_walk_rough/mujoco.yaml` | L65-71 | 启用全部 DR |

## 验证方法

1. Ruff format + check 通过
2. 重训 rough_walk PPO @ 20000 iter
3. 每 5000 iter 评估 decoupling suite，目标：
   - Vx RMSE < 0.20（当前 0.16-0.55）
   - avg_vx 与指令符号一致，幅度 > 50%
   - 后退跟踪恢复

## 后续计划

- [x] 启动 v3 训练 @ 20000 iter
- [ ] iter=5000/10000/15000 阶段性评估
- [ ] iter=20000 跑 full suite 全面评估

---

*记录人: AI (opencode)*
