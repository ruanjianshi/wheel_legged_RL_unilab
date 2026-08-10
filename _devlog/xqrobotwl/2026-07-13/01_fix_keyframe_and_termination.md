# 01 修复 keyframe 默认角度 + 终止条件

**日期**: 2026-07-13
**来源**: 重训后姿态仍异常，排查发现 keyframe 未同步 + 终止条件误杀
**关联**: [01_create_robot](../2026-07-12/01_create_robot.md)

---

## 问题描述

1. **姿态"一前一后"**：修正 `base.py` DEFAULT_LEG_ANGLES 后重训，reset 时右腿仍然向后踢
2. **episode_length=1**：修正 keyframe 后立即触发终止，每帧跌倒

## 根因分析

### Bug 1：keyframe 与 DEFAULT_ANGLE 不同步

`_devlog/xqrobotwl/ppo/2026-07-12/01_create_robot.md` 里改了 `base.py` 的 `DEFAULT_LEG_ANGLES`，但 ignorred 了 keyframe XML。reset 时 `_init_qpos` 来自 keyframe，所以 Python 常量的修改从未生效。

| 层 | 原值 | 修改后 |
|------|------|--------|
| `base.py` DEFAULT_LEG_ANGLES | `[0.1,0.15,0.15,-0.1,-0.15,-0.15]` | 已改 ✅ |
| `locomotion_task.xml` keyframe | `0.1,0.2,-0.2, -0.1,0.2,-0.2` | **未改** ❌ |
| `scene_flat.xml` keyframe | `0.1,0.2,-0.2, -0.1,0.2,-0.2` | **未改** ❌ |

旧 keyframe 中 `R_hip_pitch=+0.2` 在 axis `0 -1 0` 上 = 向后踢腿。

### Bug 2：终止条件未适配翻转轴

`_compute_terminated` 沿用 xqrobotV2 的阈值：

```python
thigh_collapsed = (dof_pos[:, 1] < 0.02) | (dof_pos[:, 4] < 0.02)
```

xqrobotV2 里所有 hip_pitch 轴均为 `+Y`，用 `< 0.02` 检测"腿过直坍塌"。
xqrobotwl 里 R_hip_pitch 轴为 `-Y`，默认值 `-0.15`，**永远 `< 0.02`** → 每秒终止。

在 `-Y` 轴上"向 0 靠近"意味着从负侧升上来 → 应检测 `> -0.02`。

## 解决方案

### 1. 同步 keyframe XML

`locomotion_task.xml` 和 `scene_flat.xml` 的 qpos 改为：

```
0.1 0.15 0.15 0
-0.1 -0.15 -0.15 0
```

与 `base.py` 的 `DEFAULT_LEG_ANGLES` 一致。

### 2. 修正右 hip_pitch 终止阈值

4 个文件的 `_compute_terminated` 中 `dof_pos[:, 4]` 检测翻转为 `> -0.02`。

## 修改文件

| 文件 | 修改内容 |
|------|---------|
| `src/unilab/assets/robots/xqrobotwl/locomotion_task.xml` | keyframe qpos: `0.2,-0.2` → `0.15,0.15`（左右对称符号） |
| `src/unilab/assets/robots/xqrobotwl/scene_flat.xml` | 同上 |
| `src/unilab/envs/locomotion/xqrobotwl/joystick.py:404` | `dof_pos[:,4] < 0.02` → `dof_pos[:,4] > -0.02` |
| `src/unilab/envs/locomotion/xqrobotwl/rough.py:262` | 同上 |
| `src/unilab/envs/locomotion/xqrobotwl/jump.py:223` | 同上 |
| `src/unilab/envs/locomotion/xqrobotwl/toe_walk.py:351` | 同上 |

## 验证方法

- 重训 flat/rough/stairs/jump 四任务
- 观察 episode_length 应正常增长（≥ 24 steps）
- base_height 应稳定在 0.65±0.05

## 后续计划

- [ ] 四任务重训
- [ ] 训练收敛后全量评估

---

*记录人: AI (opencode)*
