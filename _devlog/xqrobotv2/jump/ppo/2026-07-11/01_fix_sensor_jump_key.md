# 01 修复跳跃传感器名称 + J 键跳越持续化

**日期**: 2026-07-11
**来源**: 代码审查 + 验证跳跃键盘控制
**关联**: [Tita RL 参考](../../../docs/references/TitaRL.md)

---

## 问题描述

### 问题 1: wheel_air_time 恒为零

jump 训练中 `reward/wheel_air_time` 恒为 0.5（常数），策略收不到真实的离地反馈。

### 问题 2: J 键一跳即止

键盘按 J 后 jump_trigger 仅持续 1 帧（0.01s），训练时 jump_trigger 持续 4s。策略来不及完成下蹲→爆发→腾空→落地。

## 根因分析

### 传感器名称拼写错误

```python
# jump.py:201 — 错误
left = self._backend.get_sensor_data("left_link_wheel_force")

# xqrobotV2.xml:129 — 实际
<force name="left_wheel_force" site="left_wheel_site" />
```

`left_link_wheel_force` 不存在，每步 KeyError 被 silent catch → `wheel_contact` 恒为零。

### J 键一帧清零

```python
# play_interactive.py:1278 — 旧代码
commander.jump_trigger = 0  # 使用一次立即清零
```

训练时 `resampling_time=4s`，jump_trigger 持续 400 步。评估时仅 1 步。

## 解决方案

### 传感器名修复

`left_link_wheel_force` → `left_wheel_force`，右侧同理。

### J 键持续化

jump_trigger 持续 150 帧（1.5s），接近训练时长。

## 修改文件

| 文件 | 修改内容 |
|------|---------|
| `src/unilab/envs/locomotion/xqrobotV2/jump.py:L201-202` | 传感器名修复 |
| `scripts/play/play_interactive.py:L1275-1281` | J 键 150 帧持续 |

## 验证方法

重训后键盘 J 键应有完整的下蹲→起跳→落地动作。

## 后续计划

- [x] v2 重训启动 (iter ~0，运行中)
- [ ] 训练完成键盘验证

---

*记录人: AI (opencode)*
