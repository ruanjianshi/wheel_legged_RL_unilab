# 05 站立姿态 + 先蹲后跳 + 按键优化

**日期**: 2026-07-17
**来源**: 站立不稳、不蹲就跳、按一次跳两次
**关联**: [04_keyboard_control](2026-07-16/04_keyboard_control.md)

---

## 问题描述

1. **无 trigger 时蹲着不站**：策略恒定输出大动作偏移（L_knee=+0.80），即使 trigger=0 也在推关节
2. **不蹲就跳**：trigger 激活后直接蹬地，跳过蓄力阶段
3. **按一次跳两次**：H 键窗口 150 帧 = 2.5 秒，够两轮完整跳跃

## 根因分析

| 问题 | 根因 |
|------|------|
| 站不稳 | 策略训练 99% 时间在蹲/跳，`action_mean` 有恒定偏移 |
| 不蹲就跳 | `vertical_thrust` 从 phase≥1 就开，无需等蹲完 |
| 跳两次 | 150 帧 ~0.8s/轮 × 2 轮，physics 动量延续 |

## 解决方案

### 1. `stand_still` 奖励

```python
# trigger ≤ 0.5 时惩罚大动作 → 强迫策略输出 ≈ 0
return -sum(action^2) * standing * 5.0
```

### 2. `vertical_thrust` 门槛 phase≥25

```
phase [1, 25]: 只有蹲姿奖励
phase ≥ 25:   蹲完才能蹬地 → 先蹲后跳
```

### 3. H 键窗口 150→80 帧

```python
# play_interactive.py
if commander.jump_frames >= 80:  # 0.8s, 刚好一轮
```

## 修改文件

| 文件 | 改动 |
|------|------|
| `src/unilab/envs/locomotion/xqrobotwl/jump.py` | 新增 `_reward_stand_still`；`vertical_thrust` phase ≥1→25；`crouch_prep` 窗口 [1,30]→[1,40] |
| `conf/ppo/task/xqrobotwl_jump_flat/mujoco.yaml` | `base_height` -30→-50；`stand_still`=5.0；`lean_forward` 3→10 |
| `scripts/play/play_interactive.py` | H 键 150→80 帧 |

## 后续计划

- [x] `stand_still=5.0` 过高 → ep_len=247 → 降至 **0.5**
- [x] `stand_posture` 替代 stand_still → 罚更重 → ep_len=31 → **删除**
- [ ] 简化：`base_height=-20` + `lean_forward=5`，不再加复杂罚项
- [ ] 站立姿态根因：DEFAULT_LEG_ANGLES 在 xqrobotwl 上不是稳定平衡点，零动作下 robot 自然振荡(z 0.37↔0.64)。策略学会降高度来换取稳定。治本需改默认姿势，不接受则当前方案最大努力。

---

*记录人: AI (opencode)*
