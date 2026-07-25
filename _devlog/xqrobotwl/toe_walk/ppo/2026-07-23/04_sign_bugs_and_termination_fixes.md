# 04 — 点足行走：5 个致命 bug 修复 + 2 个终止阈值修正

## 日期

2026-07-23

## 来源

点足相位门控训练 iter 8935：`phase_swing_lift=0.003, phase_knee_lift=0.000` — 策略学会了速度跟踪但完全不会抬腿。

## 问题描述

发现 5 个符号/逻辑 bug + 2 个终止阈值错误，全部互相影响：

### Bug 1: swing_mask 在相位门控模式永远=0
`_compute_ref_dof_pos` 中 `self._swing_mask = np.zeros((self._num_envs,))`，导致 `swing_contact_penalty` 永远不罚。修复：从 `sin(phase)` 生成 swing_mask。

### Bug 2-4: 三个奖励符号反了
| 奖励 | weight | 函数返回 | 乘积 | 效果 |
|------|--------|----------|------|------|
| swing_contact_penalty | -25 | `-contact*swing` (负) | +25×接触 | **奖励**轮子着地 |
| feet_regulation | -2 | `-slip` (负) | +2×滑移 | **奖励**打滑 |
| soft_landing | -1 | `-|vz|` (负) | +1×硬着陆 | **奖励**砸地 |

修复：全部 weight 改为正值。

### Bug 5: _reward_leg_mirror 用减法而非加法
thigh = `dof[:,1] - dof[:,4]`（应为 +），calf 同理。默认对称位姿下恒定基线罚 0.09/step。修改为加法。

### 终止阈值 1: calf_extreme > 0.85 rad
点足行走膝必须弯到 1.0+ rad。阈值 0.85 意味着策略一弯腿就死 → 永远不弯。修复：0.85→2.0。

### 终止阈值 2: contact_body_names 0.1N
calf link 轻轻蹭地就触发 termination。修复：0.1N→5.0N。

## 根因分析

```
致命链：
  弯膝 → calf>0.85 → 终止 → 策略不弯膝 → phase_knee_lift=0
  轮子着地 → +25 奖励 → 策略主动着地 → phase_swing_lift=0
  滑移 → +2 奖励 → 策略主动打滑 → feet_regulation 正反馈
```

## 解决方案

1. `swing_mask`: 从 sin(phase) 生成（|sin|>0.2=摆地）
2. sign fix: swing_contact_penalty -25→25, feet_regulation -2→2, soft_landing -1→1
3. `leg_mirror`: thigh/calf 减→加
4. `calf_extreme`: 0.85→2.0
5. `contact_body_names`: 0.1N→5.0N
6. `max_tilt_deg`: 45→60（给抬腿不平衡更多容错）
7. 添加课程 ramp：swing_contact_penalty 在前 3000 iter 用 0.1x 强度

## 修改文件

| 文件 | 行 | 改动 |
|------|-----|------|
| `toe_walk.py` | 332-336 | swing_mask 从 sin(phase) 生成 |
| `toe_walk.py` | 86-92 | swing_contact_penalty 左右分离 gating |
| `toe_walk.py` | 95-110 | feet_regulation 只罚支撑侧 |
| `toe_walk.py` | 292-296 | leg_mirror 减→加 |
| `toe_walk.py` | 484-485 | calf: 0.85→2.0, thigh: 0.02→-0.5/0.5 |
| `toe_walk.py` | 494 | contact_body threshold: 0.1→5.0 |
| `toe_walk.py` | 495-512 | curriculum 三段式 ramp |
| `mujoco.yaml` | 90-92 | 三个 sign bug 修复 |
| `mujoco.yaml` | 97 | max_tilt_deg: 45→60 |

## 后续计划

- 修复后训练 iter 2815 仍有 ep_len 锁死 ~10 的问题 → 见 [05](2026-07-24)
- 需要更大的奖励结构调整 → 见 [12](2026-07-25)

## 关联日志

- [03_phase_gated](2026-07-22/03_phase_gated.md) — 前序设计
