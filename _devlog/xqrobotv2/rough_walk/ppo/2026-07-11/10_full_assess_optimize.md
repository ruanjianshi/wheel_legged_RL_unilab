# 10 全量评估三个任务 + 针对性优化

**日期**: 2026-07-11
**来源**: v6 rough_walk / v2 stairs NP3O / v2 jump 全量评估 (16场景)
**关联**: [08_fix_spin_decouple](../2026-07-10/08_fix_spin_decouple.md), [09_vy_zero_all](2026-07-11/09_vy_zero_all.md)

---

## 评估结果

### rough_walk (v6, iter=4999)

| 指令 | avg_vx | Vx RMSE | 判定 |
|------|:------|:------|:--:|
| vx=0.1 | +0.091 | 0.113 | ✅ 方向正确 |
| vx=0.3 | +0.070 | 0.266 | ⚠️ 跟踪弱 (23%) |
| vx=0.6 | +0.081 | 0.534 | ❌ 跟踪极弱 (13%) |
| vx=-0.3 | +0.068 | 0.386 | ❌ 后退向前走 |

### stairs NP3O (v2, iter=9999)

所有指令 avg_vx = -0.23，恒向后走。NP3O 训练未生效。

### jump (v2, iter=4999)

| 指令 | avg_vx | Vx RMSE | 判定 |
|------|:------|:------|:--:|
| vx=0.3 | -0.054 | 0.355 | ⚠️ 微后退 |

---

## 根因分析

### rough_walk: 地形全粗糙无速度练习平台

v4 (avg_vx=0.18) 有 60% 楼梯地形，楼梯的 3m 平台为策略提供了"安全区"练习 Vx 跟踪。v6 去楼梯后，地形为 40% 随机粗糙 + 40% 波浪 + 20% 斜坡——全线粗糙，策略始终处于"勉强站稳"状态，没有余力加速。

### stairs: NP3O viol_loss 过重

`cost_viol_loss_coef=0.1`, K 在 iter~3000 饱和到 1.0 → viol_loss 主导总损失 → PPO 的 tracking 信号被淹没 → 策略退化到最低违规行为（向后蠕行）。

### jump: tracking_lin_vel 权重太低

`tracking_lin_vel=1.0` 被跳跃专属奖励（jump_height=4.0, landing_soft=2.0, wheel_air_time=1.0）压制。策略优先学跳跃，速度跟踪附带。

---

## 优化方案

| 任务 | 优化 | 旧 | 新 |
|------|------|------|------|
| rough_walk | 加 20% 平地 | flat=0% | **flat=20%** |
| rough_walk | 斜坡减半 | slope 10%×2 | **slope 5%×2** |
| stairs | viol_loss 降 10x | 0.1 | **0.01** |
| jump | tracking 升 2x | 1.0 | **2.0** |

---

## 修改文件

| 文件 | 改动 |
|------|------|
| `src/unilab/envs/locomotion/xqrobotV2/rough.py:L74-115` | 加 20% 平地，斜坡减至 5% |
| `conf/np3o/config.yaml:L66` | cost_viol_loss_coef 0.1→0.01 |
| `conf/ppo/task/xqrobotV2_jump_flat/mujoco.yaml:L86` | tracking_lin_vel 1.0→2.0 |

---

## 后续计划

- [ ] 三个任务重训
- [ ] rough_walk 评估 Vx 跟踪应 > 0.15 (v4 水平)
- [ ] stairs 评估方向应修正
- [ ] jump 评估后退应消除

---

*记录人: AI (opencode)*
