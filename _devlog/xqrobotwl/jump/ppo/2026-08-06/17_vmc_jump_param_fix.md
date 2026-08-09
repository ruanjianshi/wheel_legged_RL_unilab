# 17 修复 VMC 版本无法起跳: feedforward 不足 + L0 动作范围过小

**日期**: 2026-08-06
**来源**: 四算法训练监控 — VMC/SRLVMC 训练 1000+ iter 只会平衡不跳(VMC thrust 卡 0.6)
**关联**: [16_ppo_vmc_srl_vmc_algorithms](2026-08-06/16_ppo_vmc_srl_vmc_algorithms.md)

---

## 问题描述

四算法对比训练中,纯PPO/SRL 在 iter 1000 已能跳起(0.55-0.65m),但 **VMC/SRLVMC 在 iter 1000 只学会平衡站立(jump_height≈0),不跳**:
- VMC 训练 thrust 奖励从 iter 883 到 1498 卡在 **0.6 不变**
- 开环 FSM 测试(零策略动作)机器人振荡 0.31-0.65m,从不超站立

## 根因分析

1. **feedforward_force=80N 不足以支撑机器人**:xqrobotwl 总重 ~18.65kg,单腿需 ~91.5N 支撑。80N/腿 < 91.5N → 站立时下沉;蹬伸到顶时 force = kp×(L0_ref−L0) + 80 ≈ 80N,仍 < 91.5N → **无法离地**。参考项目(Wheel-Legged-Lab)机器人 12.28kg 用 ff=40-60N 足够,按比例放大后 xqrobotwl 需要 ~110N。
2. **action_scale_l0=0.05 动作范围太小**:L0_ref = action×0.05 + 0.367,动作 ±1 只能覆盖 [0.317, 0.417],而跳跃需要下蹲到 ~0.28、蹬伸到 ~0.50 → **策略物理上够不到**。
3. 排查确认膝关节 30 N·m 电机**不是**瓶颈(提高到 ±60 开环结果不变)。

## 解决方案

| 参数 | 旧 | 新 | 理由 |
|------|-----|-----|------|
| `feedforward_force` | 80 | **110** | > 体重/腿(~91N), 保证蹬伸能离地 |
| `action_scale_l0` | 0.05 | **0.12** | 覆盖下蹲 0.28 ~ 蹬伸 0.50 范围 |

## 修改文件

| 文件 | 修改内容 |
|------|---------|
| `src/unilab/envs/locomotion/xqrobotwl/vmc.py` | feedforward_force 80→110, action_scale_l0 0.05→0.12 |
| `conf/ppo/task/xqrobotwl_jump_vmc_flat/mujoco.yaml` | 同上 |
| `conf/ppo/task/xqrobotwl_jump_srl_vmc_flat/mujoco.yaml` | 同上 |

## 验证方法

开环 FSM 驱动测试(SRLVMC 零策略动作 + 保持 trigger):
- 修复前: max_z=0.652, 无起跳
- 修复后: **站立 z≈0.526 → max_z=0.659, 起跳 0.133m** ✅

已重启 VMC/SRLVMC 训练(纯PPO/SRL 继续运行不受影响)。

## 后续计划

- [ ] 监控 VMC/SRLVMC 重启后是否在 iter 1000-2000 学会跳
- [ ] 若 SRLVMC 跳高仍低,考虑 FSM 参考主导(参考项目模式: `final = reference + residual_scale×policy`),当前是 `reference×gain + policy`
- [ ] 训练完成后四算法评估对比

## 关联日志

- [16_ppo_vmc_srl_vmc_algorithms](2026-08-06/16_ppo_vmc_srl_vmc_algorithms.md)
