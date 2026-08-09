# [07] 力衰减单位 bug — step_counter 是批次步, force_end 按 env 步算 → 力永不衰减

**日期**: 2026-08-07
**来源**: 重训 model_6000 评估矛盾 (训练 recover_complete 18-32% vs 无辅助评估 0%)
**关联**: [06_force_hcmd2_deadlock_fix](06_force_hcmd2_deadlock_fix.md)

---

## 问题描述

力目标解耦修复后重训, 训练曲线 recover_complete 一路涨到 18-32%
(批次显著达到站立里程碑), 但**禁用辅助力的评估始终 0% 恢复率**。
矛盾定位:
- 评估 env: 力关闭 (force_assist_enabled=false) → 机器人地面打滚, base_z 0.08-0.26m, 0% 站起
- 训练 env: 力"看似"衰减后仍 32% 站起
- 已排除: 观测组 (均 obs["obs"]), 动作缩放 (env 内部 apply_action), 采样噪声
  (确定性/随机均 0%), checkpoint 权重 (mlp.* 正确加载)

## 根因分析

**`_force_end_steps` 单位错误 (乘了 num_envs)**:

```python
# np_env.step() 每次所有 env 同步推进, step_counter 只 +1 (批次步, 非 env 步)
self._force_end_steps = force_end_iters * num_envs * 24   # ← env 步数
t_coeff = clip(1 - step_counter / _force_end_steps)       # ← step_counter 是批次步
```

- step_counter 每 iter 只 +24 (每 iter 24 次 step(), 每次推进全部 env)
- 但 force_end_steps 按 `iters × envs × 24` 算
- 需 `num_envs ×` (128×) 倍 iter 力才归零 → **训练全程力保持 ~98-100%**
- 训练 32% 恢复 = 满力 145N 抬起来的; 策略从未学会力独立
- 之前的"力撤除阵痛/反弹"曲线 (5000 塌缩→6000 反弹) 是策略漂移噪声, 非力衰减

**旁证**: constraint_value loss ~200-950 (力 cost 非零, critic 难预测);
若力为 0 该 loss 应≈0。

## 解决方案

```python
# 修正: step_counter 是批次步 (每 iter 共 24 次), 不乘 num_envs
self._force_end_steps = force_end_iters * 24
```

力在 `force_end_iters` 次迭代内线性衰减到 0。

## 修改文件

| 文件 | 修改内容 |
|------|---------|
| `src/unilab/envs/locomotion/xqrobotwl/fall_recovery.py:303-305` | `_force_end_steps = force_end_iters × 24` (去掉 `num_envs ×`), 注释说明单位 |

## 验证方法

模拟力衰减 (32 env, 按 iter 推进 step_counter):
- iter 0: F=145N (满) → iter 1000: 96N → iter 2000: 48N → iter 2900: 4.8N → iter 3000: **0N** ✓

## 后续计划

- [ ] 从 model_6000 (已学会带力站起) 续训, 力按正确机制 3000 iter 衰减 → 2000 iter 力独立
- [ ] model_9000 (力归零) / model_10000 / model_10999 无辅助评估
- [ ] 恢复成形 → 渲染视频 → 停止

## 关联日志

- [06_force_hcmd2_deadlock_fix](06_force_hcmd2_deadlock_fix.md) — 力目标解耦 (力衰减生效的前置条件)
