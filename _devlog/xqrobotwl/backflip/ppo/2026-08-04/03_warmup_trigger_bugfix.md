# 03 — 修复预热斜坡累乘 bug + 站立态 flip_progress 重置

## 日期

2026-08-04

## 来源

快速验证 200 iter 训练 (P2 环境): best_mean_reward 26.3, 但**所有翻转奖励全程 0** (flip_progress/flip_complete/launch_thrust), 只有 posture_stand 在涨, mean_episode_length 392 = 全程站立不翻。

## 问题描述

1. **翻转从未触发**: 预热斜坡 `commands[:,4] *= alpha` 每步累乘 → alpha=0.99 乘 2400 步 ≈ e⁻²⁴ ≈ 0, trigger 被乘没, FSM 永远停在站立态。这是旧 jump_srl 就存在的 bug, 被复用。
2. **站立态 flip_progress 漂移**: 站立不稳(零动作 z 0.37↔0.64 振荡)导致 `-∫gyro[1]` 累积到 +3.29 rad, 污染翻转进度测量。

## 根因分析

| Bug | 影响 |
|-----|------|
| `*= alpha` 累乘 | trigger 指数衰减到 0, FSM 不触发, 全程站立 |
| flip_progress 无重置 | 站姿晃动累积成假翻转进度 |

## 解决方案

1. **预热改为不污染原始 trigger**: 只在 FSM 决策时缩放 `jt = commands[:,4] * warmup_progress`, 每次从原始采样值重算, 不累乘。
2. **站立态重置 flip_progress**: `fsm==-1` 时清零 flip_progress/delta。

## 修改文件

| 文件 | 内容 |
|------|------|
| `src/unilab/envs/locomotion/xqrobotwl/backflip.py` | update_state: 预热改为 warmup_progress 缩放不累乘; FSM 调用用缩放 trigger; 站立态重置 flip_progress |

## 验证方法

测试 env (64 env, warmup_iters=10, 模拟 30 iters):
- 预热 0→0.5 时 44/64 env 开始翻转 (FSM 离开站立态) ✅
- 纯 ff 翻转达 **+5.37 rad (86% of 360°)** ✅
- 站立态 flip_progress=0, 无漂移 ✅

## 评估结果

修复后翻转正常触发, 纯 ff 达 86% 翻转 (与 P1 中期一致)。剩余 14% + 落地由 PPO 学习。

## 后续计划

- 重跑 quick 200 iter, 确认 flip_progress 奖励非 0、flip_complete 出现
- 确认后开全量训练

## 关联日志

- [02_env_development](2026-08-04/02_env_development.md) — 环境开发, 预热 bug 源于此
- [12_sign_bugs_entropy_fsm_fixes](../jump/ppo/2026-07-24/12_sign_bugs_entropy_fsm_fixes.md) — 跳跃 FSM 时序修复, 预热累乘是同类问题
