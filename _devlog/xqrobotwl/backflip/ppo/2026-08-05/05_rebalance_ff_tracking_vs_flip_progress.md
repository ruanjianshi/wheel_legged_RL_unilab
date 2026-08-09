# 05 — 奖励重平衡: ff_tracking 10 / flip_progress 30 (防刷旋转)

**日期**: 2026-08-05
**来源**: 04 修复后快速训练验证 (1000 iter) — flip_progress 生效但 flip_complete 仍 0
**关联**: [04_flip_complete_event_latch](2026-08-05/04_flip_complete_event_latch.md)

---

## 问题描述

04 修复后, 1000 iter 快速训练:
- `flip_progress` 奖励从 0 → 非 0 (最高 7.2/iter): 机器人**确实在翻转**, 测量修复生效 ✅
- `flip_complete` 奖励**仍恒为 0**: 策略学会转但没学会转完落地直立
- `mean_episode_length` ~156 (1.56s): 恰在翻转完成点附近终止 (落地摔)

## 根因分析

对照实验 (256 env, 500 步) 证明锁存机制本身正常:

| 策略模式 | flip_complete 触发 | 总奖励 |
|---------|------------------|--------|
| 纯 ff (策略=0) | 263 | 2504 |
| 小噪声 (std 0.05) | 279 | 2470 |
| 飞行期大动作 (模拟刷分) | **43** | **536** |

**结论**: 策略在飞行期**偏离已验证的 ff 轨迹**去"刷旋转"(spin-hacking)。根因是奖励失衡:
- `flip_progress: 80` (稠密, 每飞行步) 激励旋转速率, 不约束动作质量
- `ff_tracking: 1.0` 太弱, 挡不住策略偏离 ff 去刷分
- 结果: 翻转乱 (甩腿/过度旋转) → 落地摔 → `flip_complete` (需转完+落地直立) 永远够不着

这正是 devlog 03 (OPT-Mimic) 预判的 BAV 奖励缺陷 — 但没有给足约束权重。

## 解决方案

奖励权重重平衡 (飞行期让 ff 主导, 策略专注站立/落地):

| 权重 | 旧 | 新 | 理由 |
|------|----|----|------|
| `flip_progress` | 80 | **30** | 削弱刷旋转动机; ff 已负责翻转, 策略不需要靠它学翻转 |
| `ff_tracking` | 1.0 | **10** | 飞行期把策略钉在 P1 验证的干净 ff 轨迹上 → 翻转稳定完成 → flip_complete 可达 |

`ff_tracking` 只在飞行态(1/2/3)激活, 站立/落地期策略仍有完全自由度。

## 修改文件

| 文件 | 修改内容 |
|------|---------|
| `conf/ppo/task/xqrobotwl_backflip_flat/mujoco.yaml` | flip_progress 80→30, ff_tracking 1.0→10 (+注释) |

## 验证方法

- 新 run 快速验证 (1000 iter): `flip_complete` 应从 0 变非 0
- 渲染: 翻转干净 (不甩腿/不麻花), 落地后站起来

## 评估结果

> 2026-08-05 A/B 对照实验 (各 1000 iter) 结论: **重平衡是负结果, 已恢复基线权重**

| 指标 | 基线 (ff_tracking=1.0, flip_progress=80) | A/B (ff_tracking=10, flip_progress=30) |
|------|------------------------------------------|----------------------------------------|
| 首次 flip_complete | ~650 | ~513 |
| flip_complete 稳定幅度 | **0.45-0.75** | 0.02-0.03 (偶发) |
| mean_reward | 7-12 | ~2 |
| ep_len | ~155 | ~50-100 |

- **根因**: `ff_tracking=10` 过强 — 策略在飞行期任何偏离 ff 的修正动作都被重罚 (-0.9/步), 学不会翻转修正/落地 → ep_len 卡 50, flip_complete 极弱
- 基线 (ff_tracking=1.0) 本身就能收敛 (spin-hacking 是暂态, ~650 后 flip_complete 稳定触发), **无需重平衡**
- **决定**: 配置恢复基线权重 (flip_progress=80, ff_tracking=1.0), 基线续训到更长做稳策略

## 后续计划

- [ ] 新权重快速训练, 确认 flip_complete > 0
- [ ] 达标后渲染 + 收紧终止 (分阶段方案C 第二阶段)
- [ ] 若 flip_complete 仍 0, 降 flip_progress 或加 ff_gain 课程

---

*记录人: AI | 审核: xiaoq*
