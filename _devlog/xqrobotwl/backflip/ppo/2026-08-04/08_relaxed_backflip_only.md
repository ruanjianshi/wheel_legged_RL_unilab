# 08 — 放宽要求: 只训练后空翻 (落地不倒即可, 不追求完美平衡)

## 日期

2026-08-04

## 来源

用户反馈: 不用续训 walk_flat, 只训练后空翻。要求: 机器人能完成后空翻 + 落地不倒即可, 不要求完美站立平衡。

## 问题描述

之前的平衡 vs 翻转对抗: warm-start 后站立平衡好但翻转未完成 (最大 1.8 rad)。用户放宽要求, 简化任务。

## 解决方案

1. **不用 walk 热启动** (回到从零训, 之前从零训翻转是好的)
2. **只训练后空翻**: `flip_trigger_prob` 0.5→**1.0** (所有 episode 都翻转)
3. **放松站立要求**: `stand_balance` 改温和 `up` 奖励 (45°→0.7, 只奖不倒), 无倾角惩罚
4. **放松终止**: `max_tilt_deg` 45→**75** (真摔才终止, 45-75° 不算倒)
5. **翻转奖励加强**: flip_progress 80, flip_complete 100 (落地直立阈值放宽到 up>0.6 = 53°)

## 修改文件

| 文件 | 内容 |
|------|------|
| `src/unilab/envs/locomotion/xqrobotwl/backflip.py` | stand_balance 温和化, flip_complete 阈值 0.9→0.6 |
| `conf/ppo/task/xqrobotwl_backflip_flat/mujoco.yaml` | flip_trigger_prob=1.0, max_tilt_deg=75 |

## 验证方法

env 构造 OK。从零训 (full 模式, 不用 warmstart), 验证:
- flip_complete 出现 (翻转+落地不倒)
- 最大翻转 ≥ 5.98 rad (360°)
- 落地后 tilt < 75° 保持

## 后续计划

- full 从零训练 10000 iter
- 键盘验证: H 触发后空翻 → 落地不倒

## 关联日志

- [07_warmstart_from_walk](2026-08-04/07_warmstart_from_walk.md) — 之前的热启动尝试
