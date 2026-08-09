# 11 — 修正需求: 后空翻后"不倒"即可, 不要求"站直"

## 日期

2026-08-04

## 来源

用户明确: "我要的后空翻后，不倒，不是站直"。之前把目标设成站直(z=0.65)太严, 机器人做不到。

## 问题描述

- flip_complete 需 z>0.4, stand_height 目标 0.65 (站直) → 太难, 机器人折叠贴地(z=0.08)
- 用户只要"不倒" (轮子撑住, z>0.25), 不需要站直 0.65

## 解决方案

1. **stand_height 改"不倒"奖励**: `clip((z-0.25)/0.2, 0, 1)` — z=0.25→0, z=0.45→1(封顶), 不要求 0.65
2. **flip_complete 改 z>0.25**: 轮子撑住不折叠即可, 不要求 z>0.4

## 修改文件

| 文件 | 内容 |
|------|------|
| `src/unilab/envs/locomotion/xqrobotwl/backflip.py` | stand_height 改不倒奖励, flip_complete 改 z>0.25 |

## 验证方法

续训后: 后空翻完 z>0.25 (轮子撑住), 不折叠贴地。

## 后续计划

- resume 续训, 渲染验证
- 若 z>0.25 达成率高 → 满足用户需求, 进入 P4

## 关联日志

- [10_flip_complete_height_requirement](2026-08-04/10_flip_complete_height_requirement.md) — 前序(站直目标, 本次放宽)
