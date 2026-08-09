# 10 — flip_complete 加高度条件 (防小腿贴地取巧)

## 日期

2026-08-04

## 来源

用户渲染视频观察: 后空翻后小腿膝关节碰地面 — 机器人落地 z=0.1 (贴地) 但倾斜 <50° 也能拿 flip_complete 奖励 (条件只查 up>0.6=倾斜<53°)。

## 问题描述

- 翻转完成 (flip=6.49 rad) ✅
- 但落地 z=0.1 (小腿贴地), 倾斜 13-30°, 腿外展
- flip_complete 条件没查高度 → 贴地也算"完成" → 策略取巧不撑腿

## 解决方案

1. **flip_complete 加高度条件**: `base_height > 0.4` — 必须撑起来站高才算完成, 小腿贴地不给奖励
2. **stand_height 3→5**: 加强站直惩罚

## 修改文件

| 文件 | 内容 |
|------|------|
| `src/unilab/envs/locomotion/xqrobotwl/backflip.py` | flip_complete 加 z>0.4 条件 |
| `conf/ppo/task/xqrobotwl_backflip_flat/mujoco.yaml` | stand_height 3→5 |

## 验证方法

续训后渲染视频确认: 后空翻完 → 撑腿站直 (z>0.4), 小腿离地, 不贴地。

## 后续计划

- resume 续训, 渲染验证
- 若仍贴地: 再加强 (flip_complete 高度条件提高 / min_base_height 提高)

## 关联日志

- [09_anti_cheat_posture_fix](2026-08-04/09_anti_cheat_posture_fix.md) — 前序防取巧修复
