# 13 — 接触式终止: 机身/小腿碰地立即终止

## 日期

2026-08-04

## 来源

用户要求: 不止倾倒终止, 小腿碰地、机身碰地都要终止, 和其他配置终止一样。

## 问题描述

之前终止用 0.4s 持续判定, 但机器人贴地(z=0.08)后 0.3s 就触发下一次翻转, 从未达到 0.4s → 永不终止。

## 解决方案

1. **机身/小腿碰地立即终止**: `base_z < min_base_height(0.20)` 在非飞行态立即终止 (匹配其他配置)
2. **倾倒**: tilt > 50° 持续 0.15s 终止 (0.4→0.15, 更严)
3. **min_base_height**: 0.15→0.20 (碰地判定更敏感)

## 修改文件

| 文件 | 内容 |
|------|------|
| `src/unilab/envs/locomotion/xqrobotwl/backflip.py` | base_z 终止去持续, 立即; tilt 持续 0.4→0.15s |
| `conf/ppo/task/xqrobotwl_backflip_flat/mujoco.yaml` | min_base_height 0.15→0.20 |

## 验证方法

续训后: 机器人翻转落地若 z<0.20 (小腿/机身碰地) 立即终止, 不再贴地停留。

## 关联日志

- [12_max_episode_6s](2026-08-04/12_max_episode_6s.md) — max_episode 调整
