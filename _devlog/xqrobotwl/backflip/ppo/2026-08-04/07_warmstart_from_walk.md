# 07 — walk 热启动 + 平衡/翻转对抗修复

## 日期

2026-08-04

## 来源

多次站立平衡训练不收敛 (0.3°→60°→45° 固定轨迹)。用户指出 walk-flat 能平衡, 检查发现 `logs/rsl_rl_ppo/XqRobotWLWalkFlat/2026-07-23_19-29-36_mujoco/model_9999.pt` 站立倾角稳定 2-7°。

## 问题描述

1. backflip 从零训平衡不收敛: 三种奖励设计都学到"晃到 45-60°"的固定轨迹
2. 手调动态控制器也无法平衡 (30-180° 不一致)
3. 对照 V2 walk: 策略输出 5.36 大动作平衡, **clip_actions=100 不裁** → 之前 backflip clip=3 + action_magnitude=-0.1 压掉了平衡权限

## 解决方案

1. **恢复动作权限**: clip_actions 3→100, action_magnitude -0.1→-0.02 (对齐 V2 walk)
2. **stand_balance 奖励**: `up + 2*orient_pen` (15°→0.85, 45°→-0.29), scale=5
3. **warm-start**: 用 walk_flat 模型热启动 backflip (`warmstart_from_walk.py`): actor 前 297 维继承 walk 权重, 后 27 翻转特征置零
4. **结果**: 站立平衡彻底解决 (stand_balance 全程 3-4, 倾角 1-7°), 但**翻转未完成** (最大 1.8 rad=103°, flip_complete=0) — 继承的平衡本能阻止翻转

## 修改文件

| 文件 | 内容 |
|------|------|
| `scripts/xqrobotwl/warmstart_from_walk.py` | 新建。walk 权重热启动 backflip actor |
| `conf/ppo/task/xqrobotwl_backflip_flat/mujoco.yaml` | clip=100, action_magnitude=-0.02, base_height_target=0.65 |
| `shell/xqrobotwl/launch_ppo_backflip.sh` | 加 warmstart / resume 模式 |

## 验证方法

- 热启动 actor 站立: 倾角 1-7° ✅ (平衡继承成功)
- 10000 iter 续训: stand_balance 3-4 ✅, 但 flip_complete=0 ❌ (平衡 vs 翻转对抗)
- 对策: flip_progress 30→80, flip_complete 50→100 加强翻转奖励, resume 续训

## 后续计划

- resume 续训 (强翻转奖励), 看是否学成完整翻转
- 平衡(站立) vs 翻转的对抗是本任务核心难点

## 关联日志

- [06_standing_balance_fix](2026-08-04/06_standing_balance_fix.md) — 站立平衡迭代
- [05_play_mode_keyboard_control](2026-08-04/05_play_mode_keyboard_control.md)
