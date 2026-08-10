# 06 — 站立平衡修复 (两轮足需主动平衡)

## 日期

2026-08-04

## 来源

用户键盘验证: 机器人后空翻前连站立都不稳, 模拟发散 (NaN/inf QACC warning)。确定性策略测试: 站立 0.4s 内摔倒 (tilt 98°), 最大动作幅度 4.26。

## 问题描述

1. **站立态无平衡奖励**: state -1 只有弱 posture_stand 惩罚, 无"保持直立/高度"奖励 → 策略没学主动平衡
2. **动作幅度失控**: 策略输出 4.26 (远超合理), 关节顶到极限 → 无扭矩上限的仿真发散 NaN
3. **物理本质**: 两轮足是倒立摆, **天生不稳定**, 必须靠轮子主动滚动平衡 — 不存在静态 DEFAULT_LEG_ANGLES 平衡点 (推翻 P0-1 静态修复假设)

## 根因分析

| 问题 | 根因 |
|------|------|
| 站不稳 | state -1 无平衡奖励, 策略没动机学平衡 |
| 仿真发散 | 动作 4.26 顶爆关节 + 无扭矩上限 + clip_actions=100 |
| 走路能稳 | walk 策略学了轮平衡, backflip 训练没学 |

## ★ 关键发现 (V2 walk 对照)

加载 backup `XqRobotV2WalkFlat/model_3700.pt` 测试: vx=0 站立能平衡 (倾角 0-37° 振荡恢复, 保持 z=0.65)。**V2 walk 策略输出巨大动作 (腿 max=5.36) 平衡, clip_actions=100 不裁**。而 backflip 之前 `clip_actions=3.0 + action_magnitude=-0.1` **压掉了平衡所需的大动作权限** → 这才是站不稳的真正原因。

## 解决方案

1. **加 `stand_balance` 奖励** (state -1/5): 迭代历程 — 苛刻版 `up-5*h_err-2*wobble²` (mean_reward -160 崩) → 温和版 `up` (梯度弱, 学不好) → **当前 `up + 2*orient_pen`** (15°→0.85, 45°→-0.29, 60°→-1.0, 梯度强), scale=5.0
2. **恢复动作权限** (对齐 V2 walk): `clip_actions` 3.0→**100** (平衡需大动作), `action_magnitude` -0.1→**-0.02**
3. **base_height_target** 0.55→**0.65** (对齐 V2 walk 保持高度)
4. **站立:翻转 = 5:5** (flip_trigger_prob 0.7→0.5)

## 修改文件

| 文件 | 内容 |
|------|------|
| `src/unilab/envs/locomotion/xqrobotwl/backflip.py` | 加 `_reward_stand_balance` (up+2*orient_pen) + 注册 |
| `conf/ppo/task/xqrobotwl_backflip_flat/mujoco.yaml` | stand_balance=5.0, clip_actions=100, action_magnitude=-0.02, base_height_target=0.65, flip_trigger_prob=0.5 |

## 迭代记录

- 300 iter 验证 (苛刻版): stand_balance 全程负 (-15→-3.3), mean_reward -160, ep_len 144 → 奖励太苛刻, 策略学不会平衡
- 改为温和版 `up` 奖励后重验: 直立时 +5, 摔倒时 0 (无负惩罚), 梯度更可学

## 验证方法

env 构造 OK, clip_actions=3.0, stand_balance 注册, 站立态奖励 0.95 (正值)。重训后验证:
- 站立 (state -1, 无翻转指令) 能保持 upright 2s+
- 翻转后能恢复站立
- 无 NaN

## 后续计划

- 重训全量 10000 iter
- 键盘验证: 站立稳 → H 翻转 → 落地站稳

## 关联日志

- [05_play_mode_keyboard_control](2026-08-04/05_play_mode_keyboard_control.md) — 键盘验证暴露问题
- [04_exploration_std_explosion_fix](2026-08-04/04_exploration_std_explosion_fix.md) — 训练收敛
