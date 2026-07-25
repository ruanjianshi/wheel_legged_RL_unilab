# 12 — SRL 跳跃：符号 bug、entropy、FSM 时序全面修复

## 日期

2026-07-24

## 来源

SRL 跳跃训练连续 4 轮 action_std 单调上涨（0.30→1.5+），策略完全不收敛。

## 问题描述

1. **entropy_coef=0.03** — 太平。loss 中 `-entropy_coef * entropy` 奖励策略保持高随机性，导致 action_std 只涨不降。
2. **fsm_tracking reward 符号反** — weight=-5.0，函数返回负值 → 负×负=正 → 在**奖励**不跟 FSM。
3. **knee_ok 阈值 0.8** — 蹬地时膝角 -0.87 > |0.8|，`jump_height` reward 被误杀。
4. **蹲姿 reward 窗口 (1-40/1-25) 覆盖了蹬地阶段** — FSM 在 phase~8 就切到 push，但 crouch_prep 还在要求蹲 → 冲突信号。
5. **vertical_thrust 窗口 phase≥25 太晚** — FSM push 在 phase~8 已开始。
6. **FSM 过渡太快** — crouch 0.15s、push 0.15s，PPO 跟不上膝角从 +0.70 到 -0.87 的跳变。

## 根因分析

| Bug | 影响 |
|-----|------|
| entropy_coef=0.03 | 策略被奖励去随机，action_std 必然涨 |
| fsm_tracking sign | 罚变成奖，策略远离 FSM 目标 |
| knee_ok <0.8 ` | 蹬地时 jump_height reward=0，策略失去核心梯度 |
| crouch/push 窗口重叠 | 同一步骤被要求蹲和蹬，梯度互相抵消 |
| FSM 0.15s 过渡 | 目标位置变化太快，value function 噪声→策略发散 |

## 解决方案

1. `entropy_coef: 0.03 → 0.005`（标准 PPO 值）
2. `fsm_tracking: -5.0 → 5.0`（正 weight × 负函数 = 真罚）
3. `knee_ok <0.8 → <1.2`（蹬地膝角 -0.87 不被误杀）
4. crouch_prep 窗口: 1-40 → **1-15**，crouch_depth: 1-25 → **1-12**，thrust: ≥25 → **≥8**
5. **FSM 时序减速**: crouch 0.15s→**0.25s**，push 0.15s→**0.20s**
6. landing_soft 窗口: ≥30 → **≥35**（对齐 FSM 着陆点）

## 修改文件

| 文件 | 行 | 改动 |
|------|-----|------|
| `conf/ppo/task/xqrobotwl_jump_srl_flat/mujoco.yaml` | 46 | entropy_coef: 0.03→0.005 |
| `conf/ppo/task/xqrobotwl_jump_srl_flat/mujoco.yaml` | 94 | fsm_tracking: -5.0→5.0 |
| `src/unilab/envs/locomotion/xqrobotwl/jump_srl.py` | 256 | knee_ok: 0.8→1.2 |
| `src/unilab/envs/locomotion/xqrobotwl/jump_srl.py` | 157 | crouch_prep active: 1-40→1-15 |
| `src/unilab/envs/locomotion/xqrobotwl/jump_srl.py` | 183 | crouch_depth active: 1-25→1-12 |
| `src/unilab/envs/locomotion/xqrobotwl/jump_srl.py` | 229 | vertical_thrust active: ≥25→≥8 |
| `src/unilab/envs/locomotion/xqrobotwl/jump_srl.py` | 82-89 | FSM crouch timer: 0.15→0.25, push: 0.15→0.20 |
| `src/unilab/envs/locomotion/xqrobotwl/jump_srl.py` | 267 | landing_soft active: ≥30→≥35 |

## 验证方法

训练 iter 596→1705→5317 系列指标：
- entropy fix 后 action_std 稳定在 0.34→0.59（不再 0.30→1.5 爆炸）
- fsm_tracking 从 +7.0（奖错了）变为 -2.87（正确罚）
- jump_height 从 0.021→0.39→2.89，翻 140 倍
- ep_len 稳定在 750-920

## 后续计划

- FSM 前馈 gain 从 1.0 降到 0.15（减少对抗）
- 移除 dead `anti_loiter`，加 `FSM_tracking`、`height_progress`
- 站姿后仰问题待修（后续 lean_forward trigger-gating）

## 关联日志

- [08_wheeled_srl_framework_diag](2026-07-22/08_wheeled_srl_framework_diag.md) — 诊断
- [11_srl_convergence](2026-07-22/11_srl_convergence.md) — 前序失败
- [13_lean_forward_final](2026-07-25/13_lean_forward_final.md) — 后续修复
