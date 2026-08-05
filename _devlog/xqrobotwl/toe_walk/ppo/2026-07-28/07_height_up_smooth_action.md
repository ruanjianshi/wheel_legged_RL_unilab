# 07 — 点足高度提升 + 动作柔化

## 日期
2026-07-28

## 来源
评估最新 toe_walk model_9999 (reward=167)，发现两个问题：
1. 蹲姿过低 (h=0.38m)，抬腿时膝盖碰到机身主体
2. 抬腿动作幅度过大、不平滑

## 根因分析

### 问题 1：高度太低

`base_height_target=0.55` 但 `base_height=-60` 在 0.38m 只罚 -1.73/步，而 `phase_swing_lift=30` 奖 14.3/步。策略学会了"弯膝蹲+使劲抬腿"的廉价方案，从未被逼发明"站直也能抬腿"的运动学方案。

### 问题 2：抬腿不平稳

`action_scale=0.25` 每步可动 14°，配合 `phase_swing_lift=30` 的强大激励，策略在摆腿期膝部以最大关节速度猛抬，不做平滑过渡。

## 解决方案

| 参数 | 旧 | 新 | 原因 |
|------|----|----|------|
| `base_height` | -60 | **-150** | 0.38m 罚 -4.3/步 (原 -1.73)，策略必须站直 |
| `action_scale` | 0.25 | **0.18** | 每步动 ≤10°，多步平滑 ← 核心 |
| `joint_action_rate` | -0.03 | **-0.08** | 加倍惩罚关节急变 |

## 修改文件

| 文件 | 行 | 改动 |
|------|----|------|
| `conf/ppo/task/xqrobotwl_toe_walk_flat/mujoco.yaml:44` | action_scale: 0.25 → 0.18 |
| `conf/ppo/task/xqrobotwl_toe_walk_flat/mujoco.yaml:76` | base_height: -60 → -150 |
| `conf/ppo/task/xqrobotwl_toe_walk_flat/mujoco.yaml:78` | joint_action_rate: -0.03 → -0.08 |

## 预期效果

1. height 从 0.38m 回到 0.48-0.52m（峰值 0.49m 已经证明可达）
2. 抬腿时膝盖不再碰机身（有 5cm+ 余量）
3. 动作从"bang-bang 猛抬"转为"多步平缓抬落"
4. `phase_swing_lift` 可能从 14.3 短暂下降到 10-12，但这是合理tradeoff

## 后续计划
等待训练完成，重新评估 height、swing_lift、knee 角度变化。
