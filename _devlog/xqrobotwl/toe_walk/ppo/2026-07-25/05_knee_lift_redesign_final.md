# 05 — 点足行走：knee_lift 解锁 + 课程崩溃 + 论文驱动重设计

## 日期

2026-07-24 ~ 2026-07-25

## 来源

修复 5 个 bug 后训练 iter 5000+，`phase_knee_lift=5.87`（膝盖弯了），但 `ep_len=10`、`phase_swing_lift=0.015`（轮子始终没离地）。i3000 进入 Stage 2 课程后 ep_len crash 到 5。

## 问题描述

### 问题 1: phase_knee_lift 要求轮子先离地（鸡生蛋）
```python
l_bend = clip(knee - 0.15, 0, 1) * left_swing * l_air  # ← 要求 l_air=1
```
策略必须**先**让轮子离地才能拿弯膝奖励。但弯膝是离地的前提 → 策略卡死在"不弯膝=不罚，不罚=不探索"的死循环。

### 问题 2: 课程 ramp 太陡
Stage 1 (0-3000 iter): penalty=0.1x，Stage 2 (3000-6000): ramp 0.1→1.0x。
iter 3000 切换时 penalty 从 2.5 跳到 25（8 倍），策略当场崩溃：ep_len 27→5。

### 问题 3: 多奖励冲突
15+ 个奖励项（含 wheel_balance, leg_mirror, hip_roll, joint_action_rate）全部惩罚抬腿所需的平衡修正动作。

## 根因分析

### 鸡生蛋问题
`phase_knee_lift` 的 `l_air` 门控本质上是"先解离地，再奖励弯膝"。但离地前的中间状态（弯膝+轮子仍着地）被 swing_contact_penalty 罚 → 策略宁愿不弯。

### 课程崩溃
St 2 penalty 8 倍跳跃式增长+vel tracking 同时开启，两个新压力叠加。

## 解决方案

### 1. knee_lift 解锁（07-24）
移除 `l_air/r_air` 门控 → 弯膝即奖。添加 `phase_knee_stance` 防止双侧蹲。

### 2. 课程平滑化（07-24→07-25）
- `curriculum_steps`: 3000→4000（Stage 1 多 1000 iter 学弯膝）
- `swing_contact_penalty`: 25→10→**5**（三降，最终 5）

### 3. 论文驱动重设计（07-25）

**读取三篇参考论文**：

| 来源 | 核心洞察 | 应用到 xqrobotwl |
|------|----------|------------------|
| **BoltLocomotion (CaT)** | 太多奖励=冲突。只用 2 项 + 约束门控 | 删除 4 个冲突奖励 |
| **tron1** | action_scale=0.25，罚>奖，num_envs=4096 | action_scale 减半，lift 权重↑ |
| **TitaRL** | 动作平滑 0.8*new+0.2*old | 已有 0.7*new+0.3*old |

**具体改动**：
- **删除 4 项**: wheel_balance, wheel_symmetry, feet_distance, soft_landing
- **弱化 4 项**: leg_mirror(-0.5→-0.2), hip_roll(-2.0→-0.5), joint_action_rate(-0.1→-0.03), wheel_action_rate(-0.005→-0.001)
- **加强 3 项**: orientation(-20→-30), phase_swing_lift(20→30), phase_knee_lift(10→15)
- **稳定 3 项**: action_scale(0.5→0.25), max_tilt(60→**90**), init_noise(0.7→0.4)

## 修改文件

| 文件 | 行 | 改动 |
|------|-----|------|
| `toe_walk.py` | 150-162 | knee_lift: 移除 l_air/r_air 门控，新增 knee_stance 函数 |
| `toe_walk.py` | 59 | curriculum_steps: 3000→4000 |
| `toe_walk.py` | 303-308 | apply_action: 添加 0.7*new+0.3*old 平滑 |
| `mujoco.yaml` | 44-92 | 大规模 reward scale 重写（见上表） |
| `mujoco.yaml` | 44 | action_scale: 0.5→0.25 |
| `mujoco.yaml` | 98 | max_tilt_deg: 60→90 |
| `mujoco.yaml` | 32 | init_noise_std: 0.7→0.4 |

## 评估结果 (iter 9999, final)

| 指标 | 旧 run (iter 9999) | 新 run (iter 9999) | 改善 |
|------|-------------------|---------------------|------|
| reward | 2.8 | **167** | 60x |
| ep_len | 25 | **922** | 37x |
| **phase_swing_lift** | 2.68/20=0.13 | **14.34/30=0.48** | 轮子 48% 摆地离地 |
| phase_knee_lift | 6.83/10=0.68 | 9.12/15=0.61 | 膝弯稳定 |
| action_std | 0.27 | **0.15** | 完全收敛 |
| swing_contact_penalty | -7.22/10 | -1.93/5 | penalty 可控 |
| orientation | -0.87 | -0.96/30=0.03 | tilt 近乎 0 |
| height_err | 0.09m | 0.16m | 可接受 |

## 后续计划

- 本次训练为最终版。模型已备份至 `backup/XqRobotWLToeWalkFlat/toe_walk_ppo_v1/`。
- 论文用此版证明点足抬腿行走可行性。

## 关联日志

- [04_sign_bugs_and_termination_fixes](2026-07-23/04_sign_bugs_and_termination_fixes.md) — 前序修复
- [03_phase_gated](2026-07-22/03_phase_gated.md) — 相位门控设计
