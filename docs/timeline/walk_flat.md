# 时间线 · 平地滚动行走 (xqrobotwl_walk_flat)

> 一句话: 两轮足机器人平地稳定站立/前进后退/转向/侧移的滚动行走步态 — 已达标 (10000 iter, 验证通过, 验收指标待补)。
> 来源: _devlog/xqrobotwl/walk_flat/ppo/  (实时更新, 每完成一个阶段追加一行)

| 日期 | 阶段 | 做了什么 | 影响/效果 | 问题与解决 |
|---|---|---|---|---|
| 2026-07-12 | 起步 | 基于 xqrobotV2 创建 xqrobotwl, 新建 `XqRobotWLWalkFlat` PPO 配置 (1024 envs × 24 steps, 10000 iter, 平地 plane), 沿用 joystick.py 奖励体系 `[[01_flat]]` | — | — |
| 2026-07-22 | 全局修正 | 统一 `base_height_target` 0.65→0.55 (xqrobotwl 腿长物理上限 ~0.55-0.60m, 0.65 是 xqrobotV2 配置) `[[jump/10_stand_height_platform_fix]]` | walk 实测自然站立 median z≈0.518-0.52m, 与目标 0.55 对齐 | 0.65m 目标物理不可达; DEFAULT_LEG_ANGLES 在 xqrobotwl 上非稳定平衡点, 零动作下自然振荡 |
| 2026-07-23 | 训练 | 训练至 model_9999 (run `2026-07-23_19-29-36_mujoco`) | 站立倾角稳定 2-7°, 被后空翻/跌倒恢复用作 warmstart 基座 | — |
| 2026-08-10 | 全量重训 | 8 任务批量重训 (1024 envs), walk_flat 跑满 10000 iter | mean_reward **22.89**, ep_len 896, action_std 0.15, SPS 15179; checkpoint `logs/rsl_rl_ppo/XqRobotWLWalkFlat/2026-08-10_01-25-14_mujoco/model_9999.pt` | eval 冒烟通过, 无回归 |

## 当前状态
- 最新 checkpoint: `model_9999.pt` (2026-08-10, 10000/10000 iter, 5h53m)
- 关键指标: mean_reward 22.89 / ep_len 896 / action_std 0.15 / 平均 SPS 15179
- 达标情况: ✅ 已训 (thesis/experts/01) — viewer 验证正常; 速度/航向跟踪误差、侧移精度等验收量化指标待补

## 下一步
- [ ] 验收: 站立时长、速度跟踪误差、侧移精度 (见 thesis 实施计划 §阶段一)
- [ ] 论文待补: 速度/航向跟踪误差量化 + 侧移精度验证
