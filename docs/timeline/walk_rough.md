# 时间线 · 粗糙地形行走 (xqrobotwl_walk_rough)

> 一句话: 两轮足机器人在碎石/波浪/斜坡等地形自适应行走, 支持高度命令调腿长 — 已达标 (10000 iter, 验证通过, 地形通过率量化待补)。
> 来源: _devlog/xqrobotwl/{walk_rough,rough}/ppo/  (实时更新, 每完成一个阶段追加一行)

| 日期 | 阶段 | 做了什么 | 影响/效果 | 问题与解决 |
|---|---|---|---|---|
| 2026-07-12 | 起步 | 新建 `XqRobotWLWalkRough` PPO 配置 (1024 envs × 25 steps, 10000 iter, 地形=平地20%+粗糙35%+波浪35%+斜坡10%), Vy=0, tracking=4.0, hip_roll=-4.0 `[[01_rough]]` | — | — |
| 2026-07-22 | 地形增强 + 高度课程 | 难度提升 (flat 20%→5%, slope 5%→20%×2 且 0.1-0.35→0.15-0.50, rough 0.005-0.04→0.02-0.08m, wave 0-0.12→0.03-0.18m); base_height 奖励目标改为命令高度 (commands[4]); 新增高度课程随 vx/vyaw 展开 `[[rough/01_rough_terrain_height_curriculum]]` | 策略学会根据高度命令动态调腿长 (伸腿站高/屈膝下蹲), 高度维度课程 iter 0 窄 [0.55±0.03] → 全开 [0.40,0.90] | — |
| 2026-07-22 | 高度惩罚修复 | `base_height` -5→-30, max_episode_seconds 20→10 `[[rough/02_height_penalty_fix]]` | 站立高度正确、Q/E 高度调节生效 | 原 -5 太弱 (平坦任务用 -60), 模型对高度命令不敏感 |
| 2026-08-10 | 全量重训 | 8 任务批量重训 (1024 envs), rough 跑满 10000 iter | mean_reward **42.05**, ep_len 914, action_std 0.15, SPS 8555; checkpoint `logs/rsl_rl_ppo/XqRobotWLWalkRough/2026-08-10_01-26-08_mujoco/model_9999.pt` | eval 冒烟通过, 无回归 |

## 当前状态
- 最新 checkpoint: `model_9999.pt` (2026-08-10, 10000/10000 iter, 9h56m)
- 关键指标: mean_reward 42.05 / ep_len 914 / action_std 0.15 / 平均 SPS 8555
- 达标情况: ✅ 已训 (thesis/experts/02) — viewer 验证正常; 各地形通过率、姿态角标准差等验收指标待补

## 下一步
- [ ] 验收: 各地形 (粗糙/碎石/斜坡) 通过率、姿态角标准差 (见 thesis 实施计划 §阶段一)
- [ ] 论文待补: 与滚动行走专家的速度跟踪对比
