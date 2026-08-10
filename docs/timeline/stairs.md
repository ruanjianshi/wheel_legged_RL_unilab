# 时间线 · 上下楼梯 (xqrobotwl_stairs)

> 一句话: 两轮足机器人低台阶稳定上/下楼梯的约束步态 (NP3O) — 已达标 (10000 iter, 验证通过, 台阶通过率/约束满足量化待补)。
> 来源: _devlog/xqrobotwl/stairs/np3o/  (实时更新, 每完成一个阶段追加一行)

| 日期 | 阶段 | 做了什么 | 影响/效果 | 问题与解决 |
|---|---|---|---|---|
| 2026-07-12 | 起步 | 新建 `XqRobotWLStairs` NP3O 配置 (1024 envs × 25 steps, 10000 iter, 地形 100% 楼梯上下各 50%, cost_viol=0.01), stairs.py 继承 rough `[[01_stairs]]` | — | — |
| 2026-07-22 | 全局修正 | 统一 `base_height_target` 0.65→0.55 (xqrobotwl 腿长物理上限) `[[jump/10_stand_height_platform_fix]]` | 站立高度目标物理可达 | 0.65m 是 xqrobotV2 配置, xqrobotwl 上限约 0.55-0.60m |
| 2026-08-10 | obs 维度 bug 修复 | `XqRobotWLStairsEnv.__init__` 覆盖 `_obs_frame_dim=33` / `_critic_frame_dim=36` 并重建 history buffers `[[repo/04_launch_8_trainings_and_fix_path_obs_bugs]]` | stairs 训练可正常启动 (actor 297=33×9); obs_groups_spec = {obs: 297, critic: 511} | 07-25 后 stairs 训练从未成功: 继承 rough 的 4D 命令 (32 维) vs YAML 5D 命令 (33 维) → broadcast ValueError |
| 2026-08-10 | 全量训练 | 8 任务批量训练, stairs NP3O 跑满 10000 iter (1024 envs) | mean_reward **31.22**, ep_len 899, action_std 0.53, SPS 11213 (10h21m 最慢); checkpoint `logs/rsl_rl_np3o/XqRobotWLStairs/2026-08-10_01-31-09_mujoco/model_9999.pt` | eval 冒烟通过, 无回归 |

## 当前状态
- 最新 checkpoint: `model_9999.pt` (2026-08-10, 10000/10000 iter, NP3O)
- 关键指标: mean_reward 31.22 / ep_len 899 / action_std 0.53 / 平均 SPS 11213
- 达标情况: ✅ 已训 (thesis/experts/05 抬腿上台阶) — eval viewer 正常加载 model_9999.pt; 上下台阶成功率、力矩/姿态约束满足待验收
- 注: 楼梯为 NP3O 约束 RL, 继承 rough env, 观测维度 bug 于 2026-08-10 修复后才重新可训

## 下一步
- [ ] 验收: 上下台阶成功率、力矩/姿态约束满足 (见 thesis 实施计划 §阶段二)
- [ ] 论文待补: 上下台阶成功率、力矩/姿态约束量化
