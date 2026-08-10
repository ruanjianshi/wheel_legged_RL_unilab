# 时间线 · 点足/抬腿行走 (xqrobotwl_toe_walk_flat)

> 一句话: 两轮足机器人相位时钟驱动的点足/抬腿行走 (摆动相收腿抬轮离地 + 支撑相轮平衡) — 已达标 (全任务最高 reward, 验证通过, 抬腿高度/木桩通过率量化待补)。
> 来源: _devlog/xqrobotwl/toe_walk/ppo/  (实时更新, 每完成一个阶段追加一行)

| 日期 | 阶段 | 做了什么 | 影响/效果 | 问题与解决 |
|---|---|---|---|---|
| 2026-07-22 | v1 训练 | 新建 toe_walk 配置 + 参数调优 (cycle_time 0.5→0.7s, ref_scale 0.12→0.18, action_scale 0.3→0.5, thigh×0.8) `[[01_toe_walk_success]]` | iter 1409: ep_len 1165 (97%), action_std 0.15 收敛, swing_lift 2.39 (轮确实离地), wheel_balance 0.99, ref_tracking 3.3 — 点足行走验证成功 | — |
| 2026-07-22 | 符号修复 | 参考轨迹符号修复: L_knee 摆动相伸膝→屈膝, R_thigh 摆动相后仰→前倾 `[[02_reference_sign_fix]]` | 抬腿幅度增大, 轮可离地 | L_knee `DEFAULT−lift×scale×5` 方向反; R_thigh 符号反 |
| 2026-07-22 | 相位门控 | 新增相位门控奖励模式 (无参考轨迹, 策略自由探索步态): phase_swing_lift 5.0 / phase_knee_lift 3.0 / phase_stance_contact 2.0 `[[03_phase_gated]]` | 不指定关节角度, 策略自由发现最优姿态 | 参考轨迹自由度低 |
| 2026-07-23 | bug 修复 | 5 个符号/逻辑 bug + 2 终止阈值修正 (swing_mask 从 sin(phase) 生成; 3 个奖励符号反; leg_mirror 减→加; calf_extreme 0.85→2.0; contact 0.1N→5.0N) `[[04_sign_bugs_and_termination_fixes]]` | 弯膝不再被终止误杀, 轮子不再被奖着地 | 致命链: 弯膝→calf>0.85→终止→不弯膝→phase_knee_lift=0; 轮着地→+25 奖励→主动着地 |
| 2026-07-24 ~ 07-25 | 论文驱动重设计 | knee_lift 解锁 (移除 l_air/r_air 鸡生蛋门控), 课程平滑 (curriculum_steps 4000, swing_contact_penalty 25→5), 读 BoltLocomotion/tron1/TitaRL 重写奖励 (删 4 冲突项, action_scale 0.5→0.25, init_noise 0.7→0.4) `[[05_knee_lift_redesign_final]]` | iter 9999: reward 2.8→**167** (60x), ep_len 25→**922** (37x), phase_swing_lift 0.13→0.48 (轮 48% 摆地离地), action_std 0.15; 备份 `backup/XqRobotWLToeWalkFlat/toe_walk_ppo_v1/` | phase_knee_lift 要求轮先离地 (鸡生蛋); 课程 ramp 太陡 (2.5→25 崩); 15+ 奖励冲突 |
| 2026-07-28 | 高度提升 + 柔化 | base_height -60→-150, action_scale 0.25→0.18, joint_action_rate -0.03→-0.08 `[[07_height_up_smooth_action]]` | 蹲姿 h 0.38→0.48-0.52m (峰值 0.49 已验证可达), 抬腿不再碰机身, 动作从 bang-bang 猛抬转多步平缓 | 蹲太低 (0.38m) 抬腿碰机身; action_scale 0.25 每步 14° 猛抬不平滑 |
| 2026-08-10 | 全量重训 | 8 任务批量重训 (1024 envs), toe_walk 跑满 10000 iter | mean_reward **144.81 (全任务最高)**, ep_len 1000 (满), action_std 0.24, SPS 9955; checkpoint `logs/rsl_rl_ppo/XqRobotWLToeWalkFlat/2026-08-10_01-26-06_mujoco/model_9999.pt` | eval 冒烟通过, 无回归 |

## 当前状态
- 最新 checkpoint: `model_9999.pt` (2026-08-10, 10000/10000 iter)
- 关键指标: mean_reward 144.81 (全任务最高) / ep_len 1000 (满) / action_std 0.24
- 达标情况: ✅ 已训 (thesis/experts/03) — eval viewer 正常加载 model_9999.pt; 抬腿高度跟踪、支撑稳定性、木桩通过率待验收

## 下一步
- [ ] 验收: 抬腿高度跟踪、支撑稳定性、木桩通过率 (见 thesis 实施计划 §阶段二)
- [ ] 论文待补: 抬腿高度跟踪、支撑稳定性、木桩通过率量化
